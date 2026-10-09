"""The 2026 Ssal-KRED ortholog activity data: the first measured labels here.

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE IN THE REPOSITORY
============================================================
Every label this project has benchmarked on so far was **annotation-derived**:
the SDR deposit's classes come from UniProtKB substrate and product annotations,
and "measuring" one in :mod:`eagent.eval.feedback_simulation` means revealing an
annotation. The honest ceiling on those results is stated everywhere they
appear.

These are assay results. 109 ortholog sequences were expressed, 69 of them
solubly; those 69 were run against five substrates and the products were
quantified. That makes this the first label set in the repository where a
positive means somebody detected a product.

It is also small, correlated and censored, and each of those has a
consequence that the loader enforces rather than mentions.

THE LABEL TYPES, WHICH DO NOT POOL
==================================
* ``relative_depletion_slope`` (``1b``) -- a dimensionless ratio of initial
  substrate-depletion slopes, normalised so that the parent Ssal-KRED is 1.
  Clarified lysate, 340 nm. Not a rate, not a kcat, and not convertible to one.
* ``product_concentration`` (``2b``-``5b``, R and S separately) -- millimolar
  product after 20 h at 20 degrees C from 10 mM substrate in a crude lysate,
  with an empty-vector background subtracted. A fixed-time endpoint, so it is
  bounded by the substrate supplied and says nothing about a rate.
* ``signed_ee`` -- enantiomeric excess in per cent, positive for R preference.

A ranking that mixes these compares quantities the assays did not measure on
one scale, so :meth:`ActivityRecord.require` refuses a quantity a record does
not have and :func:`endpoint_table` keeps one substrate and one label type per
call.

WHAT A ZERO MEANS, AND WHAT A BLANK MEANS
=========================================
Three different absences, kept apart:

* **not assayed.** The 40 insoluble constructs have every activity field blank.
  The source says so explicitly, and so does this loader: they are
  :data:`NOT_ASSAYED` and are **not** catalytically negative. A protein that
  did not fold tells you about expression, not about chemistry.
* **below detection.** A soluble construct with both products at exactly 0.0 mM
  produced no product this assay could see, after background subtraction, at
  this time and loading. That is :data:`BELOW_DETECTION` -- left-censored at a
  limit of detection the source does not state. It is the closest thing to a
  negative in this data, and it is still not a measured zero.
* **missing ee.** With both products at zero, ``(R-S)/(R+S)`` is 0/0 and the
  source leaves the cell empty. A blank ee is undefined, never 0.

ee IS REPORTED, NOT DERIVED
===========================
Recomputing ``(R-S)/(R+S)`` from the printed product concentrations reproduces
the printed ee for some rows and not others, by up to about seven points. The
two are separate measurements -- a chiral separation and a quantification --
and the printed products are rounded to one or two decimals, which moves a
small-denominator ratio a long way. So both are kept, as two fields, and the
disagreement is recorded per row. This is the same discipline
:mod:`eagent.eval.kred_reference` applies to a reported catalytic efficiency
against a ``kcat``/``Km`` quotient.

The loader also found two rows the source's own audit did not list: in the
``2b`` column ``Ort-EZM-6`` prints an ee of -100 and ``Ort-EZM-24`` an ee of 0,
both with R and S at exactly zero, so both are an ee on a 0/0. They are
reported as findings and their ee is withheld.

HOW INDEPENDENT THESE 69 ARE
============================
They were found by iterated homology search from one parent, so they are a
family, not a sample of enzymes. :func:`independence` single-links them by
coverage-adjusted identity at several thresholds and reports the group count,
which is what a claim about "independent enzymes" may cite. The parent's own
row is in the data twice over (it is the normalisation point for ``1b`` and it
appears in both rounds); it is one enzyme.

THE TWO QUARANTINED SUBSTRATES
==============================
The deposited SMILES for ``1a`` encodes an alcohol where the figure draws a
ketone, and for ``5a`` a carboxylate where the figure draws an ethyl ester. The
source's curation quarantined both and proposed corrections. This loader
carries the original SMILES, the proposed correction and the quarantine flag,
and :meth:`Substrate.require_smiles` refuses to hand out a quarantined
structure. Nothing here rewrites a deposited value.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..errors import EAgentError
from .kred_bundle import bundle_dir
from .kred_reference import Finding, LINEAGE_IDENTITY_THRESHOLD

__all__ = [
    "ActivityError",
    "NOT_ASSAYED",
    "BELOW_DETECTION",
    "MEASURED",
    "SUBSTRATES",
    "ENDPOINT_SUBSTRATES",
    "LABEL_TYPES",
    "EE_RESIDUAL_NOTE_THRESHOLD",
    "Substrate",
    "Endpoint",
    "ActivityRecord",
    "ActivitySet",
    "load_activity_set",
    "endpoint_table",
    "independence",
]


class ActivityError(EAgentError):
    """A quantity was asked of a record that does not carry it."""


#: An insoluble construct: never assayed, and never a catalytic negative.
NOT_ASSAYED = "not_assayed"
#: Soluble, assayed, no product this assay could detect. Left-censored.
BELOW_DETECTION = "below_detection"
#: Soluble, assayed, product quantified.
MEASURED = "measured"

#: ``1b`` is a ratio; ``2b``-``5b`` are concentrations; ee is a per cent. Three
#: label types, and :func:`endpoint_table` serves one at a time.
LABEL_TYPES = ("relative_depletion_slope", "product_concentration", "signed_ee")

#: Printed ee against ee recomputed from the printed products: above this many
#: percentage points the row gets an information finding naming both. Not an
#: error -- the two are different measurements (see the module docstring).
EE_RESIDUAL_NOTE_THRESHOLD = 1.0


@dataclass(frozen=True)
class Substrate:
    """One assayed substrate, with its deposited structure and its audit."""

    substrate_id: str
    name: str
    smiles_as_deposited: str
    product_column: str
    label_type: str
    quarantined: bool = False
    quarantine_reason: str = ""
    proposed_smiles: str = ""

    def require_smiles(self) -> str:
        """The deposited SMILES, or a refusal naming the quarantine.

        A quarantined structure is not handed out: ``1a`` as deposited is the
        alcohol, which is the product, and docking the product of a reduction
        into a reductase would return a confident answer to the wrong question.
        """
        if self.quarantined:
            raise ActivityError(
                f"{self.substrate_id}: the deposited SMILES is quarantined -- "
                f"{self.quarantine_reason} The source value is kept in "
                f"smiles_as_deposited and a correction is proposed in "
                f"proposed_smiles; neither is used until a chemist settles it.")
        return self.smiles_as_deposited


#: The five substrates, their deposited SMILES, and the two quarantines. Taken
#: from the archive's ``zenodo_substrate_SMILES.csv`` and
#: ``curation_audit.json``; the findings are the source's own.
SUBSTRATES: Mapping[str, Substrate] = {
    "1a": Substrate(
        "1a", "ipatasertib precursor",
        "C[C@H]3CC(O)c2ncnc(N1CCN(C(=O)OC(C)(C)C)CC1)c23",
        "1b", "relative_depletion_slope", quarantined=True,
        quarantine_reason=(
            "the deposited SMILES carries an alcohol C(O) at the site the "
            "reaction reduces, while the paper's figure draws a ketone C(=O); "
            "it reads as the product, not the substrate."),
        proposed_smiles="C[C@H]3CC(=O)c2ncnc(N1CCN(C(=O)OC(C)(C)C)CC1)c23"),
    "2a": Substrate("2a", "1-phenylethanone (acetophenone)",
                    "CC(=O)C1=CC=CC=C1", "2b", "product_concentration"),
    "3a": Substrate("3a", "ethyl 2-oxo-4-phenylbutyrate",
                    "CCOC(=O)C(=O)CCC1=CC=CC=C1", "3b", "product_concentration"),
    "4a": Substrate("4a", "1-Boc-3-piperidone",
                    "CC(C)(C)OC(=O)N1CCCC(=O)C1", "4b", "product_concentration"),
    "5a": Substrate(
        "5a", "4-chloroacetoacetate as deposited", "C(C(=O)CCl)C(=O)[O-]",
        "5b", "product_concentration", quarantined=True,
        quarantine_reason=(
            "the deposited SMILES encodes the carboxylate, while the paper's "
            "figure draws the ethyl ester; the ethyl group and the charge state "
            "are both wrong for the compound depicted."),
        proposed_smiles="CCOC(=O)CC(=O)CCl"),
}

#: The four substrates whose labels are product concentrations.
ENDPOINT_SUBSTRATES = ("2a", "3a", "4a", "5a")

_ROUND_FILES = ("activity_expansion_2026/round_1.csv",
                "activity_expansion_2026/round_2.csv")
_IDENTITY_FILE = "activity_expansion_2026/derived_identity_groups.json"


@dataclass(frozen=True)
class Endpoint:
    """One (enzyme, substrate) product measurement."""

    substrate_id: str
    status: str                       # measured | below_detection | not_assayed
    product_r_mM: float | None
    product_s_mM: float | None
    ee_reported: float | None
    ee_from_products: float | None
    withheld: Mapping[str, str] = field(default_factory=dict)

    @property
    def total_product_mM(self) -> float | None:
        if self.product_r_mM is None or self.product_s_mM is None:
            return None
        return self.product_r_mM + self.product_s_mM

    @property
    def ee_residual(self) -> float | None:
        """Printed ee minus ee recomputed from the printed products."""
        if self.ee_reported is None or self.ee_from_products is None:
            return None
        return self.ee_reported - self.ee_from_products

    def require(self, quantity: str) -> float:
        """The value, or a refusal saying why there is not one.

        ``withheld`` is the authority, not the stored value. A left-censored
        endpoint carries the source's printed ``0.0``, and a gate that only
        refused ``None`` would hand that zero out as a measurement -- which is
        the one thing this class exists to prevent.
        """
        values = {"product_r_mM": self.product_r_mM, "product_s_mM": self.product_s_mM,
                  "total_product_mM": self.total_product_mM,
                  "ee_reported": self.ee_reported,
                  "ee_from_products": self.ee_from_products}
        if quantity not in values:
            raise ActivityError(f"{quantity!r} is not a quantity of an endpoint; "
                                f"known: {sorted(values)}")
        if quantity in self.withheld:
            raise ActivityError(
                f"{self.substrate_id}: no {quantity}: {self.withheld[quantity]}")
        value = values[quantity]
        if value is None:
            raise ActivityError(f"{self.substrate_id}: no {quantity}: {self._why()}")
        return value

    def _why(self) -> str:
        if self.status == NOT_ASSAYED:
            return ("the construct did not express solubly, so it was never "
                    "assayed; this is not a catalytic negative")
        if self.status == BELOW_DETECTION:
            return ("no product was detected after background subtraction; the "
                    "value is left-censored at an unstated limit of detection, "
                    "not a measured zero")
        return "the source leaves the cell empty"

    def to_dict(self) -> dict[str, Any]:
        return {"substrate_id": self.substrate_id, "status": self.status,
                "product_r_mM": self.product_r_mM, "product_s_mM": self.product_s_mM,
                "ee_reported": self.ee_reported,
                "ee_from_products": self.ee_from_products,
                "withheld": dict(self.withheld)}


@dataclass(frozen=True)
class ActivityRecord:
    """One ortholog: its sequence, whether it expressed, and its endpoints."""

    enzyme_id: str
    homolog_id: str
    sequence: str
    organism: str
    annotated_function: str
    solubly_expressed: bool
    refined: bool
    rounds: tuple[str, ...]
    relative_depletion_slope: float | None
    melting_temperature_c: float | None
    endpoints: Mapping[str, Endpoint]
    withheld: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_parent(self) -> bool:
        return self.enzyme_id == "Ssal-KRED"

    def endpoint(self, substrate_id: str) -> Endpoint:
        if substrate_id not in self.endpoints:
            raise ActivityError(
                f"{self.enzyme_id} has no endpoint for {substrate_id!r}; "
                f"assayed: {sorted(self.endpoints)}")
        return self.endpoints[substrate_id]

    def require(self, quantity: str) -> float:
        """The value, or a refusal. ``withheld`` is the authority, as above."""
        values = {"relative_depletion_slope": self.relative_depletion_slope,
                  "melting_temperature_c": self.melting_temperature_c}
        if quantity not in values:
            raise ActivityError(f"{quantity!r} is not a record-level quantity; "
                                f"known: {sorted(values)}")
        if quantity in self.withheld:
            raise ActivityError(
                f"{self.enzyme_id}: no {quantity}: {self.withheld[quantity]}")
        value = values[quantity]
        if value is None:
            raise ActivityError(f"{self.enzyme_id}: no {quantity}: not recorded")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"enzyme_id": self.enzyme_id, "homolog_id": self.homolog_id,
                "organism": self.organism,
                "annotated_function": self.annotated_function,
                "solubly_expressed": self.solubly_expressed, "refined": self.refined,
                "rounds": list(self.rounds), "sequence_length": len(self.sequence),
                "relative_depletion_slope": self.relative_depletion_slope,
                "melting_temperature_c": self.melting_temperature_c,
                "endpoints": {k: v.to_dict() for k, v in sorted(self.endpoints.items())},
                "withheld": dict(self.withheld)}


@dataclass(frozen=True)
class ActivitySet:
    """The 109 constructs, their labels, and what the checks found."""

    directory: Path
    records: tuple[ActivityRecord, ...]
    findings: tuple[Finding, ...] = ()

    def record(self, enzyme_id: str) -> ActivityRecord:
        for r in self.records:
            if r.enzyme_id == enzyme_id:
                return r
        raise KeyError(enzyme_id)

    @property
    def soluble(self) -> tuple[ActivityRecord, ...]:
        return tuple(r for r in self.records if r.solubly_expressed)

    @property
    def insoluble(self) -> tuple[ActivityRecord, ...]:
        return tuple(r for r in self.records if not r.solubly_expressed)

    def counts(self) -> dict[str, Any]:
        by_status: dict[str, dict[str, int]] = {}
        for sub in ENDPOINT_SUBSTRATES:
            tally: dict[str, int] = {MEASURED: 0, BELOW_DETECTION: 0, NOT_ASSAYED: 0}
            for r in self.records:
                tally[r.endpoint(sub).status] += 1
            by_status[sub] = tally
        return {
            "constructs": len(self.records),
            "soluble": len(self.soluble),
            "insoluble": len(self.insoluble),
            "endpoint_pairs_assayed": sum(
                1 for r in self.soluble for s in ENDPOINT_SUBSTRATES
                if r.endpoint(s).status != NOT_ASSAYED),
            "relative_slope_rows": sum(
                1 for r in self.records if r.relative_depletion_slope is not None),
            "by_substrate": by_status,
        }

    def summary(self) -> dict[str, Any]:
        return {"directory": str(self.directory.name), "counts": self.counts(),
                "findings": {sev: sum(1 for f in self.findings if f.severity == sev)
                             for sev in ("error", "warning", "info")}}


# ==========================================================================
# reading
# ==========================================================================

def _number(text: str | None) -> float | None:
    s = (text or "").strip()
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _read_rounds(bundle: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[str]],
                                        list[Finding]]:
    findings: list[Finding] = []
    merged: dict[str, dict[str, str]] = {}
    rounds: dict[str, list[str]] = {}
    for rel in _ROUND_FILES:
        path = bundle / rel
        if not path.is_file():
            raise ActivityError(f"{path} is not there; the activity data is part "
                                f"of the bundle (see kred_bundle)")
        tag = Path(rel).stem
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("ID") or "").strip()
                if not key:
                    continue
                rounds.setdefault(key, []).append(tag)
                if key not in merged:
                    merged[key] = row
                    continue
                differ = [f for f, v in row.items()
                          if (v or "").strip() != (merged[key].get(f) or "").strip()]
                if differ:
                    findings.append(Finding(
                        "error", "activity.round_disagreement", key,
                        f"this construct appears in both rounds and the rows "
                        f"differ on {differ}; which one is the measurement has "
                        f"to be settled before either is used"))
    return merged, rounds, findings


def load_activity_set(reference_dir: str | Path | None = None, *,
                      strict: bool = True) -> ActivitySet:
    """Load the ortholog activity data, with every absence kept distinct."""
    bundle = bundle_dir(reference_dir)
    merged, rounds, findings = _read_rounds(bundle)

    records: list[ActivityRecord] = []
    for key in sorted(merged):
        row = merged[key]
        soluble = (row.get("solubly_expressed") or "").strip() == "1"
        sequence = "".join((row.get("sequence") or "").split())
        activity_fields = ["1b"] + [f"{SUBSTRATES[s].product_column} ({side})"
                                    for s in ENDPOINT_SUBSTRATES for side in "RS"]
        has_values = [f for f in activity_fields if (row.get(f) or "").strip()]
        if not soluble and has_values:
            findings.append(Finding(
                "error", "activity.insoluble_carries_values", key,
                f"the construct is recorded as not solubly expressed yet carries "
                f"values in {has_values}; an insoluble row must stay blank, "
                f"because a value there would be read as a measured activity"))
        if soluble and not (row.get("1b") or "").strip():
            findings.append(Finding(
                "warning", "activity.soluble_without_slope", key,
                "soluble but no relative depletion slope recorded"))

        withheld: dict[str, str] = {}
        slope = _number(row.get("1b")) if soluble else None
        if slope is None:
            withheld["relative_depletion_slope"] = (
                "the construct did not express solubly, so it was never assayed"
                if not soluble else "the source leaves the cell empty")
        tm = _number(row.get("melting_temperature"))
        if tm is None:
            withheld["melting_temperature_c"] = (
                "no melting temperature is recorded for this construct")

        endpoints: dict[str, Endpoint] = {}
        for sub in ENDPOINT_SUBSTRATES:
            col = SUBSTRATES[sub].product_column
            r_raw, s_raw = row.get(f"{col} (R)"), row.get(f"{col} (S)")
            ee_raw = row.get(f"ee {col} (%)")
            pr, ps, ee = _number(r_raw), _number(s_raw), _number(ee_raw)
            ep_withheld: dict[str, str] = {}
            if not soluble:
                status = NOT_ASSAYED
                pr = ps = ee = None
                for q in ("product_r_mM", "product_s_mM", "total_product_mM",
                          "ee_reported", "ee_from_products"):
                    ep_withheld[q] = (
                        "the construct did not express solubly, so it was never "
                        "assayed; this is not a catalytic negative")
            elif pr is None or ps is None:
                status = NOT_ASSAYED
                findings.append(Finding(
                    "warning", "activity.soluble_endpoint_blank", f"{key}/{sub}",
                    "soluble but this endpoint's product cells are blank"))
                for q in ("product_r_mM", "product_s_mM", "total_product_mM"):
                    ep_withheld[q] = "the source leaves the cell empty"
            elif pr == 0.0 and ps == 0.0:
                status = BELOW_DETECTION
                # The printed zeros stay on the record -- they are what the
                # source says -- but they are not handed out as measurements.
                # A caller who wants them asks endpoint_table for them
                # explicitly and takes them as an upper bound.
                for q in ("product_r_mM", "product_s_mM", "total_product_mM"):
                    ep_withheld[q] = (
                        "no product was detected after background subtraction, "
                        "so this is left-censored at an unstated limit of "
                        "detection and not a measured zero; the source's printed "
                        "0.0 is on the record and endpoint_table serves it only "
                        "when include_censored is set")
            else:
                status = MEASURED

            derived = None
            if pr is not None and ps is not None and (pr + ps) > 0:
                derived = (pr - ps) / (pr + ps) * 100.0

            if status == BELOW_DETECTION and ee is not None:
                # A warning, not an error: the contradiction is in the source,
                # the loader has already withheld the value, and nothing
                # downstream can reach it. An error here would mean the shipped
                # data could never be loaded strictly, so every caller would
                # pass strict=False and strictness would stop meaning anything.
                findings.append(Finding(
                    "warning", "activity.ee_on_zero_over_zero", f"{key}/{sub}",
                    f"both products are exactly 0 mM yet an ee of {ee:g} is "
                    f"printed; (R-S)/(R+S) is 0/0 here, so the printed value "
                    f"cannot be an excess over these products. Withheld."))
                ee = None
                ep_withheld["ee_reported"] = (
                    "the source prints an ee although both products are zero, "
                    "which is an ee on a 0/0; see the finding")
            if derived is None:
                ep_withheld.setdefault("ee_from_products", (
                    "both products are zero, so (R-S)/(R+S) is 0/0 and no ee "
                    "follows from them" if status == BELOW_DETECTION
                    else "the products are not both recorded"))
            if ee is None and status == MEASURED:
                findings.append(Finding(
                    "info", "activity.product_without_ee", f"{key}/{sub}",
                    "product was detected but no ee is reported; a blank ee is "
                    "undefined, not zero"))
                ep_withheld.setdefault("ee_reported", "the source reports no ee")
            if ee is not None and derived is not None:
                residual = abs(ee - derived)
                if residual > EE_RESIDUAL_NOTE_THRESHOLD:
                    findings.append(Finding(
                        "info", "activity.ee_reported_vs_products", f"{key}/{sub}",
                        f"the reported ee is {ee:g}% and (R-S)/(R+S) over the "
                        f"printed products gives {derived:.2f}%; both are kept, "
                        f"as two fields"))
            endpoints[sub] = Endpoint(sub, status, pr, ps, ee, derived, ep_withheld)

        records.append(ActivityRecord(
            enzyme_id=key, homolog_id=(row.get("homolog_id") or "").strip(),
            sequence=sequence, organism=(row.get("organism") or "").strip(),
            annotated_function=(row.get("annotated_function") or "").strip(),
            solubly_expressed=soluble,
            refined=(row.get("refined") or "").strip() == "1",
            rounds=tuple(rounds[key]), relative_depletion_slope=slope,
            melting_temperature_c=tm, endpoints=endpoints, withheld=withheld))

    seqs = {}
    for r in records:
        seqs.setdefault(r.sequence, []).append(r.enzyme_id)
    for seq, ids in seqs.items():
        if len(ids) > 1:
            findings.append(Finding(
                "warning", "activity.identical_sequences", ", ".join(sorted(ids)),
                "these constructs carry an identical sequence and are one enzyme"))

    findings.sort(key=lambda f: ({"error": 0, "warning": 1, "info": 2}[f.severity],
                                 f.code, f.subject))
    errors = [f for f in findings if f.severity == "error"]
    if errors and strict:
        shown = "\n".join(f"  {f}" for f in errors[:10])
        raise ActivityError(f"{len(errors)} error(s) in the activity data:\n{shown}")
    return ActivitySet(directory=bundle, records=tuple(records),
                       findings=tuple(findings))


# ==========================================================================
# serving one label type at a time
# ==========================================================================

def endpoint_table(activity: ActivitySet, substrate_id: str, quantity: str, *,
                   include_censored: bool = False
                   ) -> list[tuple[ActivityRecord, float]]:
    """``(record, value)`` for one substrate and one quantity. Nothing pooled.

    Never-assayed constructs are always left out. Left-censored ones are left
    out unless ``include_censored`` is set, and then they come back at the value
    the source printed (zero) -- which a caller must treat as an upper bound,
    not a measurement. One substrate per call, because the four endpoints are
    four assays and a column that mixes them is not a variable.
    """
    if substrate_id not in SUBSTRATES:
        raise ActivityError(f"{substrate_id!r} is not an assayed substrate; "
                            f"known: {sorted(SUBSTRATES)}")
    out: list[tuple[ActivityRecord, float]] = []
    for record in activity.records:
        endpoint = record.endpoint(substrate_id) if substrate_id in record.endpoints \
            else None
        if endpoint is None or endpoint.status == NOT_ASSAYED:
            continue
        if endpoint.status == BELOW_DETECTION:
            if not include_censored:
                continue
            if quantity in ("product_r_mM", "product_s_mM", "total_product_mM"):
                out.append((record, 0.0))
            continue
        try:
            out.append((record, endpoint.require(quantity)))
        except ActivityError:
            continue
    return out


# ==========================================================================
# how many independent enzymes these are
# ==========================================================================

def independence(activity: ActivitySet, *,
                 thresholds: Sequence[float] = (0.40, 0.70, 0.90, 0.95),
                 identity_file: str | Path | None = None) -> dict[str, Any]:
    """Single-linkage groups of the soluble constructs, by sequence identity.

    These orthologs came from an iterated homology search around one parent, so
    they are a family. A claim that reads "69 enzymes" is a claim about 69
    draws, and this is the number that may actually be cited: the group count at
    a stated identity threshold. At this project's own 40 % lineage threshold
    the 69 are a handful of groups, so a grouped benchmark over them has an
    effective sample size in that handful and not in the dozens.

    The pairwise identities are expensive (about 2300 alignments), so they are
    computed once by :func:`write_identity_groups` and stored in the bundle.
    This function reads that file and refuses rather than silently recomputing,
    because a number that appears only when somebody waits a minute for it is a
    number that will quietly differ between runs.
    """
    path = Path(identity_file) if identity_file is not None \
        else activity.directory / _IDENTITY_FILE
    if not path.is_file():
        raise ActivityError(
            f"{path} is not there, so the group counts cannot be read. Run "
            f"`eagent reference activity --write-identity` to compute them; "
            f"they are not recomputed on the fly because a derived number "
            f"should be stored once and reviewed.")
    stored = json.loads(path.read_text(encoding="utf-8"))
    ids = [r.enzyme_id for r in activity.soluble]
    if sorted(stored.get("members") or []) != sorted(ids):
        raise ActivityError(
            f"{path.name} was computed for a different set of constructs than "
            f"the one loaded; recompute it")
    return {
        "soluble_constructs": len(ids),
        "note": stored.get("note", ""),
        "groups_by_threshold": stored.get("groups_by_threshold", {}),
        "cited_threshold": stored.get("cited_threshold"),
        "n_independent_at_cited_threshold": (
            stored.get("groups_by_threshold", {})
            .get(str(stored.get("cited_threshold")), {})
            .get("n_groups")),
    }


def write_identity_groups(activity: ActivitySet, *,
                          thresholds: Sequence[float] = (0.40, 0.70, 0.90, 0.95),
                          cited_threshold: float = LINEAGE_IDENTITY_THRESHOLD,
                          out_path: str | Path | None = None) -> Path:
    """Compute and store the pairwise-identity groups. Slow and deliberate.

    ``cited_threshold`` defaults to
    :data:`eagent.eval.kred_reference.LINEAGE_IDENTITY_THRESHOLD`, the same 40 %
    this project already uses to decide that two enzymes are one lineage. Citing
    the same threshold here is the point: it makes the ortholog group count
    directly comparable with the six lineages of the structure set, instead of
    two numbers that sound alike and were computed by different rules. The
    looser thresholds are stored beside it so a reader can see how far the count
    moves, which for an ortholog search is a long way.""" 
    from ..science.numbering import needleman_wunsch

    records = activity.soluble
    ids = [r.enzyme_id for r in records]
    seqs = {r.enzyme_id: r.sequence for r in records}
    pairs: dict[str, float] = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            alignment = needleman_wunsch(seqs[a], seqs[b])
            coverage = min(alignment.coverage_a(), alignment.coverage_b())
            pairs[f"{a}|{b}"] = round(alignment.identity * coverage, 6)

    groups_by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        parent = {i: i for i in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for key, value in pairs.items():
            if value >= threshold:
                a, b = key.split("|")
                parent[find(a)] = find(b)
        clusters: dict[str, list[str]] = {}
        for i in ids:
            clusters.setdefault(find(i), []).append(i)
        groups_by_threshold[f"{threshold:.2f}"] = {
            "n_groups": len(clusters),
            "largest_group": max(len(c) for c in clusters.values()),
            "singletons": sum(1 for c in clusters.values() if len(c) == 1),
        }

    values = sorted(pairs.values())
    document = {
        "note": ("single-linkage groups of the solubly expressed constructs by "
                 "coverage-adjusted global identity. These orthologs came from "
                 "an iterated homology search around one parent, so the group "
                 "count at a stated threshold -- not the construct count -- is "
                 "what a claim about independent enzymes may cite."),
        "algorithm": ("eagent.science.numbering.needleman_wunsch; identity "
                      "scaled by the covered fraction of the shorter sequence"),
        "members": sorted(ids),
        "n_pairs": len(pairs),
        "identity_min": values[0] if values else None,
        "identity_median": values[len(values) // 2] if values else None,
        "identity_max": values[-1] if values else None,
        "cited_threshold": f"{cited_threshold:.2f}",
        "groups_by_threshold": groups_by_threshold,
        "pairwise_identity": dict(sorted(pairs.items())),
    }
    target = Path(out_path) if out_path is not None \
        else activity.directory / _IDENTITY_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target
