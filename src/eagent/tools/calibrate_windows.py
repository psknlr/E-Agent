"""Calibrate a catalytic template's windows from reference complexes.

Not one of the ten interfaces: it is the offline procedure that decides whether
``evaluate_catalysis`` is allowed to believe a window, and it has no place in a
run. It exists so that the answer to "may this distance reject an enzyme?" is a
computed, tamper-evident record rather than a string in a YAML file.

The measurement is taken here by exactly the code ``evaluate_catalysis`` uses
to judge a candidate -- :func:`build_role_context` and
:func:`eagent.science.geometry.measure_all` -- so a window cannot be fitted to
one observable and then applied to another. The statistics and the record are
in :mod:`eagent.science.calibration`; this module only joins coordinates to
them.

Nothing here edits a template. A passing calibration yields the exact fields a
curator would put into the constraint (:func:`calibrated_fields`), and the
template is changed by a reviewed commit, because the authority to reject
candidates is the kind of thing that should leave a diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..schemas.templates import CatalyticTemplate
from ..science import geometry as geom
from ..science.calibration import (
    CalibrationPolicy, CalibrationRecord, CalibrationStore, ReferenceObservation,
    calibrate,
)
from ..science.structure_io import Structure, StructureParseError, read_structure
from .evaluate_catalysis import PoseBinding, build_role_context

__all__ = [
    "ReferenceComplex",
    "CalibrationReport",
    "measure_references",
    "calibrate_template",
    "calibrated_fields",
]


@dataclass(frozen=True)
class ReferenceComplex:
    """One complex of known activity, with the map from roles to its atoms.

    ``binding`` is required and explicit. Calibrating on a reference whose
    substrate was *inferred* from "the only ligand in the file" would let a
    wrong guess about which molecule is the substrate set a window, so
    inference is switched off for references even though ``evaluate_catalysis``
    allows it for candidates.
    """

    reference_id: str
    label: str                      # "active" | "inactive"
    evidence: str                   # the citation for the label
    binding: PoseBinding
    structure: Structure | None = None
    path: str | None = None
    source_type: str = "experimental"
    restrained: tuple[str, ...] = ()


@dataclass
class CalibrationReport:
    """One record per constraint, and what was written."""

    template_id: str
    records: dict[str, CalibrationRecord] = field(default_factory=dict)
    unmeasurable: dict[str, str] = field(default_factory=dict)
    written: dict[str, str] = field(default_factory=dict)

    @property
    def calibrated(self) -> list[str]:
        return sorted(n for n, r in self.records.items()
                      if r.verdict.meets_policy)

    @property
    def not_calibrated(self) -> list[str]:
        return sorted(n for n, r in self.records.items()
                      if not r.verdict.meets_policy)

    def render(self) -> str:
        lines = [f"calibration of {self.template_id}"]
        for name in sorted(self.records):
            rec = self.records[name]
            v = rec.verdict
            lines.append(
                f"  {name}: {'CALIBRATED' if v.meets_policy else 'not calibrated'}"
                f"  (n_active={v.n_active}, n_inactive={v.n_inactive}, "
                f"window={rec.window})")
            for reason in v.reasons:
                lines.append(f"      - {reason}")
            for ex in rec.exclusions:
                lines.append(f"      excluded {ex.reference_id}: {ex.reason}")
        for name, why in sorted(self.unmeasurable.items()):
            lines.append(f"  {name}: no reference could be measured -- {why}")
        return "\n".join(lines)


def _structure_of(ref: ReferenceComplex) -> tuple[Structure | None, str]:
    if ref.structure is not None:
        return ref.structure, ""
    if not ref.path:
        return None, "neither a structure nor a path was given"
    p = Path(ref.path)
    if not p.is_file():
        return None, f"{p} does not exist"
    try:
        return read_structure(p, structure_id=ref.reference_id), ""
    except StructureParseError as exc:
        return None, f"{p} could not be parsed: {exc}"


def measure_references(
    references: Iterable[ReferenceComplex], template: CatalyticTemplate,
    constraint_names: Sequence[str] | None = None,
) -> dict[str, list[ReferenceObservation]]:
    """Measure every constraint on every reference, by the evaluator's own code.

    A reference that cannot be read, or on which a constraint cannot be
    measured, still produces an observation -- with ``value=None`` -- so that
    :func:`calibrate` can list it as an exclusion with a reason instead of the
    reference quietly vanishing from the count.
    """
    wanted = [c for c in template.geometry_constraints
              if constraint_names is None or c.name in set(constraint_names)]
    out: dict[str, list[ReferenceObservation]] = {c.name: [] for c in wanted}
    for ref in references:
        structure, why = _structure_of(ref)
        measurements: dict[str, float | None] = {}
        if structure is not None:
            context = build_role_context(
                structure, ref.binding, template,
                infer_single_ligand_substrate=False)
            measurements = geom.measure_all(wanted, context.resolver())
        for constraint in wanted:
            out[constraint.name].append(ReferenceObservation(
                reference_id=ref.reference_id, label=ref.label,
                value=measurements.get(constraint.name) if structure else None,
                evidence=ref.evidence, source_type=ref.source_type,
                restrained=ref.restrained))
    return out


def calibrate_template(
    references: Iterable[ReferenceComplex], template: CatalyticTemplate,
    policy: CalibrationPolicy, *, store: CalibrationStore | None = None,
    constraint_names: Sequence[str] | None = None,
) -> CalibrationReport:
    """Calibrate every (or the named) constraint of ``template``.

    Records are written to ``store`` whether or not they pass: a calibration
    that fell short is evidence about the campaign (how far, and on what), and
    a failed record has no ``calibrated_on`` entry for a template to cite.
    """
    refs = list(references)
    report = CalibrationReport(template_id=template.template_id)
    observed = measure_references(refs, template, constraint_names)
    for name, observations in observed.items():
        if not any(o.value is not None for o in observations):
            report.unmeasurable[name] = (
                "every reference left it unmeasured; check the binding's "
                "role keys against the template's atom tokens")
        record = calibrate(name, observations, policy)
        report.records[name] = record
        if store is not None:
            report.written[name] = str(store.write(record))
    return report


def calibrated_fields(record: CalibrationRecord) -> dict[str, Any]:
    """The exact constraint fields a passing calibration licenses.

    Returned, not applied. The window is the record's, to the digit: a
    template whose window differs from the one its citation names is, by
    :meth:`CalibrationStore.verify`, an uncalibrated window again. Raises for a
    record that does not meet its policy, because there is nothing to cite.
    """
    entry = record.calibrated_on_entry
    if entry is None or record.window is None:
        raise ValueError(
            f"{record.constraint}: this calibration does not meet its policy "
            f"({'; '.join(record.verdict.reasons)}), so there is no citation "
            f"to put in a template")
    lo, hi = record.window
    return {"min_value": lo, "max_value": hi, "target": None, "tolerance": None,
            "calibrated_on": [entry]}
