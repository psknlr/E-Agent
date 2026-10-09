"""One conversion table, used by everything that puts two numbers together.

WHY THIS IS A MODULE AND NOT A DICT IN EACH CALLER
==================================================
The arithmetic in this project is almost all comparison: a variant against its
parent, a well against its plate's background, three replicates against each
other. Every one of those operations is wrong -- silently, and in the
favourable direction -- if the two sides were recorded on different scales.

A plate that reports ``0.01 mM`` in one well and ``10 uM`` in the next has
reported the *same concentration twice*. Taking the median of the raw numbers
gives ``5.005``, which is five hundred times one of them and half the other,
and nothing downstream can tell that it is not a measurement. The same shape
appears as a parent at ``1 s-1`` and a variant at ``60 min-1``: identical
activity, reported as a fifty-nine-fold improvement, and confirmed as
reproducible because the replicate spreads were left in the unconverted units
too.

So the table lives in one place, every caller converts before it compares, and
a unit that is not in the table is a **refusal** rather than a licence to
subtract anyway. Adding a conversion means adding a line here, where a
reviewer can check the factor, instead of inlining a magic number at a call
site where nobody will ever look at it again.

WHAT IS DELIBERATELY NOT HERE
=============================
Molar mass. ``mM`` and ``mg/mL`` are both concentrations and there is no
substrate-independent factor between them, so they do not share a canonical
unit and a pair of them is refused. Likewise ``U/mg`` and ``%``: a specific
activity and a conversion are different quantities, and the honest answer to
"combine these" is that they cannot be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "UNIT_TABLE",
    "normalise_unit_text",
    "canonical_unit",
    "convert_measurement",
    "Reconciliation",
    "reconcile_units",
]


#: ``normalised spelling -> (canonical unit, multiply by this to reach it)``.
#:
#: Two units share a canonical name only when the factor between them is a
#: pure number, independent of what was measured. Everything else is a
#: different quantity and belongs to a different canonical unit, so that a
#: pair of them is refused rather than converted.
UNIT_TABLE: dict[str, tuple[str, float]] = {
    # -- first-order rate constants -> per second ------------------------
    "s-1": ("s-1", 1.0), "s^-1": ("s-1", 1.0), "1/s": ("s-1", 1.0),
    "sec-1": ("s-1", 1.0), "per second": ("s-1", 1.0),
    "min-1": ("s-1", 1.0 / 60.0), "min^-1": ("s-1", 1.0 / 60.0),
    "1/min": ("s-1", 1.0 / 60.0), "per minute": ("s-1", 1.0 / 60.0),
    "h-1": ("s-1", 1.0 / 3600.0), "hr-1": ("s-1", 1.0 / 3600.0),
    "1/h": ("s-1", 1.0 / 3600.0), "per hour": ("s-1", 1.0 / 3600.0),
    # -- concentrations -> millimolar ------------------------------------
    "mm": ("mM", 1.0), "mmol/l": ("mM", 1.0), "mmol/l-1": ("mM", 1.0),
    "um": ("mM", 1e-3), "µm": ("mM", 1e-3), "umol/l": ("mM", 1e-3),
    "µmol/l": ("mM", 1e-3),
    "nm": ("mM", 1e-6), "nmol/l": ("mM", 1e-6),
    "m": ("mM", 1e3), "mol/l": ("mM", 1e3),
    # -- second-order rate constants (kcat/Km) -> per molar per second ----
    # A catalytic efficiency is a rate constant per concentration, so it has
    # its own canonical unit: it must not pool with a first-order constant
    # ("s-1") or with a concentration ("mM"), and a pair across those is
    # refused. The factors are pure arithmetic on the two unit prefixes:
    # ``mM-1 min-1 -> M-1 s-1`` is 1000 / 60, because a millimolar is a
    # thousandth of a molar (so a per-millimolar constant is a thousand times
    # a per-molar one) and a minute is sixty seconds. The orderings of the two
    # factors in a spelling ("min-1 mM-1" and "mM-1 min-1") are both in use in
    # the papers this project reads, and both are listed rather than parsed.
    "m-1 s-1": ("M-1 s-1", 1.0), "m^-1 s^-1": ("M-1 s-1", 1.0),
    "m-1s-1": ("M-1 s-1", 1.0), "1/(m*s)": ("M-1 s-1", 1.0),
    "l/(mol*s)": ("M-1 s-1", 1.0), "l mol-1 s-1": ("M-1 s-1", 1.0),
    "s-1 m-1": ("M-1 s-1", 1.0), "s^-1 m^-1": ("M-1 s-1", 1.0),
    "mm-1 s-1": ("M-1 s-1", 1e3), "mm^-1 s^-1": ("M-1 s-1", 1e3),
    "s-1 mm-1": ("M-1 s-1", 1e3), "s^-1 mm^-1": ("M-1 s-1", 1e3),
    "um-1 s-1": ("M-1 s-1", 1e6), "um^-1 s^-1": ("M-1 s-1", 1e6),
    "µm-1 s-1": ("M-1 s-1", 1e6), "µm^-1 s^-1": ("M-1 s-1", 1e6),
    "s-1 um-1": ("M-1 s-1", 1e6), "s^-1 um^-1": ("M-1 s-1", 1e6),
    "m-1 min-1": ("M-1 s-1", 1.0 / 60.0), "m^-1 min^-1": ("M-1 s-1", 1.0 / 60.0),
    "min-1 m-1": ("M-1 s-1", 1.0 / 60.0), "min^-1 m^-1": ("M-1 s-1", 1.0 / 60.0),
    "mm-1 min-1": ("M-1 s-1", 1e3 / 60.0), "mm^-1 min^-1": ("M-1 s-1", 1e3 / 60.0),
    "min-1 mm-1": ("M-1 s-1", 1e3 / 60.0), "min^-1 mm^-1": ("M-1 s-1", 1e3 / 60.0),
    "um-1 min-1": ("M-1 s-1", 1e6 / 60.0), "um^-1 min^-1": ("M-1 s-1", 1e6 / 60.0),
    "min-1 um-1": ("M-1 s-1", 1e6 / 60.0), "min^-1 um^-1": ("M-1 s-1", 1e6 / 60.0),
    # -- specific activity -> units per milligram ------------------------
    # One unit is one micromole of product per minute, so U/mg and
    # umol/min/mg are the same scale and mU/mg and nmol/min/mg are a
    # thousandth of it. The factors are definitional, not empirical.
    "u/mg": ("U/mg", 1.0), "units/mg": ("U/mg", 1.0),
    "umol/min/mg": ("U/mg", 1.0), "µmol/min/mg": ("U/mg", 1.0),
    "umol min-1 mg-1": ("U/mg", 1.0), "umol/(min*mg)": ("U/mg", 1.0),
    "mu/mg": ("U/mg", 1e-3), "nmol/min/mg": ("U/mg", 1e-3),
    "nmol min-1 mg-1": ("U/mg", 1e-3),
    # -- fractions -> percent --------------------------------------------
    "%": ("%", 1.0), "percent": ("%", 1.0), "pct": ("%", 1.0),
    # -- dimensionless ---------------------------------------------------
    "": ("", 1.0),
}


def normalise_unit_text(unit: str | None) -> str:
    """Collapse whitespace and case so two spellings of one unit compare equal.

    Spelling only. ``"U/mg"`` and ``"u/mg"`` are one unit; ``"U/mg"`` and
    ``"mU/mg"`` are two, and keeping them apart here is what sends them to
    :func:`canonical_unit` instead of letting them pool.
    """
    if unit is None:
        return ""
    return " ".join(str(unit).split()).strip().lower()


def canonical_unit(unit: str | None) -> tuple[str, float] | None:
    """``(canonical_unit, factor)``, or ``None`` for a unit not in the table.

    ``None`` is a refusal to compare, not an invitation to proceed: a caller
    that treats it as "no conversion needed" reintroduces exactly the bug this
    module exists to prevent.
    """
    if unit is None:
        return None
    return UNIT_TABLE.get(normalise_unit_text(unit))


def convert_measurement(value: float | None, unit: str | None
                        ) -> tuple[float | None, str] | None:
    """Convert one value to its canonical unit, or ``None`` if it cannot be."""
    canon = canonical_unit(unit)
    if canon is None:
        return None
    name, factor = canon
    return (None if value is None else value * factor), name


@dataclass(frozen=True)
class Reconciliation:
    """What a set of measurements looks like once they are on one scale.

    ``conflict`` is the load-bearing field. When it is non-empty the values
    were **not** reconciled and the caller must report the group as
    unmeasured, because the alternative -- aggregating anyway and noting the
    problem somewhere -- produces a number that reads like a measurement.
    """

    values: tuple[float, ...]
    unit: str
    converted: bool = False
    conflict: str = ""

    @property
    def usable(self) -> bool:
        return not self.conflict


def reconcile_units(
    pairs: Sequence[tuple[float | None, str | None]]
) -> Reconciliation:
    """Put a set of ``(value, unit)`` measurements onto one scale, or refuse.

    Three cases, in order:

    1. **one spelling** -- nothing to do, and nothing is assumed about whether
       the table knows the unit. A plate reporting ``AU/min`` throughout is
       perfectly aggregatable even though no conversion for it exists;
    2. **several spellings the table can relate** -- everything is converted
       to the shared canonical unit and :attr:`Reconciliation.converted` says
       so, so the record can state the unit it was actually aggregated in;
    3. **anything else** -- a unit the table does not know, or two units that
       canonicalise differently. :attr:`Reconciliation.conflict` names the
       spellings and the group reports no measurement. Guessing a factor here
       is how ``0.01 mM`` and ``10 uM`` become ``5.005``.
    """
    present = [(v, u) for v, u in pairs if v is not None]
    if not present:
        spellings = {normalise_unit_text(u): (u or "") for _, u in pairs}
        only = next(iter(spellings.values())) if len(spellings) == 1 else ""
        return Reconciliation(values=(), unit=only)

    by_spelling: dict[str, str] = {}
    for _, unit in present:
        by_spelling.setdefault(normalise_unit_text(unit), unit or "")

    if len(by_spelling) == 1:
        return Reconciliation(
            values=tuple(float(v) for v, _ in present),
            unit=next(iter(by_spelling.values())))

    converted: list[float] = []
    canon_names: set[str] = set()
    unknown: list[str] = []
    for value, unit in present:
        conv = convert_measurement(value, unit)
        if conv is None:
            unknown.append(unit or "(blank)")
            continue
        new_value, canon_name = conv
        assert new_value is not None
        converted.append(float(new_value))
        canon_names.add(canon_name)

    listed = ", ".join(repr(s) for s in sorted(by_spelling.values()))
    if unknown:
        return Reconciliation(
            values=(), unit="",
            conflict=(f"the measurements are reported in {listed} and "
                      f"{', '.join(sorted(set(unknown)))} is not in the "
                      f"conversion table, so they cannot be put on one scale"))
    if len(canon_names) > 1:
        return Reconciliation(
            values=(), unit="",
            conflict=(f"the measurements are reported in {listed}, which "
                      f"canonicalise to {', '.join(sorted(canon_names))} -- "
                      f"different quantities, not different scales of one"))
    return Reconciliation(values=tuple(converted),
                          unit=next(iter(canon_names)), converted=True)
