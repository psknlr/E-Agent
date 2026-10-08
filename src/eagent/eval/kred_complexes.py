"""From the reference set's 19 entries to reference complexes -- or to reasons.

:mod:`eagent.eval.kred_reference` says what the set contains and
:mod:`eagent.eval.kred_coordinates` pins the files. This module asks the
question the set was delivered for: *may any of these entries be a reference
complex for a shipped catalytic template's windows, and what do they say if so?*

It answers in three steps and keeps all three visible.

1. **Bindings** (:func:`derive_binding`). Which residue is the substrate, which
   is the cofactor, which atom is the electrophilic carbon, which the carbonyl
   oxygen, which conformer. These are derived by stated rules from the pinned
   coordinates -- never from "the ligand nearest the cofactor" -- and written
   to ``bindings/bindings.json`` with their rule and a review status, so a
   person can read exactly what was measured. A binding that cannot be derived
   is recorded with the reason, not skipped.

2. **Eligibility** (:func:`audit_entry`). A fixed list of checks per
   (entry, template): the workbook's own grade, the family, the cofactor's
   identity *and oxidation state*, a substrate (not a product) in the site, a
   matched activity measurement for the same variant, an explicit conformer,
   ligand density, and a derivable binding. Every check reports pass, fail or
   unknown with its reason; unknown fails closed. An entry is eligible only if
   all pass.

3. **Calibration** (:func:`run_scenario`). Eligible entries are reduced to one
   representative per *lineage* -- a tolerance interval counts independent
   draws, and the PDB entries of one enzyme are not that -- and handed to the
   project's own :func:`~eagent.tools.calibrate_windows.calibrate_template`,
   under a declared policy, in memory. Nothing is written to a calibration
   store, so no citation can result.

The scenarios (:data:`SCENARIOS`) are not alternatives to choose between. The
first is the template as shipped; the others each relax one requirement
(cofactor oxidation state, the workbook's conditional grades) to show how many
independent references the set could supply *at best*. A relaxed scenario's
numbers are measurements for a reader, never a window for a template.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..schemas.chem import CofactorState, cofactor_state_from_ligand_code
from ..schemas.templates import CatalyticTemplate
from ..science.calibration import (
    CalibrationPolicy, CalibrationRecord, coverage_at_confidence, minimum_actives,
)
from ..science.structure_io import Atom, Chain, Residue, Structure
from ..tools.calibrate_windows import ReferenceComplex, calibrate_template
from ..tools.evaluate_catalysis import (
    NICOTINAMIDE_HYDRIDE_DONOR_ATOM, PoseBinding, ResidueSelector,
)
from .kred_coordinates import family_class
from .kred_reference import (
    KineticRecord, ReferenceSet, StructureEntry, LigandValidation,
)

__all__ = [
    "BINDINGS_FILE",
    "BindingRecord",
    "Check",
    "EntryAudit",
    "AuditPolicy",
    "Scenario",
    "SCENARIOS",
    "AUDIT_POLICIES",
    "TEMPLATE_FAMILY_CLASS",
    "select_conformer",
    "find_carbonyl",
    "derive_binding",
    "derive_bindings",
    "write_bindings",
    "load_bindings",
    "audit_entry",
    "representatives",
    "run_scenario",
    "audit_report",
    "render_report",
    "build_excerpt",
    "write_excerpt",
]

BINDINGS_FILE = "bindings.json"

#: Template ``family_name`` -> the family class the RCSB annotation must place
#: an entity in (see :func:`eagent.eval.kred_coordinates.family_class`).
TEMPLATE_FAMILY_CLASS: dict[str, str] = {
    "SDR": "classical_sdr",
    "AKR": "aldo_keto_reductase",
    "MDR/ADH": "zinc_adh_mdr",
}

#: wwPDB component id of the oxidised form for each reduced cofactor code, and
#: back. Used only by the scenario that lets an oxidised crystallographic
#: cofactor stand in for the reduced one, to name the stand-in.
_OXIDISED_COUNTERPART = {"NDP": "NAP", "NAI": "NAD"}

#: Heavy-atom C=O distance limits, in Angstrom. A ketone C=O is 1.21-1.23; a
#: carboxylate C-O, which is delocalised, 1.25-1.27; an alcohol C-O 1.41-1.43.
#: The upper limit leaves room for a loosely refined ligand and still excludes
#: every C-O single bond.
_CARBONYL_MAX = 1.35
_SINGLE_BOND_MAX = 1.65


# ==========================================================================
# conformers
# ==========================================================================

def select_conformer(structure: Structure, altloc: str | None) -> Structure:
    """A copy of ``structure`` holding one conformer of every disordered atom.

    Atoms with no alternate location are kept; atoms with one are kept only if
    it is ``altloc``. The rule is applied to the whole model, protein and
    ligand together, because the alternate positions of a pocket residue and of
    the ligand beside it are usually correlated (6ZZO Ser148 and the substrate
    both move between A and B), and selecting A for one and the file's first
    listing for the other would mix conformers.

    With ``altloc=None`` the structure is returned **unchanged** and the caller
    must not measure a residue that has alternate locations: choosing the one
    listed first is the implicit selection this exists to remove. A residue that
    would be left with no atoms is dropped and named in ``parse_notes``.
    """
    if altloc is None:
        return structure
    chains: list[Chain] = []
    dropped_atoms = 0
    emptied: list[str] = []
    for chain in structure.chains:
        residues: list[Residue] = []
        for res in chain.residues:
            keep = [a for a in res.atoms if not a.altloc or a.altloc == altloc]
            dropped_atoms += len(res.atoms) - len(keep)
            if not keep:
                emptied.append(str(res))
                continue
            residues.append(Residue(chain=res.chain, resname=res.resname,
                                    resseq=res.resseq, icode=res.icode,
                                    atoms=keep, is_hetatm=res.is_hetatm))
        chains.append(Chain(chain_id=chain.chain_id, residues=residues))
    notes = list(structure.parse_notes)
    notes.append(f"conformer: alternate location {altloc!r} selected for the whole "
                 f"model; {dropped_atoms} atom(s) of other locations dropped"
                 + (f"; residue(s) with no {altloc!r} atoms removed: {emptied}"
                    if emptied else ""))
    return Structure(structure_id=structure.structure_id, chains=chains,
                     source_format=structure.source_format,
                     source_path=structure.source_path,
                     models_present=list(structure.models_present),
                     model_selected=structure.model_selected, parse_notes=notes)


# ==========================================================================
# bindings
# ==========================================================================

def _dist(a: Atom, b: Atom) -> float:
    return math.dist(a.coords, b.coords)


def find_carbonyl(residue: Residue, altloc: str | None = None
                  ) -> tuple[str, str] | tuple[None, str]:
    """``(carbon_name, oxygen_name)`` of the residue's one ketone C=O, or ``(None, reason)``.

    The rule, in full: take the heavy atoms of the chosen conformer. An oxygen
    is a *terminal carbonyl oxygen* if it has exactly one heavy neighbour within
    1.35 A and that neighbour is a carbon. A carbon is the *electrophile* if it
    has exactly one terminal oxygen and at least two other heavy neighbours
    (a carboxylate carbon has two terminal oxygens; an aldehyde or formyl group
    has one other neighbour; an alcohol carbon has no terminal oxygen). The
    residue qualifies only if exactly one carbon does.

    Zero or several is a refusal that says which: an alcohol (a product), a
    carboxylate only, a diketone. Nothing here looks at where the cofactor is,
    so the electrophile is never chosen *because* it is near a hydride donor.
    """
    atoms = [a for a in residue.heavy_atoms()
             if not a.altloc or altloc is None or a.altloc == altloc]
    if altloc is None and any(a.altloc for a in atoms):
        return None, ("the residue has alternate locations and none was chosen")
    neighbours: dict[str, list[Atom]] = {a.name: [] for a in atoms}
    for i, a in enumerate(atoms):
        for b in atoms[i + 1:]:
            d = _dist(a, b)
            if d <= _SINGLE_BOND_MAX:
                neighbours[a.name].append(b)
                neighbours[b.name].append(a)
    terminal_o: dict[str, Atom] = {}
    for a in atoms:
        if a.element.upper() != "O":
            continue
        near = [b for b in neighbours[a.name] if _dist(a, b) <= _CARBONYL_MAX]
        if len(neighbours[a.name]) == 1 and len(near) == 1 \
                and near[0].element.upper() == "C":
            terminal_o[a.name] = near[0]
    candidates: list[tuple[str, str]] = []
    for a in atoms:
        if a.element.upper() != "C":
            continue
        its_o = [o for o, c in terminal_o.items() if c.name == a.name]
        others = [b for b in neighbours[a.name] if b.name not in terminal_o]
        if len(its_o) == 1 and len(others) >= 2:
            candidates.append((a.name, its_o[0]))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None, ("no ketone C=O: no carbon has exactly one terminal oxygen and "
                      "two further heavy neighbours (an alcohol or a carboxylate-only "
                      "ligand is a product, not a substrate)")
    return None, (f"{len(candidates)} carbonyl carbons "
                  f"({', '.join(c for c, _ in candidates)}); the electrophile "
                  f"cannot be chosen without choosing")


@dataclass(frozen=True)
class BindingRecord:
    """What was measured on one entry, written down."""

    pdb_id: str
    chain: str
    #: Every alternate location the workbook selects for the site. One letter:
    #: that conformer is measured. Several: each is measured and shown, and
    #: they count as **one** observation (one crystal, one shared density fit).
    #: Empty: the site has no alternate locations.
    conformers: tuple[str, ...]
    substrate: dict[str, Any] | None
    substrate_atoms: dict[str, str]
    cofactor: dict[str, Any] | None
    cofactor_atoms: dict[str, str]
    failures: tuple[str, ...]
    notes: tuple[str, ...]
    derivation: str = ("find_carbonyl v1: one terminal carbonyl oxygen per "
                       "electrophilic carbon; hydride donor = C4N of the "
                       "nicotinamide component; residues named by the workbook's "
                       "scoring chain and RSCC summary")
    review_status: str = "rule_derived_unreviewed"

    @property
    def complete(self) -> bool:
        return (self.substrate is not None and self.cofactor is not None
                and {"electrophile", "carbonyl_O"} <= set(self.substrate_atoms)
                and "hydride_donor_C4" in self.cofactor_atoms)

    def to_dict(self) -> dict[str, Any]:
        return {"pdb_id": self.pdb_id, "chain": self.chain,
                "conformers": list(self.conformers), "substrate": self.substrate,
                "substrate_atoms": dict(sorted(self.substrate_atoms.items())),
                "cofactor": self.cofactor,
                "cofactor_atoms": dict(sorted(self.cofactor_atoms.items())),
                "failures": list(self.failures), "notes": list(self.notes),
                "derivation": self.derivation, "review_status": self.review_status}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BindingRecord":
        return cls(pdb_id=raw["pdb_id"], chain=raw["chain"],
                   conformers=tuple(raw.get("conformers") or ()),
                   substrate=raw.get("substrate"),
                   substrate_atoms=dict(raw.get("substrate_atoms") or {}),
                   cofactor=raw.get("cofactor"),
                   cofactor_atoms=dict(raw.get("cofactor_atoms") or {}),
                   failures=tuple(raw.get("failures") or ()),
                   notes=tuple(raw.get("notes") or ()),
                   derivation=raw.get("derivation", ""),
                   review_status=raw.get("review_status", "rule_derived_unreviewed"))

    def pose_binding(self, conformer: str | None = None) -> PoseBinding:
        """The :class:`PoseBinding` the project's evaluator reads.

        ``conformer`` only labels the pose; the conformer itself is selected on
        the structure by :func:`select_conformer`, before the binding is used.

        Protein roles are left empty on purpose: which residue is the catalytic
        tyrosine is a fact about each enzyme that a curator takes from its
        literature, and choosing it here by proximity to the ligand would make
        the later measurement of that distance circular. Every constraint that
        names a ``protein.*`` role is therefore *unmeasured*, not failed.
        """
        selector = (lambda d: None if d is None else ResidueSelector(
            chain=d["chain"], resname=d["resname"], resseq=d["resseq"],
            icode=d.get("icode", "")))
        return PoseBinding(
            pose_id=f"{self.pdb_id}:{self.chain}:{conformer or '-'}",
            substrate=selector(self.substrate),
            substrate_atoms=dict(self.substrate_atoms),
            cofactor=selector(self.cofactor),
            cofactor_atoms=dict(self.cofactor_atoms),
            protein_atoms={},
            notes=f"{self.review_status}; {self.derivation}")


def _scored_row(rows: Sequence[LigandValidation], ccds: Sequence[str],
                chain: str, target: float | None) -> tuple[LigandValidation | None, str]:
    """The validation row the workbook's summary was read from, or a reason."""
    pool = [r for r in rows if r.chain == chain and r.ccd in ccds]
    if not pool:
        return None, f"no validation row for {list(ccds)} in chain {chain}"
    if len({(r.chain, r.residue_number, r.ccd) for r in pool}) == 1:
        return pool[0], ""
    if target is not None:
        hit = [r for r in pool if r.rscc is not None and abs(r.rscc - target) <= 5e-4]
        if len({(r.chain, r.residue_number) for r in hit}) == 1:
            return hit[0], ""
    sites = sorted({f"{r.ccd}{r.residue_number}" for r in pool})
    return None, (f"{len(sites)} candidate residues ({', '.join(sites)}) and the "
                  f"summary RSCC does not single one out")


def derive_binding(entry: StructureEntry, structure: Structure,
                   rows: Sequence[LigandValidation]) -> BindingRecord:
    """Derive what to measure on one entry, or record why it cannot be."""
    failures: list[str] = []
    notes: list[str] = []
    chain = entry.scoring_chain
    conformers = tuple(entry.altlocs)
    substrate: dict[str, Any] | None = None
    cofactor: dict[str, Any] | None = None
    sub_atoms: dict[str, str] = {}
    cof_atoms: dict[str, str] = {}

    if not chain:
        return BindingRecord(entry.pdb_id, "", (), None, {}, None, {},
                             ("the workbook names no scoring chain: there is no "
                              "reaction ligand to bind",), tuple(notes))
    if len(conformers) > 1:
        notes.append(f"the workbook selects alternate locations {list(conformers)}: "
                     f"each is measured, and together they are one observation")
    if len(entry.reaction_ligand_ccds) != 1:
        failures.append(f"{len(entry.reaction_ligand_ccds)} reaction ligand components "
                        f"({'/'.join(entry.reaction_ligand_ccds) or 'none'}): one "
                        f"substrate is required, and a racemic or mixed site is not "
                        f"bound")
    else:
        lig_target = entry.ligand_rscc[0] if len(entry.ligand_rscc) == 1 else None
        row, why = _scored_row(rows, entry.reaction_ligand_ccds, chain, lig_target)
        if row is None:
            failures.append(f"substrate: {why}")
        else:
            substrate = {"chain": row.chain, "resname": row.ccd,
                         "resseq": row.residue_number, "icode": ""}
            residue = next((r for r in structure.residues()
                            if r.chain == row.chain and r.resseq == row.residue_number
                            and r.resname.upper() == row.ccd), None)
            if residue is None:
                failures.append(f"substrate {row.ccd}{row.residue_number} is not in the file")
            else:
                found: set[tuple[str, str]] = set()
                reasons: list[str] = []
                for conf in (conformers or (None,)):
                    carbon, oxygen = find_carbonyl(residue, conf)
                    if carbon is None:
                        reasons.append(f"conformer {conf or '-'}: {oxygen}")
                    else:
                        found.add((carbon, oxygen))
                if reasons:
                    failures.append("substrate atoms: " + "; ".join(reasons))
                elif len(found) != 1:
                    failures.append(f"substrate atoms differ between conformers: "
                                    f"{sorted(found)}")
                else:
                    carbon, oxygen = next(iter(found))
                    sub_atoms = {"electrophile": carbon, "carbonyl_O": oxygen}

    cof_rows = [r for r in rows if r.chain == chain
                and cofactor_state_from_ligand_code(r.ccd) is not CofactorState.UNKNOWN]
    codes = sorted({r.ccd for r in cof_rows})
    if not cof_rows:
        failures.append(f"cofactor: no nicotinamide cofactor row in chain {chain}")
    else:
        row, why = _scored_row(rows, codes, chain, entry.cofactor_rscc)
        if row is None:
            failures.append(f"cofactor: {why}")
        else:
            cofactor = {"chain": row.chain, "resname": row.ccd,
                        "resseq": row.residue_number, "icode": ""}
            donor = NICOTINAMIDE_HYDRIDE_DONOR_ATOM.get(row.ccd)
            if donor is None:
                failures.append(f"cofactor {row.ccd} is not in the donor-atom table")
            else:
                cof_atoms = {"hydride_donor_C4": donor}
    return BindingRecord(entry.pdb_id, chain, conformers, substrate, sub_atoms,
                         cofactor, cof_atoms, tuple(failures), tuple(notes))


def derive_bindings(rs: ReferenceSet, structures: Mapping[str, Structure]
                    ) -> list[BindingRecord]:
    """A binding record for every entry that names a scoring chain."""
    out: list[BindingRecord] = []
    for entry in rs.structures:
        if entry.pdb_id not in structures or not entry.scoring_chain:
            continue
        out.append(derive_binding(entry, structures[entry.pdb_id],
                                  rs.validation_rows(entry.pdb_id)))
    return out


def write_bindings(bindings: Iterable[BindingRecord], directory: str | Path) -> Path:
    out = Path(directory) / "bindings"
    out.mkdir(parents=True, exist_ok=True)
    path = out / BINDINGS_FILE
    path.write_text(json.dumps({
        "product": "kred_calibration_reference/bindings",
        "review_status_note": ("every binding is rule-derived from the pinned "
                               "coordinates and has not been reviewed by a person; "
                               "protein roles are deliberately unbound"),
        "bindings": [b.to_dict() for b in sorted(bindings, key=lambda b: b.pdb_id)],
    }, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_bindings(directory: str | Path) -> dict[str, BindingRecord]:
    path = Path(directory) / "bindings" / BINDINGS_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {b["pdb_id"]: BindingRecord.from_dict(b) for b in raw["bindings"]}


# ==========================================================================
# eligibility
# ==========================================================================

@dataclass(frozen=True)
class Check:
    name: str
    status: str          # "pass" | "fail" | "unknown"
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class AuditPolicy:
    """Which requirements an audit enforces. Each relaxation is a named choice."""

    name: str
    description: str
    allowed_grades: frozenset[str] = frozenset({"conditional_pose_reference"})
    #: ``template``: only the template's own cofactor components (NDP).
    #: ``oxidised_counterpart``: also the oxidised form of the same cofactor (NAP).
    #: ``any_nicotinamide``: NAD(H) or NADP(H), either oxidation state -- the
    #: template's cofactor requirement is dropped, its family requirement is not.
    cofactor: str = "template"
    min_rscc: float = 0.80
    require_single_conformer: bool = True
    allow_product_state: bool = False


AUDIT_POLICIES: dict[str, AuditPolicy] = {
    "strict": AuditPolicy(
        "strict", "the template as shipped: a reduced cofactor of the template's "
                  "own identity, the workbook's conditional pose grade, one conformer"),
    "oxidised_same_cofactor": AuditPolicy(
        "oxidised_same_cofactor",
        "the oxidised form of the template's own cofactor (NADP+ for NADPH) is "
        "accepted. The workbook says a model that does this must be saved as a "
        "derived version; here it is only measured",
        cofactor="oxidised_counterpart"),
    "any_nicotinamide_cofactor": AuditPolicy(
        "any_nicotinamide_cofactor",
        "the template's cofactor requirement is dropped: NAD(H) or NADP(H), in "
        "either oxidation state. The family requirement stays",
        cofactor="any_nicotinamide"),
    "every_graded_pose": AuditPolicy(
        "every_graded_pose",
        "as any_nicotinamide_cofactor, and every grade that places a ligand in the "
        "site is accepted: review_clash entries (both conformers) and the "
        "product-state structure. The most the set could supply if no judgement "
        "were applied",
        allowed_grades=frozenset({"conditional_pose_reference", "review_clash",
                                  "conditional_product_reference"}),
        cofactor="any_nicotinamide", require_single_conformer=False,
        allow_product_state=True),
}


def _matched_kinetics(rs: ReferenceSet, entry: StructureEntry) -> list[KineticRecord]:
    return [k for k in rs.kinetics
            if entry.pdb_id in k.experimental_complex_pdb_ids
            and k.variant == entry.variant]


def audit_entry(rs: ReferenceSet, entry: StructureEntry, template: CatalyticTemplate,
                *, manifest: Mapping[str, Any], binding: BindingRecord | None,
                policy: AuditPolicy) -> "EntryAudit":
    """Run every check for one (entry, template) under one policy."""
    checks: list[Check] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        checks.append(Check(name, "unknown" if ok is None else ("pass" if ok else "fail"),
                            detail))

    # 1. the workbook's grade
    add("workbook_grade", entry.geometry_use in policy.allowed_grades,
        f"geometry_use is {entry.geometry_use!r} ({entry.role}); this policy "
        f"accepts {sorted(policy.allowed_grades)}")

    # 2. family
    pin = (manifest.get("files") or {}).get(entry.pdb_id) or {}
    have = pin.get("family_class") or family_class(pin.get("annotations"))
    want = TEMPLATE_FAMILY_CLASS.get(template.family_name)
    if want is None:
        add("family", None, f"the template family {template.family_name!r} has no "
                            f"family class in TEMPLATE_FAMILY_CLASS")
    else:
        add("family", have == want,
            f"the RCSB annotates the scored entity as {have}; template "
            f"{template.template_id} is for {want}")

    # 3. cofactor identity and oxidation state (from the component id in the file)
    code = binding.cofactor["resname"] if binding and binding.cofactor else None
    if code is None:
        add("cofactor", None, "no cofactor residue could be bound in the scoring chain")
    else:
        state = cofactor_state_from_ligand_code(code)
        template_codes = {c.upper() for c in template.cofactor_ligand_codes}
        required = getattr(template.required_cofactor_state, "value",
                           str(template.required_cofactor_state))
        if policy.cofactor == "template":
            accepted = set(template_codes)
        elif policy.cofactor == "oxidised_counterpart":
            accepted = template_codes | {_OXIDISED_COUNTERPART.get(c, c)
                                         for c in template_codes}
        else:
            accepted = {code.upper()} if state is not CofactorState.UNKNOWN else set()
        ok = code.upper() in accepted
        article = "an" if state.value[:1] in "aeiou" else "a"
        why = (f"component {code} is {article} {state.value} nicotinamide cofactor; "
               f"the template requires {sorted(template_codes)} ({required})")
        if ok and code.upper() not in template_codes:
            why += (f"; ACCEPTED ONLY UNDER THIS POLICY ({policy.cofactor}) as a "
                    f"stand-in for {sorted(template_codes)}, which it is not")
        add("cofactor", ok, why)

    # 4. a substrate, not a product, in the site
    if entry.geometry_use == "conditional_product_reference" and not policy.allow_product_state:
        add("substrate_state", False,
            f"{entry.bound_reaction_ligand} is the product; a product-state structure "
            f"is not a pre-reaction geometry")
    elif not entry.has_reaction_ligand:
        add("substrate_state", False, "no reaction ligand in the entry")
    else:
        add("substrate_state", True,
            f"{entry.bound_reaction_ligand} ({'/'.join(entry.reaction_ligand_ccds)}); "
            f"state: {entry.experimental_state}")

    # 5. matched activity for this exact variant, and the same substrate
    matched = _matched_kinetics(rs, entry)
    live = [k for k in matched
            if k.record_status != "not_determined"
            and ((k.kcat_s or 0) > 0 or (k.efficiency_reported_m_s or 0) > 0)]
    if not matched:
        add("activity_evidence", False,
            "no kinetic record names this entry as its experimental complex for this "
            "variant; a structure of a binder is not an active reference")
    elif not live:
        add("activity_evidence", False, f"matched records {[k.label_id for k in matched]} "
                                        f"carry no positive measured value")
    else:
        same = [k for k in live if k.substrate.lower()
                == entry.bound_reaction_ligand.lower()]
        if same:
            add("activity_evidence", True,
                "; ".join(f"{k.label_id} [{k.tier}, {k.label_type}, kcat "
                          f"{k.kcat_s if k.kcat_s is not None else 'n/a'} s^-1]"
                          for k in same))
        else:
            add("activity_evidence", False,
                f"activity was measured for {sorted({k.substrate for k in live})}, "
                f"the ligand in the site is {entry.bound_reaction_ligand}")

    # 6. conformer
    if policy.require_single_conformer:
        if entry.altlocs and len(entry.altlocs) != 1:
            add("conformer", False, f"the workbook lists locations {entry.altlocs}: "
                                    f"an explicit single conformer is required")
        elif entry.altlocs:
            add("conformer", True, f"alternate location {entry.altlocs[0]} named by the workbook "
                                   f"(shared density fit; occupancy-weighted, not independent)")
        else:
            add("conformer", True, "no alternate locations at the bound ligand")
    else:
        add("conformer", True, f"{list(entry.altlocs) or 'none'}; each conformer is measured "
                               f"separately and counted as one observation")

    # 7. density
    values = list(entry.ligand_rscc) + ([entry.cofactor_rscc] if entry.cofactor_rscc is not None else [])
    if not values:
        add("density", None, "the workbook records no RSCC for this entry")
    else:
        worst = min(values)
        add("density", worst >= policy.min_rscc,
            f"lowest of ligand/cofactor RSCC is {worst:.3f}; minimum {policy.min_rscc}")

    # 8. a derivable binding
    if binding is None:
        add("binding", None, "no binding was derived for this entry")
    elif binding.complete:
        add("binding", True,
            f"substrate {binding.substrate['resname']}{binding.substrate['resseq']} "
            f"{binding.substrate_atoms}; cofactor {binding.cofactor['resname']}"
            f"{binding.cofactor['resseq']} {binding.cofactor_atoms}")
    else:
        add("binding", False, "; ".join(binding.failures) or "incomplete binding")

    failed = [c for c in checks if c.status != "pass"]
    return EntryAudit(entry.pdb_id, template.template_id, policy.name, tuple(checks),
                      not failed, tuple(f"{c.name}: {c.detail}" for c in failed))


@dataclass(frozen=True)
class EntryAudit:
    pdb_id: str
    template_id: str
    policy: str
    checks: tuple[Check, ...]
    eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"pdb_id": self.pdb_id, "template_id": self.template_id,
                "policy": self.policy, "eligible": self.eligible,
                "checks": [c.to_dict() for c in self.checks],
                "reasons": list(self.reasons)}


# ==========================================================================
# scenarios and the calibration run
# ==========================================================================

@dataclass(frozen=True)
class Scenario:
    name: str
    audit_policy: str
    template_id: str
    calibration_policies: tuple[CalibrationPolicy, ...]
    purpose: str


_POLICY_MODEST = CalibrationPolicy(
    coverage=0.80, confidence=0.80, min_inactives=5, max_inactive_inside_upper=0.50,
    source=("audit default chosen by the build agent for this report, not by a "
            "campaign: a modest bar (80 % coverage at 80 % confidence, at least five "
            "known inactives, upper Wilson bound on the inactive fraction inside the "
            "window at most 0.5), so that failing it says something. A stricter "
            "policy fails by more."))
_POLICY_STRICT = CalibrationPolicy(
    coverage=0.90, confidence=0.90, min_inactives=10, max_inactive_inside_upper=0.30,
    source=("audit default chosen by the build agent for this report, not by a "
            "campaign: the conventional '90 % coverage at 90 % confidence' tolerance "
            "claim, at least ten known inactives, upper bound 0.3"))

SDR_TEMPLATE_ID = "cat.sdr.nadph_carbonyl_reduction.v1"

SCENARIOS: tuple[Scenario, ...] = (
    Scenario("as_shipped", "strict", SDR_TEMPLATE_ID, (_POLICY_MODEST, _POLICY_STRICT),
             "the shipped NADPH-SDR template and the set's conditional grades, "
             "with nothing relaxed. This is the calibration the template could cite."),
    Scenario("oxidised_same_cofactor", "oxidised_same_cofactor", SDR_TEMPLATE_ID,
             (_POLICY_MODEST, _POLICY_STRICT),
             "NADP+ accepted for NADPH. Relaxes the oxidation state only. Measured "
             "for a reader; the result cannot be cited."),
    Scenario("any_nicotinamide_cofactor", "any_nicotinamide_cofactor", SDR_TEMPLATE_ID,
             (_POLICY_MODEST, _POLICY_STRICT),
             "the cofactor requirement dropped (NAD(H)/NADP(H), either state), the "
             "family requirement kept. Measured for a reader; cannot be cited."),
    Scenario("every_graded_pose", "every_graded_pose", SDR_TEMPLATE_ID,
             (_POLICY_MODEST, _POLICY_STRICT),
             "everything the workbook places a substrate or product in the site "
             "for, both conformers, any nicotinamide cofactor. The ceiling of what "
             "the set could supply; cannot be cited."),
)


def _rank(entry: StructureEntry) -> tuple[int, float]:
    """Order for choosing one representative per lineage: grade first, then the
    worse of the two densities (higher is better), then the PDB id for a stable
    tie-break. Resolution is deliberately not used: it says nothing about the
    ligand, which is what the workbook's own audit found."""
    grade = {"conditional_pose_reference": 0, "review_clash": 1,
             "conditional_product_reference": 2}.get(entry.geometry_use, 3)
    vals = list(entry.ligand_rscc) + ([entry.cofactor_rscc] if entry.cofactor_rscc is not None else [])
    return grade, -(min(vals) if vals else 0.0)


def representatives(entries: Iterable[StructureEntry]) -> list[StructureEntry]:
    """One entry per lineage: the best-graded, so n counts lineages, not files."""
    best: dict[str, StructureEntry] = {}
    for e in sorted(entries, key=lambda e: (_rank(e), e.pdb_id)):
        best.setdefault(e.lineage, e)
    return sorted(best.values(), key=lambda e: e.pdb_id)


def _evidence_for(rs: ReferenceSet, entry: StructureEntry) -> str:
    matched = _matched_kinetics(rs, entry)
    labels = ", ".join(f"{k.label_id} ({rs.source(k.source_id).title[:50]}; {k.source_location})"
                       for k in matched) or "no matched kinetic record"
    return (f"{entry.pdb_id} chain {entry.scoring_chain}: workbook grade "
            f"{entry.geometry_use}; ligand RSCC {list(entry.ligand_rscc)}, cofactor "
            f"RSCC {entry.cofactor_rscc}; activity label(s): {labels}")


def run_scenario(rs: ReferenceSet, scenario: Scenario, template: CatalyticTemplate, *,
                 manifest: Mapping[str, Any], bindings: Mapping[str, BindingRecord],
                 structures: Mapping[str, Structure],
                 full_checks: bool = True) -> dict[str, Any]:
    """Audit every entry, measure the eligible ones, calibrate on representatives.

    Every eligible entry is measured, every conformer the workbook selects, and
    shown. Only the **representatives** -- one entry per lineage, and the first
    conformer of it -- enter the calibration, because a tolerance interval
    counts independent draws and a second PDB entry of the same enzyme, or the
    second conformer of one ligand, is not one.
    """
    policy = AUDIT_POLICIES[scenario.audit_policy]
    audits = [audit_entry(rs, e, template, manifest=manifest,
                          binding=bindings.get(e.pdb_id), policy=policy)
              for e in rs.structures]
    eligible = [rs.structure(a.pdb_id) for a in audits if a.eligible]
    reps = representatives(eligible)
    rep_ids = {e.pdb_id for e in reps}

    windows = {c.name: c.window() for c in template.geometry_constraints}
    calibration_refs: list[ReferenceComplex] = []
    measured: list[dict[str, Any]] = []
    for entry in sorted(eligible, key=lambda e: e.pdb_id):
        binding = bindings[entry.pdb_id]
        for k, conf in enumerate(binding.conformers or (None,)):
            ref = ReferenceComplex(
                reference_id=f"{entry.pdb_id}:{entry.scoring_chain}:{conf or '-'}",
                label="active", evidence=_evidence_for(rs, entry),
                binding=binding.pose_binding(conf),
                structure=select_conformer(structures[entry.pdb_id], conf))
            counted = entry.pdb_id in rep_ids and k == 0
            values = _measure_only(ref, template)
            measured.append({
                "reference_id": ref.reference_id, "lineage": entry.lineage,
                "conformer": conf, "representative": entry.pdb_id in rep_ids,
                "in_calibration": counted, "measurements": values,
                # The shipped, uncalibrated window each value is compared with.
                # Informational: an uncalibrated window rejects nothing, and the
                # comparison is of a measurement with a window nobody fitted.
                "inside_shipped_window": {
                    name: (None if v is None else
                           bool(windows[name][0] <= v <= windows[name][1]))
                    for name, v in values.items()}})
            if counted:
                calibration_refs.append(ref)

    report: dict[str, Any] = {
        "scenario": scenario.name, "purpose": scenario.purpose,
        "audit_policy": {"name": policy.name, "description": policy.description,
                         "min_rscc": policy.min_rscc},
        "template_id": template.template_id,
        # Every check of every entry once (the first scenario); afterwards only
        # the verdict and the reasons, which is what differs between scenarios.
        "entries": [a.to_dict() if full_checks else
                    {"pdb_id": a.pdb_id, "eligible": a.eligible,
                     "reasons": list(a.reasons)} for a in audits],
        "eligible_entries": [e.pdb_id for e in eligible],
        "eligible_lineages": sorted({e.lineage for e in eligible}),
        "representatives": sorted(rep_ids),
        "n_independent_actives": len(calibration_refs),
        "n_known_inactives": 0,
        "known_inactives_note": (
            "the set contains no complex shown not to turn over: its 'activity_only' "
            "mutants have measured kcat > 0 and no matched structure, and an ND is "
            "not a zero"),
        "measurements": measured,
        "shipped_windows": {name: {"min": None if math.isinf(lo) else lo,
                                   "max": None if math.isinf(hi) else hi}
                            for name, (lo, hi) in sorted(windows.items())},
        "calibrations": [],
    }
    for pol in scenario.calibration_policies:
        calib = calibrate_template(calibration_refs, template, pol, store=None)
        report["calibrations"].append({
            "policy": pol.to_dict(),
            "records": {name: _record_summary(r)
                        for name, r in sorted(calib.records.items())},
            "unmeasurable": dict(sorted(calib.unmeasurable.items())),
        })
    return report


def _measure_only(ref: ReferenceComplex, template: CatalyticTemplate) -> dict[str, Any]:
    from ..tools.calibrate_windows import measure_references

    out: dict[str, Any] = {}
    for name, observations in measure_references([ref], template).items():
        out[name] = observations[0].value
    return out


def _record_summary(rec: CalibrationRecord) -> dict[str, Any]:
    v = rec.verdict
    return {"window": None if rec.window is None else list(rec.window),
            "meets_policy": v.meets_policy, "n_active": v.n_active,
            "n_inactive": v.n_inactive, "n_actives_needed": v.n_actives_needed,
            "achieved_confidence": round(v.achieved_confidence, 4),
            "coverage_supported": round(v.coverage_supported, 4),
            "reasons": list(v.reasons),
            "exclusions": [e.to_dict() for e in rec.exclusions]}


def audit_report(rs: ReferenceSet, templates: Mapping[str, CatalyticTemplate], *,
                 manifest: Mapping[str, Any], bindings: Mapping[str, BindingRecord],
                 structures: Mapping[str, Structure],
                 identity_counts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The whole audit as one JSON-able document, deterministic given its inputs."""
    template = templates[SDR_TEMPLATE_ID]
    scenarios = [run_scenario(rs, s, template, manifest=manifest, bindings=bindings,
                              structures=structures, full_checks=(i == 0))
                 for i, s in enumerate(SCENARIOS)]
    needed = {f"coverage {p.coverage} at confidence {p.confidence}":
              minimum_actives(p.coverage, p.confidence)
              for p in (_POLICY_MODEST, _POLICY_STRICT)}
    two = {f"{c}@{conf}": round(coverage_at_confidence(2, conf), 4)
           for c, conf in ((0.8, 0.8), (0.9, 0.9))}
    # Which shipped template each entry could possibly serve, family and cofactor
    # alone: the matrix behind "this set cannot calibrate the other two templates".
    applicability: dict[str, dict[str, Any]] = {}
    for tid, tpl in sorted(templates.items()):
        rows = {}
        for entry in rs.structures:
            pin = (manifest.get("files") or {}).get(entry.pdb_id) or {}
            want = TEMPLATE_FAMILY_CLASS.get(tpl.family_name)
            b = bindings.get(entry.pdb_id)
            code = b.cofactor["resname"] if b and b.cofactor else None
            rows[entry.pdb_id] = {
                "family": pin.get("family_class"), "template_family": want,
                "family_match": pin.get("family_class") == want,
                "cofactor_code": code,
                "cofactor_in_template": code in {c.upper() for c in tpl.cofactor_ligand_codes}
                if code else None}
        applicability[tid] = {
            "family_name": tpl.family_name,
            "entries_matching_family_and_cofactor": sorted(
                p for p, r in rows.items() if r["family_match"] and r["cofactor_in_template"]),
            "entries": rows}
    return {
        "reference_set": {"directory": str(rs.directory.name), "data_digest": rs.manifest_digest,
                          "counts": rs.counts(), "independence": rs.independence()},
        "lineage_counts_by_identity_threshold": dict(identity_counts or {}),
        "scenarios": scenarios,
        "template_applicability": applicability,
        "what_a_calibration_needs": {
            "independent_active_lineages": needed,
            "supported_by_two_actives_coverage_at_confidence": two,
            "note": ("a Wilks tolerance interval counts independent draws; the set "
                     "supplies one or two lineages with a placed substrate, and no "
                     "complex shown not to turn over"),
        },
    }


def render_report(report: Mapping[str, Any]) -> str:
    """The audit as text a person reads before they read the JSON."""
    out: list[str] = []
    rs = report["reference_set"]
    out.append("KRED calibration reference set v0.1 -- eligibility audit and calibration run")
    out.append(f"data digest {rs['data_digest'][:16]}...  "
               f"{rs['counts']['structures']} structures, "
               f"{rs['counts']['kinetic_records']} kinetic records")
    ind = rs["independence"]
    out.append(f"independent lineages: {ind['n_independent_structure_lineages']} among the "
               f"{ind['structure_entries']} entries; {ind['n_independent_core_lineages']} "
               f"behind the {ind['core_records']} core kinetic records; "
               f"{ind['n_independent_pose_lineages']} with a substrate or product placed "
               f"in the site")
    out.append("")
    for sc in report["scenarios"]:
        out.append(f"== scenario: {sc['scenario']} ==")
        out.append(f"   {sc['purpose']}")
        out.append(f"   eligible entries: {sc['eligible_entries'] or 'none'}; "
                   f"lineages: {sc['eligible_lineages'] or 'none'}; "
                   f"independent actives: {sc['n_independent_actives']}; "
                   f"known inactives: {sc['n_known_inactives']}")
        for m in sc["measurements"]:
            vals = ", ".join(
                f"{k}={format(v, '.2f')}"
                f"[{'inside' if m['inside_shipped_window'][k] else 'OUTSIDE'} the shipped window]"
                for k, v in sorted(m.get("measurements", {}).items()) if v is not None)
            unmeasured = sorted(k for k, v in m.get("measurements", {}).items() if v is None)
            where = ("in calibration" if m["in_calibration"] else
                     "not counted: " + ("same lineage as a better-graded entry"
                                        if not m["representative"]
                                        else "second conformer of one entry"))
            out.append(f"   measured {m['reference_id']} ({where}): {vals}")
            if unmeasured:
                out.append(f"       unmeasured (no protein residue is bound): {', '.join(unmeasured)}")
        for cal in sc["calibrations"]:
            p = cal["policy"]
            out.append(f"   policy coverage {p['coverage']} / confidence {p['confidence']} / "
                       f"min inactives {p['min_inactives']}:")
            for name, rec in cal["records"].items():
                state = "CALIBRATED" if rec["meets_policy"] else "not calibrated"
                out.append(f"     {name}: {state}; window {rec['window']}; "
                           f"n_active={rec['n_active']} n_inactive={rec['n_inactive']}; "
                           f"needs {rec['n_actives_needed']} actives")
        out.append("")
    out.append("== per-entry reasons, as shipped ==")
    first = report["scenarios"][0]
    for e in first["entries"]:
        out.append(f"   {e['pdb_id']}: {'ELIGIBLE' if e['eligible'] else 'not eligible'}")
        for r in e["reasons"]:
            out.append(f"       - {r}")
    out.append("")
    need = report["what_a_calibration_needs"]
    out.append("== what a calibration needs ==")
    for k, v in need["independent_active_lineages"].items():
        out.append(f"   {k}: at least {v} independent active references")
    out.append(f"   two actives support at most: {need['supported_by_two_actives_coverage_at_confidence']}")
    out.append(f"   {need['note']}")
    return "\n".join(out) + "\n"


# ==========================================================================
# excerpts: small real-coordinate fixtures with their provenance
# ==========================================================================

def _cif_token(value: Any) -> str:
    text = str(value)
    if text == "":
        return "."
    if any(c in text for c in " \t'\""):
        return f'"{text}"' if "'" in text and '"' not in text else f"'{text}'"
    return text


def build_excerpt(structure: Structure, binding: BindingRecord, *, radius: float = 8.0
                  ) -> list[Atom]:
    """The atoms a hermetic test needs: the bound substrate and cofactor, and
    every residue with an atom within ``radius`` of the substrate (all alternate
    locations kept, all chains)."""
    chain = binding.chain
    keep_keys: set[tuple[str, int, str, str]] = set()
    sub = binding.substrate
    sub_atoms: list[Atom] = []
    if sub is not None:
        for r in structure.residues():
            if r.chain == sub["chain"] and r.resseq == sub["resseq"] \
                    and r.resname.upper() == sub["resname"]:
                sub_atoms = r.heavy_atoms()
                keep_keys.add(r.key)
    if binding.cofactor is not None:
        c = binding.cofactor
        for r in structure.residues():
            if r.chain == c["chain"] and r.resseq == c["resseq"] \
                    and r.resname.upper() == c["resname"]:
                keep_keys.add(r.key)
    for r in structure.residues():
        if r.key in keep_keys or r.is_water:
            continue
        if any(math.dist(a.coords, s.coords) <= radius for a in r.heavy_atoms()
               for s in sub_atoms):
            keep_keys.add(r.key)
    return [a for r in structure.residues() if r.key in keep_keys for a in r.atoms]


def write_excerpt(atoms: Sequence[Atom], path: str | Path, *, pdb_id: str,
                  provenance: Sequence[str]) -> Path:
    """Write ``atoms`` as a minimal mmCIF that :func:`read_mmcif` reads back."""
    lines = [f"data_{pdb_id}", "#", *(f"# {p}" for p in provenance), "#",
             f"_entry.id {pdb_id}", "#", "loop_"]
    tags = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
            "label_comp_id", "label_asym_id", "label_seq_id", "pdbx_PDB_ins_code",
            "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv",
            "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id",
            "pdbx_PDB_model_num"]
    lines += [f"_atom_site.{t}" for t in tags]
    for a in atoms:
        lines.append(" ".join([
            "HETATM" if a.is_hetatm else "ATOM", str(a.serial), _cif_token(a.element),
            _cif_token(a.name), _cif_token(a.altloc), _cif_token(a.resname),
            _cif_token(a.chain), "." if a.is_hetatm else str(a.resseq),
            _cif_token(a.icode), f"{a.x:.3f}", f"{a.y:.3f}", f"{a.z:.3f}",
            "?" if a.occupancy is None else f"{a.occupancy:.2f}",
            "?" if a.bfactor_or_plddt is None else f"{a.bfactor_or_plddt:.2f}",
            str(a.resseq), _cif_token(a.resname), _cif_token(a.chain),
            _cif_token(a.name), str(a.model)]))
    lines.append("#")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
