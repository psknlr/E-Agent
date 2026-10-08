"""Pinning the coordinates the reference set is about.

The workbook describes 19 PDB entries and grades each one. A grade is a claim
about a file -- "chain B, altloc A, ligand RSCC 0.909" -- and the RCSB revises
entries, so the claim is only about the bytes it was made on. This module is how
those bytes are named: each of the 19 asymmetric-unit mmCIF files is fetched by
:meth:`eagent.connectors.structure.RCSBPDBConnector.download_structure` (which
refuses to call out unless the file host has been probed), hashed, and recorded
in ``coordinates/coordinates.manifest.json`` together with the handful of
facts the workbook's claims can be checked against.

The files themselves are **not** committed. They are public-domain (CC0) but are
a few megabytes each, and a hash plus a verified route is enough to recreate
them; what is committed is the pin. A run that needs them fetches into a cache
directory, and a cached file whose hash is not the pinned one is an error that
names both, never a file that is quietly replaced.

WHAT IS CHECKED AGAINST THE FILE
================================
:func:`verify_coordinates` reads each file and checks, with no network:

* the hash is the pinned one;
* the resolution and the method are the workbook's;
* every row of the workbook's per-ligand validation table names a residue that
  is really in the file -- same chain, same author number, same component --
  and, where the row names an alternate location, an atom with that location;
* the reaction ligand and the cofactor of each graded entry are present in the
  scoring chain;
* the cofactor's oxidation state, read from its wwPDB component id, agrees with
  the state the workbook wrote in words (NAD is NAD+, NDP is NADPH).

That last group is what the workbook's density and state grades rest on, so a
file that does not satisfy it is not "the entry the audit looked at".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..connectors.structure import RCSBPDBConnector, StructureFile, validate_mmcif_bytes
from ..schemas.chem import CofactorState, cofactor_state_from_ligand_code
from ..science.numbering import needleman_wunsch
from ..science.structure_io import Structure, mmcif_categories, read_mmcif
from .kred_reference import (
    Finding, LINEAGE_GROUPS, LINEAGE_IDENTITY_THRESHOLD, ReferenceSet,
    REFERENCE_VERSION,
)

__all__ = [
    "COORDINATES_MANIFEST",
    "default_cache_dir",
    "entry_facts",
    "fetch_coordinates",
    "load_coordinates_manifest",
    "write_coordinates_manifest",
    "verify_coordinates",
    "family_class",
    "scoring_entity_id",
    "enzyme_sequences",
    "sequence_identities",
    "lineage_findings",
    "lineage_counts",
]

COORDINATES_MANIFEST = "coordinates.manifest.json"

_FACT_CATEGORIES = (
    "_entry", "_struct", "_struct_keywords", "_exptl", "_refine", "_reflns",
    "_entity", "_entity_poly", "_pdbx_audit_revision_history",
)

#: What a cofactor state written in words must be, given the component id in the
#: file. Words are the workbook's; the table is the project's
#: (:func:`~eagent.schemas.chem.cofactor_state_from_ligand_code`).
_STATE_WORDS = {
    CofactorState.OXIDIZED: ("NAD+", "NADP+"),
    CofactorState.REDUCED: ("NADH", "NADPH"),
}

#: The word for the cofactor each component id denotes. The check is on identity
#: as well as oxidation state: NADH and NADPH are both reduced, and a workbook
#: that calls an NDP cofactor "NADH" has described a different cofactor.
_CODE_WORD = {"NDP": "NADPH", "NAI": "NADH", "NAP": "NADP+", "NAD": "NAD+"}


def default_cache_dir() -> Path:
    """Where fetched coordinate files live: under the repository's ignored cache."""
    return Path(__file__).resolve().parents[3] / ".cache" / "kred_coordinates"


def _first_number(*values: str | None) -> float | None:
    for v in values:
        if v is None or v in (".", "?"):
            continue
        try:
            return float(v)
        except ValueError:
            continue
    return None


def entry_facts(text: str, pdb_id: str) -> dict[str, Any]:
    """The facts about one entry that the workbook's claims can be checked on.

    Everything here is read from the file; nothing is looked up. Alternate
    locations are listed per ligand residue rather than summarised, because
    which conformer a claim is about is the whole question for several entries.
    """
    pid = pdb_id.strip().upper()
    cats = mmcif_categories(text, _FACT_CATEGORIES)

    def one(category: str) -> Mapping[str, str]:
        rows = cats.get(category) or [{}]
        return rows[0]

    structure = read_mmcif(text, structure_id=pid)
    ligands = []
    for res in structure.ligands():
        occ = [a.occupancy for a in res.atoms if a.occupancy is not None]
        ligands.append({
            "chain": res.chain, "resseq": res.resseq, "icode": res.icode,
            "ccd": res.resname, "n_atoms": len(res.atoms),
            "altlocs": sorted({a.altloc for a in res.atoms if a.altloc}),
            "occupancy_min": min(occ) if occ else None,
            "occupancy_max": max(occ) if occ else None,
        })
    polymer_alt = sum(1 for r in structure.residues()
                      if not r.is_hetatm and not r.is_water and r.has_altloc())
    polymers = []
    for row in cats.get("_entity_poly") or []:
        seq = "".join((row.get("pdbx_seq_one_letter_code_can") or "").split())
        polymers.append({"entity_id": row.get("entity_id"),
                         "type": row.get("type"),
                         "chains": [c for c in (row.get("pdbx_strand_id") or "").split(",") if c],
                         "sequence_length": len(seq), "sequence": seq})
    revisions = cats.get("_pdbx_audit_revision_history") or []
    dates = sorted(r.get("revision_date", "") for r in revisions if r.get("revision_date"))
    return {
        "entry_id": one("_entry").get("id"),
        "title": one("_struct").get("title"),
        "keywords": one("_struct_keywords").get("pdbx_keywords"),
        "method": one("_exptl").get("method"),
        "resolution_a": _first_number(one("_refine").get("ls_d_res_high"),
                                      one("_reflns").get("d_resolution_high")),
        "entities": [{"id": r.get("id"), "type": r.get("type"),
                      "description": r.get("pdbx_description"),
                      "copies": r.get("pdbx_number_of_molecules")}
                     for r in cats.get("_entity") or []],
        "polymers": polymers,
        "chains": [c.chain_id for c in structure.chains],
        "n_atoms": structure.n_atoms(),
        "ligands": ligands,
        "polymer_residues_with_altloc": polymer_alt,
        "revision_count": len(revisions),
        "last_revision_date": dates[-1] if dates else None,
    }


def scoring_entity_id(facts: Mapping[str, Any], scoring_chain: str) -> str | None:
    """The polymer entity an entry is scored on: the one carrying the scoring
    chain, else the first polypeptide. ``None`` when the file has no polypeptide."""
    polymers = [p for p in facts.get("polymers") or []
                if str(p.get("type", "")).startswith("polypeptide")]
    for p in polymers:
        if scoring_chain and scoring_chain in p.get("chains", []):
            return str(p["entity_id"])
    return str(polymers[0]["entity_id"]) if polymers else None


#: ``class -> identifiers (InterPro / Pfam) that place an entity in it``. The
#: classes are the ones the shipped templates distinguish. They are read off the
#: RCSB's own annotation of the polymer entity; where the annotation carries
#: none of them the entity is ``unclassified``, never assumed to be a member.
_FAMILY_MARKERS: dict[str, frozenset[str]] = {
    "classical_sdr": frozenset({"IPR002347", "PF00106"}),
    "zinc_adh_mdr": frozenset({"IPR002328", "IPR013154"}),
    "extended_sdr_epimerase_dehydratase": frozenset({"IPR001509"}),
    "aldo_keto_reductase": frozenset({"IPR023210", "IPR001395", "PF00248"}),
}


def family_class(annotations: Mapping[str, Any] | None) -> str:
    """Which family an entity's RCSB annotation places it in.

    ``classical_sdr`` (InterPro IPR002347 / Pfam PF00106) is the family the
    shipped ``cat.sdr.*`` templates describe. An entity annotated as a
    zinc-type alcohol dehydrogenase, or as the NAD-dependent
    epimerase/dehydratase family, is a different catalytic arrangement and is
    named as such: applying the Ser-Tyr-Lys-Asn rules to it would be applying
    one family's mechanism to another family's structure.
    """
    if not annotations:
        return "unclassified"
    ids = {a["id"] for a in (annotations.get("interpro") or [])}
    ids |= {a["id"] for a in (annotations.get("pfam") or [])}
    go = {a["id"] for a in (annotations.get("go") or [])}
    if ids & _FAMILY_MARKERS["classical_sdr"]:
        return "classical_sdr"
    if ids & _FAMILY_MARKERS["zinc_adh_mdr"] or "GO:0008270" in go:
        return "zinc_adh_mdr"
    if ids & _FAMILY_MARKERS["extended_sdr_epimerase_dehydratase"]:
        return "extended_sdr_epimerase_dehydratase"
    if ids & _FAMILY_MARKERS["aldo_keto_reductase"]:
        return "aldo_keto_reductase"
    return "unclassified"


def fetch_coordinates(rs: ReferenceSet, connector: RCSBPDBConnector,
                      cache_dir: str | Path, *,
                      pinned: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fetch every entry of the set and return the manifest describing the files.

    ``pinned`` is a previous manifest. With it, each download must hash to the
    pinned value; without it the files are being pinned for the first time and
    the hashes are whatever the RCSB serves now. The RCSB's family annotation of
    the scored polymer entity is recorded beside each hash, because the
    eligibility audit needs it and the manifest is the copy that survives.
    """
    files: dict[str, Any] = {}
    previous = (pinned or {}).get("files") or {}
    for entry in rs.structures:
        expected = (previous.get(entry.pdb_id) or {}).get("sha256")
        got: StructureFile = connector.download_structure(
            entry.pdb_id, cache_dir, expected_sha256=expected)
        text = validate_mmcif_bytes(got.path.read_bytes(), entry.pdb_id)
        facts = entry_facts(text, entry.pdb_id)
        entity = scoring_entity_id(facts, entry.scoring_chain)
        annotations = (connector.polymer_entity_annotations(entry.pdb_id, entity)
                       if entity else None)
        files[entry.pdb_id] = {
            "sha256": got.sha256, "bytes": got.size_bytes,
            "url": got.source_url,
            "retrieved_at": got.retrieved_at
            or (previous.get(entry.pdb_id) or {}).get("retrieved_at"),
            "facts": facts,
            "annotations": annotations,
            "family_class": family_class(annotations),
        }
    return {
        "product": "kred_calibration_reference/coordinates",
        "version": REFERENCE_VERSION,
        "files": files,
        "licence": ("CC0 1.0 -- wwPDB: 'Data files contained in the PDB archive "
                    "are available under the CC0 1.0 Universal Public Domain "
                    "Dedication' (https://www.wwpdb.org/about/usage-policies, "
                    "read 2026-10-08)"),
        "route": ("eagent.connectors.structure.RCSBPDBConnector.download_structure"
                  " -> https://files.rcsb.org/download/<ID>.cif (the asymmetric "
                  "unit; checked against a passing probe before any request)"),
        "note": ("the files are not committed; they are fetched into a cache "
                 "directory and must hash to the values here. The RCSB revises "
                 "entries, so the same identifier can serve other bytes later."),
    }


def write_coordinates_manifest(manifest: Mapping[str, Any], directory: str | Path) -> Path:
    out = Path(directory) / "coordinates"
    out.mkdir(parents=True, exist_ok=True)
    path = out / COORDINATES_MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_coordinates_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "coordinates" / COORDINATES_MANIFEST
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class _Residue:
    chain: str
    resseq: int
    ccd: str
    altlocs: tuple[str, ...]


def _residues(structure: Structure) -> dict[tuple[str, int, str], _Residue]:
    out: dict[tuple[str, int, str], _Residue] = {}
    for res in structure.ligands():
        out[(res.chain, res.resseq, res.resname.upper())] = _Residue(
            res.chain, res.resseq, res.resname.upper(),
            tuple(sorted({a.altloc for a in res.atoms if a.altloc})))
    return out


def verify_coordinates(rs: ReferenceSet, cache_dir: str | Path,
                       manifest: Mapping[str, Any], *,
                       only: Iterable[str] | None = None
                       ) -> tuple[list[Finding], dict[str, Structure]]:
    """Check each cached file against its pin and against the workbook's claims.

    Returns ``(findings, structures)``; the structures are the parsed files, for
    a caller that goes on to measure them. A missing file is a finding, not an
    exception, so one absent entry does not hide the state of the other 18.
    ``only`` limits the check to the named entries (a test that damages one
    entry's claims need not parse all nineteen files to see it).
    """
    import hashlib

    findings: list[Finding] = []
    structures: dict[str, Structure] = {}
    pins = manifest.get("files") or {}

    def err(code: str, subject: str, message: str) -> None:
        findings.append(Finding("error", code, subject, message))

    wanted = None if only is None else {p.upper() for p in only}
    for entry in rs.structures:
        pid = entry.pdb_id
        if wanted is not None and pid not in wanted:
            continue
        pin = pins.get(pid)
        if pin is None:
            err("coordinates.unpinned", pid, "no hash is recorded for this entry")
            continue
        path = Path(cache_dir) / f"{pid}.cif"
        if not path.is_file():
            err("coordinates.absent", pid,
                f"{path} is not there; run `eagent reference fetch-coordinates`")
            continue
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest != pin["sha256"]:
            err("coordinates.hash", pid,
                f"{path.name} hashes to {digest[:16]}... but the pin is "
                f"{pin['sha256'][:16]}...; this is not the file the analysis "
                f"was pinned to")
            continue
        try:
            text = validate_mmcif_bytes(body, pid)
            structure = read_mmcif(text, structure_id=pid)
        except Exception as exc:                      # a parse refusal is a finding
            err("coordinates.unreadable", pid, str(exc))
            continue
        structures[pid] = structure
        facts = pin.get("facts") or {}

        res = facts.get("resolution_a")
        if res is None or abs(res - entry.resolution_a) > 0.0051:
            err("coordinates.resolution", pid,
                f"the file says {res}, the workbook {entry.resolution_a}")
        if (facts.get("method") or "").upper() != "X-RAY DIFFRACTION":
            findings.append(Finding("warning", "coordinates.method", pid,
                                    f"method is {facts.get('method')!r}"))

        present = _residues(structure)
        for row in rs.validation_rows(pid):
            hit = present.get((row.chain, row.residue_number, row.ccd))
            if hit is None:
                err("coordinates.validation_row", f"{pid}:{row.chain}{row.residue_number}",
                    f"the validation table has {row.ccd} at chain {row.chain} "
                    f"residue {row.residue_number}; the file has no such residue")
            elif row.altloc and row.altloc not in hit.altlocs:
                err("coordinates.altloc", f"{pid}:{row.chain}{row.residue_number}",
                    f"the validation row is for altloc {row.altloc}; the residue's "
                    f"locations in the file are {list(hit.altlocs) or 'none'}")

        if entry.scoring_chain:
            in_chain = {r.ccd for k, r in present.items() if r.chain == entry.scoring_chain}
            for ccd in entry.reaction_ligand_ccds:
                if ccd not in in_chain:
                    err("coordinates.ligand_absent", pid,
                        f"reaction ligand {ccd} is not in scoring chain "
                        f"{entry.scoring_chain}")
            cofactor_codes = sorted(
                c for c in in_chain
                if cofactor_state_from_ligand_code(c) is not CofactorState.UNKNOWN)
            for code in cofactor_codes:
                state = cofactor_state_from_ligand_code(code)
                words = ((_CODE_WORD[code],) if code in _CODE_WORD
                         else _STATE_WORDS.get(state, ()))
                if entry.cofactor_state and not any(w in entry.cofactor_state for w in words):
                    # A workbook that says the state is *ambiguous* is not
                    # contradicted by a file that deposits one component: it is
                    # worth a line, because the deposited code is then the only
                    # thing an automated reading would use.
                    sev = ("warning" if "ambig" in entry.cofactor_state.lower()
                           else "error")
                    findings.append(Finding(
                        sev, "coordinates.cofactor_state", pid,
                        f"the file's cofactor {code} is {state.value} "
                        f"({'/'.join(words)}); the workbook describes it as "
                        f"{entry.cofactor_state!r}"))
            if entry.cofactor_state and not cofactor_codes \
                    and entry.cofactor_state.split()[0] in ("NAD+", "NADP+", "NADH", "NADPH"):
                err("coordinates.cofactor_absent", pid,
                    f"the workbook says {entry.cofactor_state} but no nicotinamide "
                    f"cofactor component is in chain {entry.scoring_chain}")
    return findings, structures


# ==========================================================================
# lineages, from the pinned sequences
# ==========================================================================

def enzyme_sequences(rs: ReferenceSet, manifest: Mapping[str, Any]
                     ) -> dict[str, dict[str, str]]:
    """``enzyme -> {pdb_id: sequence}`` of the polypeptide each entry is scored on.

    The entity that carries the scoring chain, or the first polypeptide where
    no chain is named. Taken from the pinned manifest, so it needs no coordinate
    file and no network.
    """
    out: dict[str, dict[str, str]] = {}
    for entry in rs.structures:
        facts = (manifest.get("files") or {}).get(entry.pdb_id, {}).get("facts") or {}
        polymers = [p for p in facts.get("polymers") or []
                    if str(p.get("type", "")).startswith("polypeptide")]
        chosen = next((p for p in polymers
                       if not entry.scoring_chain or entry.scoring_chain in p["chains"]),
                      polymers[0] if polymers else None)
        if chosen and chosen.get("sequence"):
            out.setdefault(entry.enzyme, {})[entry.pdb_id] = chosen["sequence"]
    return out


def _identity(a: str, b: str) -> float:
    """Global identity scaled by the covered fraction of the shorter sequence."""
    alignment = needleman_wunsch(a, b)
    return alignment.identity * min(alignment.coverage_a(), alignment.coverage_b())


def _representative(rs: ReferenceSet, entries: Mapping[str, str]) -> str:
    """The entry whose sequence stands for an enzyme: wild type first, then by id."""
    return sorted(entries, key=lambda pid: (rs.structure(pid).variant != "WT", pid))[0]


def sequence_identities(rs: ReferenceSet, manifest: Mapping[str, Any]
                        ) -> dict[tuple[str, str], float]:
    """Coverage-adjusted identity between every pair of distinct enzymes.

    One representative entry per enzyme (a wild-type one where there is one).
    Within an enzyme the entries are the same protein up to engineered
    substitutions, which :func:`lineage_findings` checks separately.
    """
    sequences = enzyme_sequences(rs, manifest)
    names = sorted(sequences)
    reps = {n: sequences[n][_representative(rs, sequences[n])] for n in names}
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[(a, b)] = _identity(reps[a], reps[b])
    return out


def lineage_counts(rs: ReferenceSet, manifest: Mapping[str, Any]
                   ) -> dict[str, dict[str, Any]]:
    """How many independent groups the entries form at several identity levels.

    Single-linkage at each threshold, so a chain of 35 % pairs does not hide
    inside a 40 % rule: the numbers at 30 %, 40 % and 50 % are what the count
    is worth under each reading, and the 40 % one is the one the loader uses.
    """
    identities = sequence_identities(rs, manifest)
    enzymes = sorted({e.enzyme for e in rs.structures})
    out: dict[str, dict[str, Any]] = {}
    for threshold in (0.30, 0.40, 0.50):
        parent = {e: e for e in enzymes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # declared merges first: the groups the loader already uses
        by_group: dict[str, list[str]] = {}
        for e in enzymes:
            by_group.setdefault(LINEAGE_GROUPS[e], []).append(e)
        for members in by_group.values():
            for other in members[1:]:
                parent[find(other)] = find(members[0])
        for (a, b), ident in identities.items():
            if ident >= threshold:
                parent[find(a)] = find(b)
        groups: dict[str, list[str]] = {}
        for e in enzymes:
            groups.setdefault(find(e), []).append(e)
        out[f"{threshold:.2f}"] = {
            "n_groups": len(groups),
            "groups": sorted(sorted(g) for g in groups.values()),
        }
    return out


def lineage_findings(rs: ReferenceSet, manifest: Mapping[str, Any], *,
                     threshold: float = LINEAGE_IDENTITY_THRESHOLD) -> list[Finding]:
    """Check the declared lineage groups against the pinned sequences.

    * a pair of enzymes at or above ``threshold`` that the loader keeps in
      different groups is an **error**: it would be counted as two independent
      samples of one lineage;
    * entries of one enzyme whose sequences are less than 90 % identical are an
      error too, because the workbook's own label ("WT", "G37D") would then be
      describing different proteins;
    * pairs within ten points below the threshold are noted, since a threshold
      is a choice and these are the pairs it separates.
    """
    findings: list[Finding] = []
    sequences = enzyme_sequences(rs, manifest)
    for enzyme, entries in sorted(sequences.items()):
        ids = sorted(entries)
        for other in ids[1:]:
            ident = _identity(entries[ids[0]], entries[other])
            if ident < 0.90:
                findings.append(Finding(
                    "error", "lineage.same_enzyme_differs", enzyme,
                    f"{ids[0]} and {other} are both labelled {enzyme} but are only "
                    f"{ident:.0%} identical"))
    for (a, b), ident in sorted(sequence_identities(rs, manifest).items()):
        same_group = LINEAGE_GROUPS[a] == LINEAGE_GROUPS[b]
        if ident >= threshold and not same_group:
            findings.append(Finding(
                "error", "lineage.merge_required", f"{a} / {b}",
                f"{ident:.0%} identical (coverage-adjusted), at or above "
                f"{threshold:.0%}, yet the loader keeps them in different "
                f"lineage groups ({LINEAGE_GROUPS[a]} / {LINEAGE_GROUPS[b]}); "
                f"they would be counted as two independent samples of one lineage"))
        elif threshold - 0.10 <= ident < threshold and not same_group:
            findings.append(Finding(
                "info", "lineage.near_threshold", f"{a} / {b}",
                f"{ident:.0%} identical, within ten points below the {threshold:.0%} "
                f"threshold that separates them"))
    return findings
