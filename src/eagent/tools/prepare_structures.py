"""Interface ``prepare_structures``: pick a structure, and say why that one.

Why this module is shaped the way it is
---------------------------------------

**Choosing a structure is a scientific decision, so it is recorded as one.**
The usual implementation picks whatever file is at hand -- an AlphaFold model,
because it is always available -- and the rest of the pipeline then measures
catalytic geometry in an apo, substrate-free, cofactor-free model and reports
it as structural evidence. The priority policy here is therefore explicit and
ordered, and every choice carries a reason string that is written into the QC
table:

1. a **matching experimental complex**: an experimental structure of this
   protein with the cofactor the catalytic template demands, in the state it
   demands (and, when the operator can name the substrate's ligand code, with
   the substrate or a close analogue too);
2. a **usable experimental structure of the enzyme**: same protein, but apo or
   missing part of the catalytic system;
3. a **reusable predicted structure** that is already on disk;
4. a **new prediction**, which costs money and GPU time and is therefore last;
5. an **experimental structure of a homologue**, ranked *below* a predicted
   model of the actual candidate. This ordering is deliberate and is the one
   people get backwards: a 2.0 A crystal structure of a 45 %-identical
   relative is a beautiful picture of a different protein, and its pocket
   residues are literally not the residues that will be mutated. A predicted
   model of the right sequence at least concerns the right molecule.

**The crystallographic NAD(P)+ trap.** A large fraction of deposited
ketoreductase structures carry the *oxidised* cofactor: NAD (NAD+) or NAP
(NADP+). The reduced forms are the separate chemical components NAI (NADH)
and NDP (NADPH). An oxidised nicotinamide has no hydride to donate; measuring
a "hydride transfer distance" from its C4 to a ketone carbon produces a
perfectly reasonable number that describes the reverse reaction's product
state at best, and nothing at all at worst. Whenever the catalytic template
demands a reduced cofactor and the file carries the oxidised form, this step
raises a **BLOCKER**. It does not quietly relabel the ligand.

**A high mean pLDDT hides a disordered active site.** Mean pLDDT is dominated
by the well-predicted core. The substrate-binding loops -- exactly the
residues a selectivity argument depends on -- are the ones that are
disordered, and a model at mean pLDDT 92 can have a pocket at 55. Both numbers
are therefore recorded, the pocket-local one is the one
:attr:`~eagent.schemas.candidate.StructureRecord.pocket_confidence` reads, and
PAE is carried through when a companion JSON is present because a confident
pocket in a domain that is confidently placed somewhere wrong is still wrong.

**Numbering is rebuilt, never assumed.** Author numbering in a PDB entry is
not the candidate's index, and the two differ by a tag, a construct boundary,
a cleaved propeptide or a numbering scheme. Every structure gets its own
:class:`~eagent.science.numbering.ResidueMap`, built by alignment, and
``residue_atom_mapping.tsv`` writes that correspondence out residue by residue
so a reviewer can check the one thing no downstream step can detect: that the
residue being measured is the residue that was meant.

**Nothing here invents a structure.** With ``ctx.policy.allow_network`` false
-- the default -- the only structures available are the ones in the local
index. A candidate with no entry produces a PARTIAL or FAILED result that
names the candidate, its sequence hash and the cache directory the file is
missing from. New predictions go through :class:`StructurePredictor`, which is
absent by default and raises :class:`~eagent.errors.ToolUnavailableError`; a
predictor that runs off this machine additionally requires network permission
*and* a named human authorisation covering that exact sequence hash, because
sending an unpublished construct to a web service discloses it irreversibly.

Calibration note
----------------
:data:`DEFAULT_POCKET_RADIUS_A`, :data:`DEFAULT_MIN_IDENTITY_SAME_PROTEIN` and
:data:`DEFAULT_MIN_IDENTITY_USABLE_SCAFFOLD` are **QC-localisation and
source-selection defaults, not catalytic criteria**. They decide which
residues get averaged for a confidence number and which structures are allowed
to represent a candidate; they never decide whether a candidate is active. No
catalytic window appears anywhere in this module -- those live in a sourced
:class:`~eagent.schemas.templates.CatalyticTemplate`. All three defaults need
per-family calibration and are overridable through :class:`StructurePolicy`.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..connectors.base import AccessPolicy, NetworkDisabledError
from ..context import RunContext
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..errors import EAgentError, ToolUnavailableError
from ..provenance import sha256_file, sha256_obj
from ..schemas import (
    Candidate,
    CatalyticTemplate,
    CofactorState,
    ConfidenceLevel,
    LigandSource,
    StructureRecord,
    cofactor_state_from_ligand_code,
)
from ..science.geometry import GeometryError, pocket_shell
from ..science.numbering import (
    NumberingError,
    ResidueMap,
    build_map,
    residue_one_letter,
)
from ..science.structure_io import (
    Atom,
    Chain,
    Residue,
    Structure,
    StructureParseError,
    read_structure,
)
from .base import ScientificInterface
from .handoff import CANDIDATES_KEY, as_candidates, serialise_candidates

__all__ = [
    "DEFAULT_POCKET_RADIUS_A",
    "PRIORITY_MATCHING_COMPLEX",
    "PRIORITY_EXPERIMENTAL_ENZYME",
    "PRIORITY_REUSABLE_PREDICTION",
    "PRIORITY_NEW_PREDICTION",
    "PRIORITY_HOMOLOGUE_SCAFFOLD",
    "PRIORITY_REJECTED",
    "expected_chain_count",
    "DEFAULT_MIN_IDENTITY_SAME_PROTEIN",
    "DEFAULT_MIN_IDENTITY_USABLE_SCAFFOLD",
    "METAL_ELEMENTS",
    "StructureSourceError",
    "StructureSource",
    "StructureIndexEntry",
    "StructureIndex",
    "StructurePolicy",
    "ChainChoice",
    "CofactorAssessment",
    "AssemblyAssessment",
    "PocketConfidence",
    "StructureAssessment",
    "PredictionRequest",
    "PredictionOutcome",
    "StructurePredictor",
    "UnavailablePredictor",
    "select_chain",
    "ligand_inventory",
    "metals_in",
    "assess_cofactor",
    "assess_assembly",
    "pocket_residues",
    "pocket_confidence",
    "read_pae_matrix",
    "mutation_tokens",
    "classify_source",
    "guard_sequence_submission",
    "write_mmcif_text",
    "PrepareStructures",
]


# ---------------------------------------------------------------------------
# calibration defaults -- see the module docstring
# ---------------------------------------------------------------------------

#: Radius, in angstrom, of the shell used to localise "the pocket" for a
#: confidence average. A QC-localisation default: it decides which residues are
#: averaged, never whether a geometry is catalytic. Needs per-family
#: calibration -- a deep AKR barrel and a shallow SDR cleft do not have pockets
#: of the same size.
DEFAULT_POCKET_RADIUS_A: float = 6.0

#: Sequence identity at or above which a structure is treated as a structure
#: *of this protein* rather than of a relative. A source-selection default
#: needing per-family calibration: in a family where a 3 % difference is a
#: different substrate specificity this is too permissive, and for a distant
#: but structurally rigid fold it is too strict.
DEFAULT_MIN_IDENTITY_SAME_PROTEIN: float = 0.95

#: Identity below which a structure is not used at all. Also a selection
#: default, not a catalytic one: below it, pocket residues are not comparable
#: position by position, so a mutation proposal derived from the structure
#: would be a proposal about the other protein.
DEFAULT_MIN_IDENTITY_USABLE_SCAFFOLD: float = 0.30

#: Elements counted as metals when a single-atom HETATM group is inventoried.
#: Chemistry, not a tunable threshold. The catalytic zinc of an MDR/ADH is a
#: mechanistic component, so it must never be filtered out as "not a ligand".
METAL_ELEMENTS: frozenset[str] = frozenset({
    "ZN", "MG", "MN", "FE", "CU", "NI", "CO", "CA", "NA", "K", "CD", "MO",
    "W", "V", "CR", "HG", "PB", "SR", "BA", "AG", "AU", "PT", "PD",
})

#: Priority ranks assigned by :func:`classify_source`. Lower is better, and the
#: numbers are stored in ``StructureRecord.priority_rank`` so a reviewer can
#: sort the QC table by the policy that produced it.
PRIORITY_MATCHING_COMPLEX: int = 1
PRIORITY_EXPERIMENTAL_ENZYME: int = 2
PRIORITY_REUSABLE_PREDICTION: int = 3
#: The "run a new prediction" tier. It orders *attempts* -- a prediction is
#: tried only after the cache has been exhausted, because it costs GPU time --
#: and the model it produces is then ranked as what it is, a predicted
#: structure (:data:`PRIORITY_REUSABLE_PREDICTION`), not better for having been
#: made in this run.
PRIORITY_NEW_PREDICTION: int = 4
PRIORITY_HOMOLOGUE_SCAFFOLD: int = 5
PRIORITY_REJECTED: int = 99


class StructureSourceError(EAgentError):
    """A structure named by the index cannot be read or does not exist.

    Raised (and caught per candidate) rather than returning a placeholder
    structure, because every downstream measurement would otherwise be made on
    coordinates nobody supplied.
    """


class StructureSource(str, enum.Enum):
    """Where a structure came from. The values match
    :attr:`~eagent.schemas.candidate.StructureRecord.source`.

    The distinction that matters is not "good vs bad" but *what the B-factor
    column means*: in an experimental entry it is a refinement B-factor (low is
    good, A^2), in a predicted model it is pLDDT (high is good, 0-100). Reading
    one as the other inverts the confidence of every pocket in the run, so the
    source enum is what gates that read.
    """

    PDB_COMPLEX = "pdb_complex"
    PDB_APO = "pdb_apo"
    AFDB = "afdb"
    PREDICTED = "predicted"
    HOMOLOGY = "homology"

    @property
    def is_experimental(self) -> bool:
        """Whether the coordinates were measured rather than computed."""
        return self in (StructureSource.PDB_COMPLEX, StructureSource.PDB_APO)

    @property
    def carries_plddt(self) -> bool:
        """Whether the B-factor column holds pLDDT and may be read as
        confidence. False for experimental entries and for homology models,
        whose B-factor column means whatever the modelling program put there."""
        return self in (StructureSource.AFDB, StructureSource.PREDICTED)


# ---------------------------------------------------------------------------
# the local structure index
# ---------------------------------------------------------------------------


class StructureIndexEntry(BaseModel):
    """One structure available on this machine, with its retrieval provenance.

    The index records only what cannot be measured from the coordinates: where
    the file came from, which database and snapshot, which chain the curator
    meant, and whether a PAE file accompanies it. Everything that *can* be
    measured -- ligands, assembly, identity, missing regions, confidence -- is
    measured from the file itself by :class:`PrepareStructures`, because an
    index that asserts "holo" is a claim nobody re-checks.
    """

    model_config = ConfigDict(extra="forbid")

    structure_id: str
    path: str = Field(..., description="Relative to the index file's directory, "
                                       "or absolute.")
    source: StructureSource
    candidate_id: str | None = None
    sequence_sha256: str | None = Field(
        None, description="``sequence_hash`` of the protein this file is of; the "
                          "join key that survives accession re-annotation.")
    chain: str | None = Field(
        None, description="Author chain the curator intends. When null the chain "
                          "is chosen by alignment and the choice is reported.")
    database: str | None = None
    database_version: str | None = Field(
        None, description="Snapshot string of the database; null when unknown, "
                          "never invented.")
    assembly_in_file: str | None = Field(
        None, description="What the depositor or curator says this file is "
                          "(e.g. 'author biological assembly 1, homotetramer'). "
                          "A statement about the file, re-checked against the "
                          "chain count, not trusted on its own.")
    pae_json: str | None = Field(
        None, description="Companion predicted-aligned-error JSON, when the "
                          "predictor emitted one.")
    evidence: str = Field("", description="Why this file is in the cache: entry "
                                          "id, resolution, publication.")

    @model_validator(mode="after")
    def _must_name_its_protein(self) -> "StructureIndexEntry":
        if not self.candidate_id and not self.sequence_sha256:
            raise ValueError(
                f"structure index entry '{self.structure_id}' names neither a "
                f"candidate_id nor a sequence_sha256, so nothing can say which "
                f"protein it is a structure of"
            )
        return self


@dataclass
class StructureIndex:
    """The set of structures this machine already holds.

    Deliberately a plain local index rather than a live lookup: offline is the
    default, and a cache miss is a fact about this machine that must be
    reported as "I need file X", never resolved by inventing coordinates.
    """

    entries: list[StructureIndexEntry] = field(default_factory=list)
    root: Path = field(default_factory=Path)
    cache_version: str | None = None
    source_path: Path | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "StructureIndex":
        """Load ``{"cache_version": ..., "entries": [...]}`` from disk.

        A malformed entry raises rather than being skipped: a silently dropped
        entry is a structure that exists on disk and that the run then reports
        as missing, which sends an operator looking for a file they already
        have.
        """
        p = Path(path)
        if not p.is_file():
            raise StructureSourceError(f"structure index not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or "entries" not in raw:
            raise StructureSourceError(
                f"{p}: expected an object with an 'entries' list"
            )
        entries = [StructureIndexEntry(**e) for e in raw["entries"]]
        return cls(entries=entries, root=p.parent,
                   cache_version=raw.get("cache_version"), source_path=p)

    def for_candidate(self, candidate_id: str,
                      sequence_sha256: str | None = None) -> list[StructureIndexEntry]:
        """Entries claiming to be structures of this candidate.

        Matching on either the candidate id or the sequence hash, because the
        two identify a protein in different ways and a cache curated before the
        candidate ids were assigned only has the hash.
        """
        out: list[StructureIndexEntry] = []
        for e in self.entries:
            if e.candidate_id and e.candidate_id == candidate_id:
                out.append(e)
            elif sequence_sha256 and e.sequence_sha256 == sequence_sha256:
                out.append(e)
        return out

    def resolve(self, entry: StructureIndexEntry) -> Path:
        """Absolute path of an entry's coordinate file, or raise."""
        p = Path(entry.path)
        if not p.is_absolute():
            p = self.root / p
        if not p.is_file():
            raise StructureSourceError(
                f"structure '{entry.structure_id}' is indexed at {p} but no "
                f"such file exists"
            )
        return p

    def resolve_pae(self, entry: StructureIndexEntry) -> Path | None:
        """Absolute path of the companion PAE JSON, or ``None`` when absent."""
        if not entry.pae_json:
            return None
        p = Path(entry.pae_json)
        if not p.is_absolute():
            p = self.root / p
        return p if p.is_file() else None


class StructurePolicy(BaseModel):
    """Tunable selection and QC-localisation parameters, in one visible place.

    Kept as a model rather than as keyword arguments so the exact values used
    land in ``Provenance.parameters`` and a reviewer can see what the run
    considered "the same protein".
    """

    model_config = ConfigDict(extra="forbid")

    pocket_radius_angstrom: float = Field(DEFAULT_POCKET_RADIUS_A, gt=0.0)
    min_identity_same_protein: float = Field(DEFAULT_MIN_IDENTITY_SAME_PROTEIN,
                                             ge=0.0, le=1.0)
    min_identity_usable_scaffold: float = Field(
        DEFAULT_MIN_IDENTITY_USABLE_SCAFFOLD, ge=0.0, le=1.0)
    substrate_ligand_codes: tuple[str, ...] = Field(
        (), description="PDB chemical component ids that would count as the "
                        "substrate or a close analogue in an experimental "
                        "complex. Empty means the substrate match cannot be "
                        "assessed -- which is reported, not assumed away.")
    allow_new_prediction: bool = True
    predict_when_experimental_exists: bool = Field(
        False, description="Predict even when an experimental structure of the "
                           "candidate is already available. Off by default: a "
                           "prediction costs GPU time and ranks below an "
                           "experimental structure of the same protein.")


# ---------------------------------------------------------------------------
# measured assessments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainChoice:
    """Which chain was used for a candidate, and why that one.

    The runner-up is kept because "chain A, 99 % identity" and "chain A, 99 %
    identity, and chain B was 98 %" describe different situations: the second
    is a homodimer whose second copy may complete the pocket.
    """

    chain_id: str
    identity: float | None
    coverage: float
    reason: str
    alternatives: tuple[tuple[str, float | None], ...] = ()


@dataclass(frozen=True)
class CofactorAssessment:
    """What cofactor the file actually contains, in what oxidation state."""

    found_code: str | None
    found_state: CofactorState
    required_cofactor: str | None
    required_state: CofactorState
    verdict: str
    is_blocker: bool
    message: str


@dataclass(frozen=True)
class AssemblyAssessment:
    """Whether the file's chain content can support the family's mechanism."""

    n_polymer_chains: int
    chain_ids: tuple[str, ...]
    template_assembly_state: str | None
    expected_chain_count: int | None
    verdict: str
    message: str


@dataclass(frozen=True)
class PocketConfidence:
    """Global and pocket-local confidence, kept apart on purpose."""

    mean_plddt: float | None
    pocket_plddt: float | None
    n_pocket_residues: int
    pocket_definition: str
    pae_mean: float | None = None
    pae_pocket_mean: float | None = None
    notes: tuple[str, ...] = ()


@dataclass
class StructureAssessment:
    """Everything measured about one candidate/structure pair.

    Carries the live objects (``structure``, ``residue_map``) as well as the
    serialisable :class:`~eagent.schemas.candidate.StructureRecord`, so the
    caller can measure further without re-reading and re-aligning, while the
    envelope still only ever carries JSON.
    """

    candidate_id: str
    entry: StructureIndexEntry
    structure: Structure
    chain_choice: ChainChoice
    residue_map: ResidueMap
    record: StructureRecord
    cofactor: CofactorAssessment
    assembly: AssemblyAssessment
    confidence: PocketConfidence
    pocket_residue_keys: tuple[tuple[str, int, str, str], ...] = ()
    #: Catalytic role label -> 0-based candidate index, carried from the
    #: candidate's :class:`~eagent.schemas.candidate.CatalyticMapping` so the
    #: mapping table can say which residue plays which role without the
    #: assessment holding a reference to the whole candidate.
    catalytic_roles: dict[str, int] = field(default_factory=dict)
    selection_reason: str = ""
    selected: bool = False
    flags: list[tuple[str, Severity, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# chain selection and sequence agreement
# ---------------------------------------------------------------------------


def select_chain(structure: Structure, candidate_sequence: str,
                 requested_chain: str | None = None) -> tuple[Chain, ResidueMap, ChainChoice]:
    """Choose the chain this candidate corresponds to, and build its map.

    When the index names a chain, that chain is used and its agreement with the
    candidate is still measured -- a curator naming the wrong chain is exactly
    the error this reports. When no chain is named, every polymer chain is
    aligned and the best agreement wins, ranked on identity and then coverage
    rather than on identity alone: a six-residue peptide chain can be 100 %
    identical to a fragment of the candidate and is not the enzyme.

    Raises :class:`StructureSourceError` when the requested chain is absent or
    when no chain carries polymer residues, because silently falling back to
    "the first chain" is how a run measures the wrong subunit.
    """
    if requested_chain is not None:
        chain = structure.chain(requested_chain)
        if chain is None:
            raise StructureSourceError(
                f"{structure.structure_id}: no chain '{requested_chain}' "
                f"(present: {[c.chain_id for c in structure.chains]})"
            )
        rmap = build_map(candidate_sequence, chain)
        return chain, rmap, ChainChoice(
            chain_id=chain.chain_id,
            identity=rmap.structure_identity,
            coverage=rmap.coverage,
            reason=(f"chain named by the structure index; measured identity "
                    f"{_pct(rmap.structure_identity)} at coverage "
                    f"{rmap.coverage:.0%}"),
        )

    scored: list[tuple[float, float, Chain, ResidueMap]] = []
    for chain in structure.chains:
        if not chain.polymer_residues():
            continue
        try:
            rmap = build_map(candidate_sequence, chain)
        except NumberingError:
            continue
        scored.append((rmap.structure_identity or 0.0, rmap.coverage, chain, rmap))
    if not scored:
        raise StructureSourceError(
            f"{structure.structure_id}: no chain could be aligned to the "
            f"candidate; the file may contain no polymer residues at all"
        )
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    identity, coverage, chain, rmap = scored[0]
    alternatives = tuple((c.chain_id, i) for i, _cov, c, _m in scored[1:])
    return chain, rmap, ChainChoice(
        chain_id=chain.chain_id,
        identity=rmap.structure_identity,
        coverage=coverage,
        reason=(f"best alignment of {len(scored)} polymer chain(s): identity "
                f"{_pct(rmap.structure_identity)}, coverage {coverage:.0%}"),
        alternatives=alternatives,
    )


def mutation_tokens(rmap: ResidueMap) -> list[str]:
    """Differences between the candidate and the structure, as ``Y155F`` tokens.

    Read in the direction that matters downstream: the candidate has the first
    letter, the *structure* has the last. A structure carrying ``Y155F`` is a
    structure of the inactive mutant, and measuring the catalytic tyrosine's
    geometry in it measures a phenylalanine.
    """
    out: list[str] = []
    for _index, cand_letter, struct_letter, pos in rmap.mismatches:
        out.append(f"{cand_letter}{pos.token}{struct_letter}")
    return out


def _constant_offset(rmap: ResidueMap) -> tuple[int, str | None]:
    """Author number minus 1-based candidate position, when it is constant.

    Returns ``(0, note)`` when it is not. A non-constant offset is normal (an
    unobserved loop shifts nothing, an engineered insertion shifts everything
    after it) and is precisely why ``ResidueMap`` exists; recording a single
    offset in that case would invite someone to add it to an index.
    """
    offsets = {pos.resseq - (i + 1) for i, pos in rmap.index_to_author.items()}
    if len(offsets) == 1:
        return next(iter(offsets)), None
    return 0, (f"numbering offset is not constant ({len(offsets)} distinct "
               f"values); use the ResidueMap, never an offset")


# ---------------------------------------------------------------------------
# ligands, cofactor state, assembly
# ---------------------------------------------------------------------------


def ligand_inventory(structure: Structure) -> dict[str, int]:
    """Non-water HETATM component ids and how many copies of each.

    Includes metal ions. A ketoreductase run that filters ions out loses the
    catalytic zinc of every MDR/ADH candidate.
    """
    counts: dict[str, int] = {}
    for res in structure.ligands():
        code = res.resname.strip().upper()
        counts[code] = counts.get(code, 0) + 1
    return counts


def metals_in(structure: Structure) -> list[str]:
    """Metal element symbols present as single-atom HETATM groups."""
    found: set[str] = set()
    for res in structure.ligands():
        if len(res.atoms) != 1:
            continue
        el = res.atoms[0].element.strip().upper()
        if el in METAL_ELEMENTS:
            found.add(el)
    return sorted(found)


def assess_cofactor(structure: Structure,
                    template: CatalyticTemplate | None) -> CofactorAssessment:
    """Compare the cofactor in the file against the one the mechanism needs.

    THE TRAP: a deposited ternary complex very often carries NAD (NAD+) or NAP
    (NADP+) -- the *oxidised* cofactor, with no hydride to give. The reduced
    components are NAI (NADH) and NDP (NADPH). Treating the oxidised form as a
    hydride donor yields a complete, plausible, meaningless hydride-transfer
    geometry. When the template demands a reduced cofactor and the file carries
    the oxidised one, this returns ``is_blocker=True``.

    An unrecognised component id yields :attr:`CofactorState.UNKNOWN` and a
    non-blocking verdict that says the state must be resolved from the source
    entry -- never a guess, in either direction.
    """
    required = template.required_cofactor if template else None
    required_state = template.required_cofactor_state if template \
        else CofactorState.UNKNOWN
    acceptable = {c.strip().upper() for c in (template.cofactor_ligand_codes
                                              if template else [])}
    inventory = ligand_inventory(structure)

    candidates: list[tuple[str, CofactorState]] = []
    for code in inventory:
        state = cofactor_state_from_ligand_code(code)
        if state is not CofactorState.UNKNOWN or code in acceptable:
            candidates.append((code, state))

    if required is None:
        return CofactorAssessment(
            found_code=candidates[0][0] if candidates else None,
            found_state=candidates[0][1] if candidates else CofactorState.NOT_APPLICABLE,
            required_cofactor=None, required_state=required_state,
            verdict="no_cofactor_required",
            is_blocker=False,
            message=("the catalytic template declares no cofactor requirement"
                     if template else
                     "no catalytic template was resolved for this candidate, so "
                     "the cofactor requirement is unknown and was not checked"),
        )

    if not candidates:
        return CofactorAssessment(
            found_code=None, found_state=CofactorState.UNKNOWN,
            required_cofactor=required, required_state=required_state,
            verdict="cofactor_absent", is_blocker=False,
            message=(f"the template requires {required} "
                     f"[{required_state.value}] and this file contains no "
                     f"nicotinamide-like component (ligands: "
                     f"{sorted(inventory) or 'none'}); it is an apo structure "
                     f"and the cofactor must be modelled in before any "
                     f"hydride-transfer geometry is measured"),
        )

    # Prefer a component whose state is actually determinable.
    candidates.sort(key=lambda t: t[1] is CofactorState.UNKNOWN)
    code, state = candidates[0]

    if required_state is CofactorState.REDUCED and state is CofactorState.OXIDIZED:
        return CofactorAssessment(
            found_code=code, found_state=state,
            required_cofactor=required, required_state=required_state,
            verdict="oxidised_cofactor_where_reduced_is_required",
            is_blocker=True,
            message=(f"the template requires {required} in the reduced state, "
                     f"but this structure carries component {code}, which is "
                     f"the OXIDISED cofactor. An oxidised nicotinamide has no "
                     f"hydride to transfer; its C4 may not be used as a hydride "
                     f"donor. Use an entry with NAI/NDP, or model the reduced "
                     f"cofactor in explicitly and record it as transplanted"),
        )
    if state is CofactorState.UNKNOWN:
        return CofactorAssessment(
            found_code=code, found_state=state,
            required_cofactor=required, required_state=required_state,
            verdict="cofactor_state_unknown", is_blocker=False,
            message=(f"component {code} is present but its oxidation state is "
                     f"not derivable from the component id; resolve it from the "
                     f"source entry before treating it as a hydride donor"),
        )
    if required_state is not CofactorState.UNKNOWN and state is not required_state:
        return CofactorAssessment(
            found_code=code, found_state=state,
            required_cofactor=required, required_state=required_state,
            verdict="cofactor_state_mismatch", is_blocker=True,
            message=(f"the template requires {required} [{required_state.value}] "
                     f"and the structure carries {code} [{state.value}]"),
        )
    return CofactorAssessment(
        found_code=code, found_state=state,
        required_cofactor=required, required_state=required_state,
        verdict="cofactor_state_matches", is_blocker=False,
        message=f"{code} is present in the required {state.value} state",
    )


#: Assembly words a template may use, and the chain count each implies. Unknown
#: words return ``None`` rather than a guess: "dimer of dimers" is not 2.
_ASSEMBLY_CHAIN_COUNT: dict[str, int] = {
    "monomer": 1, "monomeric": 1,
    "dimer": 2, "homodimer": 2, "heterodimer": 2, "dimeric": 2,
    "trimer": 3, "homotrimer": 3, "trimeric": 3,
    "tetramer": 4, "homotetramer": 4, "heterotetramer": 4, "tetrameric": 4,
    "pentamer": 5, "hexamer": 6, "homohexamer": 6, "hexameric": 6,
    "octamer": 8, "homooctamer": 8, "octameric": 8,
}


def expected_chain_count(assembly_state: str | None) -> int | None:
    """Chain count implied by a template's assembly word, or ``None``.

    Only an unambiguous single word is accepted, after parenthetical asides are
    dropped: ``"homotetramer"`` and ``"homotetramer (biological assembly 1)"``
    both give 4. A phrase that is not one word -- ``"dimer of dimers"``,
    ``"tetramer in solution, dimer in the crystal"`` -- gives ``None``, because
    a substring scan would read the first number it recognised and turn an
    unparsed phrase into a silent assembly verdict. ``None`` is reported as an
    uncertainty, which is the honest outcome.
    """
    if not assembly_state:
        return None
    text = assembly_state.strip().lower()
    while "(" in text and ")" in text:
        head, _, rest = text.partition("(")
        _inside, _, tail = rest.partition(")")
        text = f"{head} {tail}".strip()
    tokens = [t.strip(".,;:") for t in text.replace("-", " ").split()
              if t.strip(".,;:")]
    if len(tokens) == 1:
        return _ASSEMBLY_CHAIN_COUNT.get(tokens[0])
    return None


def assess_assembly(structure: Structure,
                    template: CatalyticTemplate | None) -> AssemblyAssessment:
    """Does this file contain enough of the oligomer for the mechanism?

    THE TRAP: the asymmetric unit of a crystal is not the biological assembly.
    In most SDRs the substrate pocket is walled by the neighbouring subunit's
    loop, and in many AKRs and MDRs the cofactor site is completed across the
    interface. Docking into a lone monomer extracted from a tetrameric enzyme
    gives a pocket that is open to solvent on the side that does the
    selecting, and every pose in it is an artefact of the missing subunit.

    A deficit is reported, never repaired: regenerating the assembly needs the
    symmetry operators from the entry, which this reader does not parse.
    """
    chain_ids = tuple(c.chain_id for c in structure.chains if c.polymer_residues())
    n = len(chain_ids)
    want = expected_chain_count(template.assembly_state if template else None)
    state = template.assembly_state if template else None

    if template is None:
        return AssemblyAssessment(n, chain_ids, None, None, "unchecked",
                                  "no catalytic template resolved; the required "
                                  "assembly state is unknown and was not checked")
    if state is None:
        return AssemblyAssessment(n, chain_ids, None, None, "template_silent",
                                  f"the catalytic template declares no assembly "
                                  f"state; this file has {n} polymer chain(s) "
                                  f"and whether that is enough is unresolved")
    if want is None:
        return AssemblyAssessment(n, chain_ids, state, None, "assembly_unparsed",
                                  f"the template's assembly state '{state}' was "
                                  f"not understood as a chain count; this file "
                                  f"has {n} polymer chain(s), compare by hand")
    if n < want:
        return AssemblyAssessment(
            n, chain_ids, state, want, "subunit_deficient",
            f"the mechanism needs a {state} ({want} chains) and this file has "
            f"{n}; the pocket may be incomplete because the neighbouring "
            f"subunit is absent. Regenerate the biological assembly before "
            f"docking into it")
    if n > want:
        return AssemblyAssessment(
            n, chain_ids, state, want, "extra_chains",
            f"this file has {n} polymer chains where the template expects "
            f"{want} ({state}); crystal packing copies are present and the "
            f"chain used for measurement must be stated")
    return AssemblyAssessment(n, chain_ids, state, want, "assembly_matches",
                              f"{n} polymer chain(s), matching the template's "
                              f"{state}")


# ---------------------------------------------------------------------------
# pocket-local confidence
# ---------------------------------------------------------------------------


def pocket_residues(structure: Structure, seed_atoms: Sequence[Atom],
                    radius_angstrom: float) -> list[Residue]:
    """Protein residues within ``radius_angstrom`` of the seed atoms.

    A SEARCH SCOPE for averaging a confidence number, nothing more: membership
    says only "this residue has a heavy atom near the seed in this one model".
    The seeds themselves are included when they are protein residues, because
    the confidence of the catalytic residues is the first thing a reader wants.
    """
    seeds = [a for a in seed_atoms if a.is_heavy]
    if not seeds:
        return []
    shell = pocket_shell(structure, seeds, 0.0, radius_angstrom)
    seed_keys = {a.residue_key for a in seeds}
    by_key: dict[tuple[str, int, str, str], Residue] = {r.key: r for r in shell}
    for res in structure.residues():
        if res.key in seed_keys and not res.is_hetatm:
            by_key.setdefault(res.key, res)
    return [by_key[k] for k in sorted(by_key)]


def _residue_bvalue(res: Residue) -> float | None:
    """Mean B-factor/pLDDT over a residue's heavy atoms, or ``None``.

    ``None`` when the column was absent: a missing confidence value is not a
    low one, and averaging over whatever happens to be present would quietly
    change the denominator.
    """
    vals = [a.bfactor_or_plddt for a in res.heavy_atoms()
            if a.bfactor_or_plddt is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def pocket_confidence(structure: Structure, chain: Chain,
                      pocket: Sequence[Residue], source: StructureSource,
                      pocket_definition: str,
                      pae: Sequence[Sequence[float]] | None = None,
                      pae_indices: Sequence[int] | None = None) -> PocketConfidence:
    """Mean and pocket-local pLDDT, plus PAE summaries when available.

    Both numbers are reported because they answer different questions and
    disagree exactly when it matters: a model can be confidently folded
    (mean 92) and have an active site nobody should measure (pocket 55). Only
    the pocket number is a statement about the residues a catalytic geometry
    depends on.

    For an experimental entry, the B-factor column is *not* pLDDT and both
    values stay ``None`` with a note. Returning a number there would put a
    refinement B-factor on a 0-100 confidence axis, where 20 would read as
    catastrophic rather than excellent.
    """
    notes: list[str] = []
    if not source.carries_plddt:
        notes.append(
            f"source '{source.value}' is not a predicted model: its B-factor "
            f"column is not pLDDT, so no confidence value was derived from it"
        )
        mean_plddt = pocket_plddt = None
    else:
        per_residue = [v for v in (_residue_bvalue(r)
                                   for r in chain.polymer_residues())
                       if v is not None]
        mean_plddt = sum(per_residue) / len(per_residue) if per_residue else None
        if mean_plddt is None:
            notes.append("the predicted model carries no B-factor/pLDDT column")
        pocket_vals = [v for v in (_residue_bvalue(r) for r in pocket
                                   if not r.is_hetatm) if v is not None]
        pocket_plddt = sum(pocket_vals) / len(pocket_vals) if pocket_vals else None
        if pocket_plddt is None and pocket:
            notes.append("no pLDDT values on the pocket residues")

    pae_mean = pae_pocket = None
    if pae:
        n = len(pae)
        total = sum(sum(row) for row in pae)
        pae_mean = total / (n * n) if n else None
        if pae_indices:
            idx = [i for i in pae_indices if 0 <= i < n]
            if idx:
                pae_pocket = sum(pae[i][j] for i in idx for j in idx) / (len(idx) ** 2)
            else:
                notes.append("pocket indices fell outside the PAE matrix; no "
                             "pocket-local PAE was computed")
        else:
            notes.append("PAE was summarised globally only: the residue axis of "
                         "the matrix could not be tied to candidate positions")

    return PocketConfidence(
        mean_plddt=mean_plddt, pocket_plddt=pocket_plddt,
        n_pocket_residues=len([r for r in pocket if not r.is_hetatm]),
        pocket_definition=pocket_definition,
        pae_mean=pae_mean, pae_pocket_mean=pae_pocket, notes=tuple(notes),
    )


def read_pae_matrix(path: str | Path) -> list[list[float]]:
    """Read a predicted-aligned-error matrix from a companion JSON.

    Accepts the three layouts the common predictors emit (a bare object with
    ``pae`` or ``predicted_aligned_error``, or a one-element list holding such
    an object). Anything else raises: a PAE file whose layout is not recognised
    is not summarised as "no PAE available", because the two mean different
    things to a reader deciding whether a pocket is reliably placed.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    obj: Any = raw[0] if isinstance(raw, list) and raw else raw
    matrix: Any = None
    if isinstance(obj, Mapping):
        for key in ("pae", "predicted_aligned_error", "pae_matrix"):
            if key in obj:
                matrix = obj[key]
                break
    if not isinstance(matrix, list) or not matrix:
        raise StructureSourceError(
            f"{path}: no 'pae' / 'predicted_aligned_error' matrix found; the "
            f"file layout is not recognised and was not guessed at"
        )
    out: list[list[float]] = []
    n = len(matrix)
    for row in matrix:
        if not isinstance(row, list) or len(row) != n:
            raise StructureSourceError(
                f"{path}: the PAE matrix is not square ({n} rows, a row of "
                f"length {len(row) if isinstance(row, list) else 'scalar'})"
            )
        out.append([float(v) for v in row])
    return out


# ---------------------------------------------------------------------------
# source priority policy
# ---------------------------------------------------------------------------


def classify_source(entry: StructureIndexEntry, identity: float | None,
                    cofactor: CofactorAssessment, inventory: Mapping[str, int],
                    policy: StructurePolicy) -> tuple[int, str]:
    """Priority rank and the reason for it. The ordering is the module's policy.

    The reason string is returned rather than logged because it is the only
    defence against the silent version of this step: a run that picked a
    predicted apo model while a liganded crystal structure sat in the cache,
    and recorded nothing about why.
    """
    ident = identity if identity is not None else 0.0
    substrate_codes = {c.strip().upper() for c in policy.substrate_ligand_codes}
    has_substrate = bool(substrate_codes & {c.upper() for c in inventory})

    if entry.source.is_experimental:
        if ident < policy.min_identity_usable_scaffold:
            return PRIORITY_REJECTED, (
                f"experimental structure at {_pct(identity)} identity, below the "
                f"{policy.min_identity_usable_scaffold:.0%} floor: its pocket "
                f"residues are not positionally comparable to the candidate's")
        if ident < policy.min_identity_same_protein:
            return PRIORITY_HOMOLOGUE_SCAFFOLD, (
                f"experimental structure of a homologue ({_pct(identity)} "
                f"identity, below the {policy.min_identity_same_protein:.0%} "
                f"same-protein threshold): ranked below a predicted model of "
                f"the candidate itself, because it is a measured picture of a "
                f"different protein")
        cofactor_ok = cofactor.verdict in ("cofactor_state_matches",
                                           "no_cofactor_required")
        if cofactor_ok and (has_substrate or not substrate_codes):
            detail = "with the substrate/analogue bound" if has_substrate else (
                "no substrate ligand codes were supplied, so substrate "
                "occupancy could not be assessed")
            return PRIORITY_MATCHING_COMPLEX, (
                f"experimental structure of this protein ({_pct(identity)} "
                f"identity) carrying the required cofactor "
                f"({cofactor.found_code or 'n/a'}); {detail}")
        return PRIORITY_EXPERIMENTAL_ENZYME, (
            f"experimental structure of this protein ({_pct(identity)} "
            f"identity) but the catalytic system is incomplete: "
            f"{cofactor.verdict}")

    if ident < policy.min_identity_usable_scaffold:
        return PRIORITY_REJECTED, (
            f"model at {_pct(identity)} identity to the candidate, below the "
            f"{policy.min_identity_usable_scaffold:.0%} floor: an indexed model "
            f"that does not match the candidate sequence is a model of another "
            f"protein, usually a stale accession in the cache")

    if entry.source.carries_plddt:
        if ident < policy.min_identity_same_protein:
            return PRIORITY_HOMOLOGUE_SCAFFOLD, (
                f"predicted model of a related sequence ({_pct(identity)} "
                f"identity, below the {policy.min_identity_same_protein:.0%} "
                f"same-protein threshold); its pocket is not the candidate's "
                f"pocket residue for residue")
        return PRIORITY_REUSABLE_PREDICTION, (
            f"predicted model already on disk ({entry.source.value}, "
            f"{_pct(identity)} identity to the candidate); reused rather than "
            f"re-predicted")
    return PRIORITY_HOMOLOGUE_SCAFFOLD, (
        f"homology model ({_pct(identity)} identity); its B-factor column is "
        f"not a calibrated confidence and its pocket is inherited from the "
        f"template structure")


def _pct(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.1%}"


# ---------------------------------------------------------------------------
# the prediction seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRequest:
    """What a predictor is asked for. Carries the seed so a rerun is a rerun."""

    candidate_id: str
    sequence: str
    out_dir: Path
    seed: int


@dataclass(frozen=True)
class PredictionOutcome:
    """What a predictor returned.

    Note what is *not* here: a confidence number. pLDDT is read back out of the
    file's B-factor column by this module, so the recorded confidence always
    describes the coordinates that were actually stored rather than a summary
    the adapter reported.
    """

    path: Path
    model_name: str
    model_version: str
    pae_json: Path | None = None
    notes: str = ""


@runtime_checkable
class StructurePredictor(Protocol):
    """Adapter seam for single-chain structure prediction.

    An object, not a function, so three facts travel with the capability:
    whether it is installed, what version produced the coordinates, and
    whether running it sends the sequence off this machine. The last one is
    enforced in code by :func:`guard_sequence_submission`.
    """

    name: str
    version: str
    runs_remotely: bool

    def is_available(self) -> bool: ...

    def predict(self, request: PredictionRequest) -> PredictionOutcome: ...


@dataclass(frozen=True)
class UnavailablePredictor:
    """The default: no predictor is installed, and that is reported loudly.

    The alternative -- a built-in fallback that threads a sequence onto a
    homologue and calls it a prediction -- would produce a file that looks like
    every other structure in the run while being an unrecorded homology model.
    """

    name: str = "structure_predictor"
    version: str = "absent"
    runs_remotely: bool = False
    install_hint: str = ("install a local predictor (e.g. ColabFold/Boltz) and "
                         "pass it as predictor=, or add the model to the "
                         "structure index")

    def is_available(self) -> bool:
        return False

    def predict(self, request: PredictionRequest) -> PredictionOutcome:
        raise ToolUnavailableError(self.name, self.install_hint)


def guard_sequence_submission(predictor: StructurePredictor, sequence: str,
                              ctx: RunContext,
                              access: AccessPolicy | None = None) -> list[str]:
    """Refuse to send an unpublished sequence to a service that is off-machine.

    Two independent permissions are required, and both are checked here rather
    than trusted to the adapter:

    * ``ctx.policy.allow_network`` must be true, otherwise the run was
      configured to be reproducible from local inputs and a remote call would
      break that silently;
    * the sequence must be covered by a named
      :class:`~eagent.connectors.base.SubmissionAuthorization`, or be recorded
      as already public.

    Disclosure is irreversible -- no later policy change un-sends a construct
    -- so the default (an :class:`AccessPolicy` that authorises nothing) is a
    refusal. A predictor that runs locally skips both checks and the fact that
    it ran locally is returned as a note.
    """
    if not getattr(predictor, "runs_remotely", False):
        return [f"{predictor.name} ran locally; no sequence left this machine"]
    if not ctx.policy.allow_network:
        raise NetworkDisabledError(
            f"{predictor.name} runs off this machine and "
            f"ctx.policy.allow_network is False; no sequence was sent"
        )
    policy = access or AccessPolicy.from_execution_policy(ctx.policy)
    return policy.check_outbound(predictor.name, {"sequence": sequence})


# ---------------------------------------------------------------------------
# mmCIF output
# ---------------------------------------------------------------------------

_CIF_COLUMNS: tuple[str, ...] = (
    "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
    "label_comp_id", "label_asym_id", "label_seq_id", "pdbx_PDB_ins_code",
    "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv",
    "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id",
    "pdbx_PDB_model_num",
)


def _cif_token(value: Any) -> str:
    """One mmCIF value, quoted when it has to be, never truncated.

    ``None`` becomes ``?`` (unknown) and an empty string becomes ``.``
    (inapplicable), which is how :func:`eagent.science.structure_io.read_mmcif`
    reads them back as ``None`` / ``""``. Writing ``1.00`` for an absent
    occupancy would fabricate a measurement; writing ``0.00`` for an absent
    B-factor would fabricate a perfect one.
    """
    if value is None:
        return "?"
    s = str(value)
    if s == "":
        return "."
    if any(ch.isspace() for ch in s) or s[0] in "'\"_#;$[]" or s.lower() in (
            "loop_", "stop_"):
        if "'" not in s:
            return f"'{s}'"
        if '"' not in s:
            return f'"{s}"'
        raise StructureSourceError(
            f"mmCIF value {s!r} contains whitespace and both quote characters "
            f"and cannot be written without changing it"
        )
    return s


def write_mmcif_text(structure: Structure | Iterable[Atom],
                     data_block: str = "structure") -> str:
    """Serialise atoms as a minimal mmCIF ``_atom_site`` loop.

    mmCIF rather than PDB because this project stores structures in mmCIF: PDB
    cannot hold a four-character component id, a chain id longer than one
    character, or a residue number outside -999..9999, and the usual
    workaround -- truncation -- changes chemical identity (see
    :mod:`eagent.science.structure_io`). ``type_symbol`` is always written, so
    the file reads back without the element having to be inferred.

    Only the coordinate loop is written. No header, no symmetry, no assembly
    records: this module does not parse them, and emitting fields it did not
    read would be asserting metadata nobody checked.
    """
    atoms = list(structure.atoms()) if isinstance(structure, Structure) \
        else list(structure)
    if not atoms:
        raise StructureSourceError("nothing to write: the atom selection is empty")
    block = "".join(ch if (ch.isalnum() or ch in "_-.") else "_"
                    for ch in (data_block or "structure"))
    lines = [f"data_{block}", "#", "loop_"]
    lines += [f"_atom_site.{c}" for c in _CIF_COLUMNS]
    for n, a in enumerate(atoms, start=1):
        row = (
            "HETATM" if a.is_hetatm else "ATOM",
            n,
            a.element.strip().upper(),
            a.name.strip(),
            a.altloc.strip(),
            a.resname.strip(),
            a.chain.strip(),
            a.resseq,
            a.icode.strip(),
            f"{a.x:.3f}", f"{a.y:.3f}", f"{a.z:.3f}",
            None if a.occupancy is None else f"{a.occupancy:.2f}",
            None if a.bfactor_or_plddt is None else f"{a.bfactor_or_plddt:.2f}",
            a.resseq,
            a.resname.strip(),
            a.chain.strip(),
            a.name.strip(),
            a.model,
        )
        lines.append(" ".join(_cif_token(v) for v in row))
    lines.append("#")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# small table helpers
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    """One TSV cell: tabs and newlines flattened, ``None`` written as empty.

    An empty cell means "not measured". It is deliberately distinct from
    ``0``, which would read as a measurement.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ";".join(_cell(v) for v in value)
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _write_tsv(path: Path, header: Sequence[str],
               rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(_cell(row.get(c)) for c in header) + "\n")
    return path


STRUCTURE_QC_COLUMNS: tuple[str, ...] = (
    "candidate_id", "structure_id", "source", "priority_rank", "selected",
    "selection_reason", "path", "format", "chain", "chain_choice_reason",
    "identity_to_candidate", "coverage", "n_observed_residues",
    "covers_candidate_1based", "missing_regions_1based", "is_mutant",
    "mutations_in_structure", "numbering_offset", "n_polymer_chains",
    "assembly_in_file", "template_assembly_state", "assembly_verdict",
    "ligands", "metals", "cofactor_code", "cofactor_state",
    "required_cofactor", "required_cofactor_state", "cofactor_verdict",
    "mean_plddt", "pocket_plddt", "pocket_confidence", "n_pocket_residues",
    "pocket_definition", "pae_mean", "pae_pocket_mean", "qc_notes",
)

RESIDUE_MAP_COLUMNS: tuple[str, ...] = (
    "candidate_id", "structure_id", "chain", "candidate_index0",
    "candidate_resnum1", "candidate_letter", "observed", "author_chain",
    "author_resseq", "author_icode", "structure_resname", "structure_letter",
    "is_mismatch", "n_atoms", "n_heavy_atoms", "mean_bfactor_or_plddt",
    "bvalue_kind", "in_pocket", "catalytic_role",
)


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------


class PrepareStructures(ScientificInterface):
    """Select and QC one structure per candidate, with the reason recorded.

    The step consumes candidates (with their family call and catalytic
    mapping), a local structure index, and the catalytic templates; it produces
    a :class:`~eagent.schemas.candidate.StructureRecord` per candidate plus the
    tables a reviewer needs to disagree with the selection.
    """

    name = "prepare_structures"
    description = ("Choose a structure per candidate by an explicit source "
                   "priority, QC it against the candidate sequence and the "
                   "catalytic template, and record pocket-local confidence")
    required_fields: tuple[str, ...] = ()
    required_approvals: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ("annotate_family",)
    version = "0.1.0"

    # -- entry point -------------------------------------------------------
    def execute(
        self,
        ctx: RunContext,
        *,
        candidates: Sequence[Candidate] | Sequence[Mapping[str, Any]]
        | Mapping[str, Any] | None = None,
        index: StructureIndex | None = None,
        index_path: str | Path | None = None,
        catalytic_templates: Mapping[str, CatalyticTemplate]
        | Sequence[CatalyticTemplate] | None = None,
        policy: StructurePolicy | None = None,
        predictor: StructurePredictor | None = None,
        access_policy: AccessPolicy | None = None,
        **_: Any,
    ) -> ToolResult:
        """Prepare one structure per candidate, or say exactly what is missing.

        ``candidates`` may arrive as models or as the serialised mapping the
        previous step published; :func:`~eagent.tools.handoff.as_candidates`
        settles that here, at the boundary, so a dict handed to code typed
        against the model fails with a named payload instead of an
        ``AttributeError`` raised halfway through a structure assessment.
        """
        policy = policy or StructurePolicy()
        candidates = as_candidates(candidates, source=self.name)
        if not candidates:
            return ToolResult.failure(
                self.name,
                "no candidates were supplied; prepare_structures cannot invent "
                "the proteins to find structures for",
                code="missing_input",
            )
        if index is None:
            if index_path is None:
                return ToolResult.failure(
                    self.name,
                    "no structure index was supplied. Offline is the default, so "
                    "the only structures available are the ones already on this "
                    "machine: pass index= or index_path= pointing at a JSON "
                    "index of cached coordinate files",
                    code="missing_structure_index",
                )
            try:
                index = StructureIndex.from_json(index_path)
            except (StructureSourceError, ValueError, json.JSONDecodeError) as exc:
                return ToolResult.failure(
                    self.name, f"structure index could not be loaded: {exc}",
                    code="bad_structure_index")

        templates = _template_lookup(catalytic_templates)
        result = ToolResult(status=Status.SUCCESS)
        struct_dir = ctx.dir(self.name, "structures")
        conf_dir = ctx.dir(self.name, "confidence_metrics")

        assessments: list[StructureAssessment] = []
        selected: dict[str, StructureAssessment] = {}
        missing: list[dict[str, Any]] = []
        predictions_made: dict[str, str] = {}
        disclosure_notes: list[str] = []

        for cand in candidates:
            template = _template_for(cand, templates)
            if template is None:
                result.add_uncertainty(
                    "no_catalytic_template",
                    f"Which catalytic template applies to {cand.candidate_id}? "
                    f"Without one the cofactor requirement and the assembly "
                    f"state cannot be checked, and they were not assumed.",
                    affects=[cand.candidate_id],
                    resolvable_by="a sourced CatalyticTemplate for this family",
                )

            entries = index.for_candidate(cand.candidate_id,
                                          cand.sequence_record.sequence_sha256)
            per_candidate: list[StructureAssessment] = []
            for entry in entries:
                try:
                    per_candidate.append(
                        self._assess(ctx, cand, entry, index, template, policy,
                                     conf_dir)
                    )
                except (StructureSourceError, StructureParseError, NumberingError,
                        GeometryError) as exc:
                    result.add_flag("structure_unreadable", Severity.WARN,
                                    f"{entry.structure_id}: {exc}",
                                    subject=cand.candidate_id)

            best_rank = min((a.record.priority_rank for a in per_candidate),
                            default=PRIORITY_REJECTED)
            has_prediction = any(a.entry.source.carries_plddt
                                 for a in per_candidate)
            # Predict when nothing on disk reaches the reusable-prediction tier,
            # or when the operator explicitly wants a model of the candidate
            # alongside an experimental structure (useful when the experimental
            # entry is a mutant or is missing the substrate-binding loops).
            need_prediction = (best_rank > PRIORITY_REUSABLE_PREDICTION
                               or (policy.predict_when_experimental_exists
                                   and not has_prediction))
            if need_prediction and policy.allow_new_prediction:
                made = self._predict(ctx, cand, index, template, policy, predictor,
                                     access_policy, conf_dir, result,
                                     disclosure_notes)
                if made is not None:
                    per_candidate.append(made)
                    predictions_made[cand.candidate_id] = made.entry.structure_id

            if not per_candidate:
                missing.append({
                    "candidate_id": cand.candidate_id,
                    "sequence_sha256": cand.sequence_record.sequence_sha256,
                    "accession": cand.sequence_record.accession,
                    "needed": ("an experimental entry or a predicted model for "
                               "this sequence, indexed in "
                               f"{index.source_path or index.root}"),
                })
                continue

            per_candidate.sort(key=lambda a: (a.record.priority_rank,
                                              -(a.record.sequence_identity_to_candidate
                                                or 0.0)))
            best = per_candidate[0]
            if best.record.priority_rank >= PRIORITY_REJECTED:
                missing.append({
                    "candidate_id": cand.candidate_id,
                    "sequence_sha256": cand.sequence_record.sequence_sha256,
                    "accession": cand.sequence_record.accession,
                    "needed": (f"every indexed structure was rejected: "
                               f"{best.selection_reason}"),
                })
            else:
                best.selected = True
                selected[cand.candidate_id] = best
                self._store_structure(best, struct_dir)
            assessments.extend(per_candidate)

        for a in assessments:
            for code, severity, message in a.flags:
                result.add_flag(code, severity, message,
                                subject=f"{a.candidate_id}/{a.entry.structure_id}")

        qc_path = _write_tsv(ctx.path(self.name, "structure_qc.tsv"),
                             STRUCTURE_QC_COLUMNS,
                             [self._qc_row(a) for a in assessments])
        map_path = _write_tsv(ctx.path(self.name, "residue_atom_mapping.tsv"),
                              RESIDUE_MAP_COLUMNS,
                              [row for a in assessments
                               for row in self._mapping_rows(a, _roles_of(a))])

        result.artifacts.append(Artifact(
            key="structures_dir", path=str(struct_dir), kind="object",
            n_records=len(selected),
            summary="selected structures, stored as mmCIF, one per candidate",
        ))
        result.artifacts.append(Artifact(
            key="structure_qc", path=str(qc_path), kind="table",
            sha256=sha256_file(qc_path), n_records=len(assessments),
            summary=("one row per candidate/structure pair: why it ranked where "
                     "it did, sequence agreement, ligands, cofactor state, "
                     "assembly and pocket-local confidence"),
        ))
        result.artifacts.append(Artifact(
            key="residue_atom_mapping", path=str(map_path), kind="table",
            sha256=sha256_file(map_path),
            summary=("candidate index <-> author numbering, residue by residue, "
                     "with per-residue atom counts and B-factor/pLDDT"),
        ))
        result.artifacts.append(Artifact(
            key="confidence_metrics_dir", path=str(conf_dir), kind="object",
            n_records=len(assessments),
            summary="per-structure mean and pocket pLDDT, PAE summary, notes",
        ))

        self._finalise(ctx, result, candidates, assessments, selected, missing,
                       policy, index, predictions_made, disclosure_notes)
        return result

    # -- per-structure assessment -----------------------------------------
    def _assess(self, ctx: RunContext, cand: Candidate,
                entry: StructureIndexEntry, index: StructureIndex,
                template: CatalyticTemplate | None, policy: StructurePolicy,
                conf_dir: Path) -> StructureAssessment:
        """Read one structure and measure everything that can be measured."""
        path = index.resolve(entry)
        structure = read_structure(path, structure_id=entry.structure_id)
        chain, rmap, choice = select_chain(structure, cand.sequence,
                                           entry.chain)
        inventory = ligand_inventory(structure)
        cofactor = assess_cofactor(structure, template)
        assembly = assess_assembly(structure, template)
        rank, reason = classify_source(entry, rmap.structure_identity, cofactor,
                                       inventory, policy)

        seed_atoms, pocket_def = self._pocket_seeds(structure, chain, rmap, cand,
                                                    template)
        pocket = pocket_residues(structure, seed_atoms,
                                 policy.pocket_radius_angstrom) if seed_atoms else []
        pae, pae_indices, pae_note = self._pae_for(entry, index, cand, rmap, pocket)
        confidence = pocket_confidence(structure, chain, pocket, entry.source,
                                       pocket_def, pae, pae_indices)
        if pae_note:
            confidence = PocketConfidence(
                confidence.mean_plddt, confidence.pocket_plddt,
                confidence.n_pocket_residues, confidence.pocket_definition,
                confidence.pae_mean, confidence.pae_pocket_mean,
                confidence.notes + (pae_note,),
            )

        offset, offset_note = _constant_offset(rmap)
        mutations = mutation_tokens(rmap)
        observed = rmap.observed_indices()
        qc_notes = list(structure.quality_notes()) + list(rmap.notes) \
            + list(confidence.notes)
        if offset_note:
            qc_notes.append(offset_note)
        if not seed_atoms:
            qc_notes.append(
                "pocket could not be localised: no ligand and no mapped "
                "catalytic residue, so no pocket-local confidence was computed")

        record = StructureRecord(
            structure_id=entry.structure_id,
            source=entry.source.value,
            path=str(path),
            format="mmcif" if path.suffix.lower() in (".cif", ".mmcif") else "pdb",
            priority_rank=rank,
            sequence_identity_to_candidate=rmap.structure_identity,
            covers_residues=((observed[0] + 1, observed[-1] + 1)
                             if observed else None),
            missing_regions=[(s + 1, e + 1) for s, e in rmap.unobserved_regions()],
            is_mutant_relative_to_candidate=bool(mutations),
            mutations_in_structure=mutations,
            assembly=entry.assembly_in_file or f"{assembly.n_polymer_chains} "
                                               f"polymer chain(s) in file",
            bound_ligands=sorted(inventory),
            cofactor_in_structure=cofactor.found_code,
            cofactor_state_in_structure=cofactor.found_state,
            ligand_source=(LigandSource.EXPERIMENTAL_OBSERVED
                           if entry.source.is_experimental and inventory
                           else LigandSource.UNKNOWN),
            mean_plddt=confidence.mean_plddt,
            pocket_plddt=confidence.pocket_plddt,
            numbering_offset=offset,
            qc_notes=qc_notes,
        )

        assessment = StructureAssessment(
            candidate_id=cand.candidate_id, entry=entry, structure=structure,
            chain_choice=choice, residue_map=rmap, record=record,
            cofactor=cofactor, assembly=assembly, confidence=confidence,
            pocket_residue_keys=tuple(r.key for r in pocket),
            catalytic_roles={str(role): int(idx) for role, idx
                             in cand.catalytic_mapping.role_to_index.items()},
            selection_reason=reason,
        )
        self._flag(assessment, template, policy)
        _write_confidence_json(conf_dir, assessment)
        return assessment

    def _pocket_seeds(self, structure: Structure, chain: Chain, rmap: ResidueMap,
                      cand: Candidate,
                      template: CatalyticTemplate | None) -> tuple[list[Atom], str]:
        """Atoms the pocket shell is grown from, and a description of them.

        A bound cofactor or substrate is the best seed, because it is where the
        chemistry happens and it is observed. Failing that, the mapped
        catalytic residues are used. Failing both, there is no pocket: the
        function returns nothing rather than falling back to the centre of
        mass, which in a two-domain MDR/ADH is the interdomain cleft and in a
        barrel fold is solvent.
        """
        ligands = [r for r in structure.ligands() if len(r.atoms) > 1]
        if template is not None and template.cofactor_ligand_codes:
            wanted = {c.strip().upper() for c in template.cofactor_ligand_codes}
            preferred = [r for r in ligands if r.resname.strip().upper() in wanted]
            if preferred:
                return ([a for r in preferred for a in r.heavy_atoms()],
                        f"heavy atoms of bound {sorted({r.resname.strip() for r in preferred})}")
        if ligands:
            return ([a for r in ligands for a in r.heavy_atoms()],
                    f"heavy atoms of bound {sorted({r.resname.strip() for r in ligands})}")

        atoms: list[Atom] = []
        roles: list[str] = []
        for role, idx in sorted(cand.catalytic_mapping.role_to_index.items()):
            try:
                pos = rmap.to_author(int(idx))
            except NumberingError:
                continue
            if pos is None:
                continue
            res = chain.residue(pos.resseq, pos.icode)
            if res is None:
                continue
            atoms.extend(res.heavy_atoms())
            roles.append(f"{role}@{pos}")
        if atoms:
            return atoms, f"heavy atoms of mapped catalytic residues {roles}"
        return [], "no seed available"

    def _pae_for(self, entry: StructureIndexEntry, index: StructureIndex,
                 cand: Candidate, rmap: ResidueMap,
                 pocket: Sequence[Residue]) -> tuple[list[list[float]] | None,
                                                     list[int] | None, str | None]:
        """Load the companion PAE matrix and work out which rows are the pocket.

        The matrix axis is the *predicted* sequence. Tying it to candidate
        positions is only legitimate when the prediction was made from this
        exact sequence, so the mapping is attempted only at full identity and
        matching length; otherwise the matrix is summarised globally and the
        reason is recorded. An off-by-a-few row mapping would report the PAE of
        the wrong loop.
        """
        pae_path = index.resolve_pae(entry)
        if pae_path is None:
            return None, None, None
        try:
            matrix = read_pae_matrix(pae_path)
        except (StructureSourceError, ValueError, json.JSONDecodeError) as exc:
            return None, None, f"PAE file {pae_path} could not be read: {exc}"

        same = (rmap.structure_identity == 1.0
                and len(matrix) == len(cand.sequence))
        if not same:
            return matrix, None, (
                f"PAE rows were not tied to candidate positions: the matrix has "
                f"{len(matrix)} rows, the candidate has {len(cand.sequence)} "
                f"residues at identity {_pct(rmap.structure_identity)}")
        wanted = {r.key for r in pocket}
        indices: list[int] = []
        for i in rmap.observed_indices():
            pos = rmap.index_to_author[i]
            for key in wanted:
                if key[0] == pos.chain and key[1] == pos.resseq \
                        and key[2] == pos.icode:
                    indices.append(i)
                    break
        return matrix, indices or None, None

    def _flag(self, a: StructureAssessment, template: CatalyticTemplate | None,
              policy: StructurePolicy) -> None:
        """Attach the QC flags this structure earns."""
        if a.cofactor.is_blocker:
            a.flags.append(("cofactor_oxidation_state_mismatch", Severity.BLOCKER,
                            a.cofactor.message))
        elif a.cofactor.verdict == "cofactor_state_unknown":
            a.flags.append(("cofactor_state_unknown", Severity.WARN,
                            a.cofactor.message))
        elif a.cofactor.verdict == "cofactor_absent":
            a.flags.append(("cofactor_absent_from_structure", Severity.WARN,
                            a.cofactor.message))

        if a.assembly.verdict == "subunit_deficient":
            a.flags.append(("assembly_subunit_deficient", Severity.WARN,
                            a.assembly.message))
        elif a.assembly.verdict in ("assembly_unparsed", "template_silent"):
            a.flags.append(("assembly_unverified", Severity.INFO,
                            a.assembly.message))

        if a.record.is_mutant_relative_to_candidate:
            a.flags.append(("structure_is_mutant", Severity.WARN,
                            f"the structure differs from the candidate at "
                            f"{len(a.record.mutations_in_structure)} position(s) "
                            f"({', '.join(a.record.mutations_in_structure[:8])}); "
                            f"check that none of them is a catalytic residue "
                            f"before measuring geometry"))

        if a.record.missing_regions:
            shown = ", ".join(f"{s}-{e}" for s, e in a.record.missing_regions[:5])
            a.flags.append(("missing_regions", Severity.INFO,
                            f"candidate residues {shown} have no coordinates; "
                            f"no structural claim can be made about them"))

        if a.entry.source.carries_plddt:
            level = a.record.pocket_confidence
            if level is ConfidenceLevel.WEAK:
                a.flags.append(("pocket_confidence_weak", Severity.WARN,
                                f"pocket pLDDT {a.record.pocket_plddt:.1f} "
                                f"(mean {_fmt(a.record.mean_plddt)}): the active "
                                f"site is poorly predicted even if the fold is "
                                f"not"))
            elif level is ConfidenceLevel.INSUFFICIENT:
                a.flags.append(("pocket_confidence_unmeasured", Severity.WARN,
                                "no pocket-local pLDDT could be computed; the "
                                "global mean must not be used in its place"))

        unobserved_roles = _unobserved_catalytic_roles(a)
        if unobserved_roles:
            a.flags.append(("catalytic_residue_unobserved", Severity.WARN,
                            f"mapped catalytic role(s) {unobserved_roles} have no "
                            f"coordinates in this structure; their geometry "
                            f"cannot be measured and must be reported as "
                            f"unevaluated, not as failing"))

        if a.record.priority_rank == PRIORITY_HOMOLOGUE_SCAFFOLD:
            a.flags.append(("homologue_structure", Severity.WARN,
                            a.selection_reason))
        if a.record.priority_rank >= PRIORITY_REJECTED:
            a.flags.append(("structure_rejected", Severity.INFO,
                            a.selection_reason))

    # -- prediction --------------------------------------------------------
    def _predict(self, ctx: RunContext, cand: Candidate, index: StructureIndex,
                 template: CatalyticTemplate | None, policy: StructurePolicy,
                 predictor: StructurePredictor | None,
                 access_policy: AccessPolicy | None, conf_dir: Path,
                 result: ToolResult,
                 disclosure_notes: list[str]) -> StructureAssessment | None:
        """Run a new prediction, or report precisely why none was made.

        Returns ``None`` instead of raising when the predictor is simply
        absent: one candidate without a model must not abort the other
        ninety-nine. The absence is recorded as an uncertainty naming the tool,
        which is what makes it fixable.
        """
        engine = predictor or UnavailablePredictor()
        if not engine.is_available():
            result.add_uncertainty(
                "no_structure_predictor",
                f"{cand.candidate_id} has no usable structure on disk and no "
                f"predictor is installed ('{engine.name}'), so no model was "
                f"produced. Nothing was substituted for it.",
                affects=[cand.candidate_id],
                resolvable_by=getattr(engine, "install_hint",
                                      "install a structure predictor"),
            )
            result.add_next("prepare_structures",
                            "Re-run once a predictor is available or the model "
                            "is added to the structure index",
                            {"candidate_id": cand.candidate_id}, requires_human=True)
            return None
        if not ctx.policy.allow_gpu_models:
            result.add_flag("prediction_skipped", Severity.INFO,
                            f"{cand.candidate_id}: ctx.policy.allow_gpu_models is "
                            f"False, so no new prediction was run")
            return None

        out_dir = ctx.dir(self.name, "predictions", cand.candidate_id)
        notes = guard_sequence_submission(engine, cand.sequence, ctx, access_policy)
        disclosure_notes.extend(f"{cand.candidate_id}: {n}" for n in notes)
        outcome = engine.predict(PredictionRequest(
            candidate_id=cand.candidate_id, sequence=cand.sequence,
            out_dir=out_dir, seed=ctx.seed_for(f"{self.name}:{cand.candidate_id}"),
        ))
        entry = StructureIndexEntry(
            structure_id=f"{cand.candidate_id}__{outcome.model_name}",
            path=str(outcome.path), source=StructureSource.PREDICTED,
            candidate_id=cand.candidate_id,
            sequence_sha256=cand.sequence_record.sequence_sha256,
            database=outcome.model_name, database_version=outcome.model_version,
            pae_json=str(outcome.pae_json) if outcome.pae_json else None,
            evidence=(f"predicted in this run by {outcome.model_name} "
                      f"{outcome.model_version}. {outcome.notes}").strip(),
        )
        index.entries.append(entry)
        return self._assess(ctx, cand, entry, index, template, policy, conf_dir)

    # -- outputs -----------------------------------------------------------
    def _store_structure(self, a: StructureAssessment, struct_dir: Path) -> None:
        """Copy the chosen structure into the run as mmCIF.

        Re-serialised through the parser rather than copied byte for byte, so
        the stored file is exactly the atoms every measurement in this run was
        made on. A copied original can contain a second NMR model, or records
        the reader ignored, and then the artifact and the measurement describe
        different things. The source path stays in the QC table.
        """
        out = struct_dir / f"{a.candidate_id}__{a.entry.structure_id}.cif"
        out.write_text(write_mmcif_text(a.structure, a.entry.structure_id),
                       encoding="utf-8")
        a.record.path = str(out)
        a.record.format = "mmcif"

    def _qc_row(self, a: StructureAssessment) -> dict[str, Any]:
        r = a.record
        return {
            "candidate_id": a.candidate_id,
            "structure_id": r.structure_id,
            "source": r.source,
            "priority_rank": r.priority_rank,
            "selected": a.selected,
            "selection_reason": a.selection_reason,
            "path": r.path,
            "format": r.format,
            "chain": a.chain_choice.chain_id,
            "chain_choice_reason": a.chain_choice.reason,
            "identity_to_candidate": r.sequence_identity_to_candidate,
            "coverage": a.residue_map.coverage,
            "n_observed_residues": len(a.residue_map.observed_indices()),
            "covers_candidate_1based": (f"{r.covers_residues[0]}-"
                                        f"{r.covers_residues[1]}"
                                        if r.covers_residues else None),
            "missing_regions_1based": [f"{s}-{e}" for s, e in r.missing_regions],
            "is_mutant": r.is_mutant_relative_to_candidate,
            "mutations_in_structure": r.mutations_in_structure,
            "numbering_offset": r.numbering_offset,
            "n_polymer_chains": a.assembly.n_polymer_chains,
            "assembly_in_file": a.entry.assembly_in_file,
            "template_assembly_state": a.assembly.template_assembly_state,
            "assembly_verdict": a.assembly.verdict,
            "ligands": r.bound_ligands,
            "metals": metals_in(a.structure),
            "cofactor_code": r.cofactor_in_structure,
            "cofactor_state": r.cofactor_state_in_structure.value,
            "required_cofactor": a.cofactor.required_cofactor,
            "required_cofactor_state": a.cofactor.required_state.value,
            "cofactor_verdict": a.cofactor.verdict,
            "mean_plddt": r.mean_plddt,
            "pocket_plddt": r.pocket_plddt,
            "pocket_confidence": r.pocket_confidence.value,
            "n_pocket_residues": a.confidence.n_pocket_residues,
            "pocket_definition": a.confidence.pocket_definition,
            "pae_mean": a.confidence.pae_mean,
            "pae_pocket_mean": a.confidence.pae_pocket_mean,
            "qc_notes": r.qc_notes,
        }

    def _mapping_rows(self, a: StructureAssessment,
                      roles: Mapping[int, str]) -> list[dict[str, Any]]:
        """One row per candidate residue: the correspondence, written out.

        Unobserved residues get a row too, with ``observed=false``. Omitting
        them would make a gap indistinguishable from a residue that happens to
        be missing from the table, and the gap is the thing a reviewer is
        looking for.
        """
        rmap = a.residue_map
        chain = a.structure.chain(a.chain_choice.chain_id)
        pocket = set(a.pocket_residue_keys)
        kind = "plddt" if a.entry.source.carries_plddt else "bfactor"
        mismatched = {i for i, _c, _s, _p in rmap.mismatches}
        rows: list[dict[str, Any]] = []
        for i, letter in enumerate(rmap.candidate_sequence):
            pos = rmap.index_to_author.get(i)
            res = chain.residue(pos.resseq, pos.icode) if (pos and chain) else None
            rows.append({
                "candidate_id": a.candidate_id,
                "structure_id": a.entry.structure_id,
                "chain": a.chain_choice.chain_id,
                "candidate_index0": i,
                "candidate_resnum1": i + 1,
                "candidate_letter": letter,
                "observed": pos is not None,
                "author_chain": pos.chain if pos else None,
                "author_resseq": pos.resseq if pos else None,
                "author_icode": (pos.icode or None) if pos else None,
                "structure_resname": res.resname.strip() if res else None,
                "structure_letter": (residue_one_letter(res.resname)
                                     if res else None),
                "is_mismatch": i in mismatched,
                "n_atoms": len(res.atoms) if res else None,
                "n_heavy_atoms": len(res.heavy_atoms()) if res else None,
                "mean_bfactor_or_plddt": _residue_bvalue(res) if res else None,
                "bvalue_kind": kind if res else None,
                "in_pocket": (res.key in pocket) if res else False,
                "catalytic_role": roles.get(i),
            })
        return rows

    def _finalise(self, ctx: RunContext, result: ToolResult,
                  candidates: Sequence[Candidate],
                  assessments: Sequence[StructureAssessment],
                  selected: Mapping[str, StructureAssessment],
                  missing: Sequence[Mapping[str, Any]], policy: StructurePolicy,
                  index: StructureIndex, predictions: Mapping[str, str],
                  disclosure_notes: Sequence[str]) -> None:
        """Set status, provenance and the data payload."""
        blockers = [a for a in assessments if a.selected
                    and any(s is Severity.BLOCKER for _c, s, _m in a.flags)]
        if not selected:
            result.status = Status.FAILED
            result.message = (
                f"no usable structure for any of {len(candidates)} candidate(s). "
                f"Needed: " + "; ".join(
                    f"{m['candidate_id']} ({m['sequence_sha256']}) -> {m['needed']}"
                    for m in missing[:5])
            )
            # A failed status with no blocking code is unclassifiable: the
            # controller cannot tell a routine structure-cache miss (repairable
            # input) from an internal fault, and a run that only needed a
            # coordinate file dead-ends as "unclassified".
            result.add_flag("no_usable_structure", Severity.BLOCKER,
                            result.message, subject="structures")
        elif missing or blockers:
            result.status = Status.PARTIAL
            result.message = (
                f"{len(selected)}/{len(candidates)} candidate(s) have a structure; "
                f"{len(missing)} missing, {len(blockers)} selected structure(s) "
                f"carry a blocking QC problem"
            )
        else:
            result.message = (f"{len(selected)} candidate(s) prepared from "
                              f"{len(assessments)} assessed structure(s)")

        for m in missing:
            result.add_uncertainty(
                "structure_unavailable",
                f"Where is a structure for {m['candidate_id']}? {m['needed']}",
                affects=[str(m["candidate_id"])],
                resolvable_by="add the file to the structure index, or enable a "
                              "predictor",
            )
        if missing:
            result.add_next(
                "operator_fetch_structures",
                "Offline run: these coordinate files must be fetched into the "
                "local cache before the complex step can run",
                {"candidates": [m["candidate_id"] for m in missing],
                 "cache": str(index.source_path or index.root)},
                requires_human=True,
            )
        if selected:
            result.add_next(
                "model_complexes",
                "Structures are prepared; assemble the full catalytic system "
                "(protein + substrate + correctly-reduced cofactor + metals)",
                {"candidates": sorted(selected)},
            )

        result.provenance = Provenance(
            tool=self.name, tool_version=self.version,
            inputs_sha256={
                "candidates": sha256_obj([c.sequence_record.sequence_sha256
                                          for c in candidates]),
                "structure_index": sha256_obj(
                    [e.model_dump(mode="json") for e in index.entries]),
            },
            databases={e.database: (e.database_version or "unrecorded")
                       for e in index.entries if e.database},
            models={sid.split("__", 1)[-1]: "see structure index"
                    for sid in predictions.values()},
            parameters={
                "policy": policy.model_dump(mode="json"),
                "source_priority": {
                    "1_matching_experimental_complex": PRIORITY_MATCHING_COMPLEX,
                    "2_experimental_structure_of_enzyme": PRIORITY_EXPERIMENTAL_ENZYME,
                    "3_reusable_prediction": PRIORITY_REUSABLE_PREDICTION,
                    "4_new_prediction": PRIORITY_NEW_PREDICTION,
                    "5_homologue_scaffold": PRIORITY_HOMOLOGUE_SCAFFOLD,
                },
                "structure_index": str(index.source_path or index.root),
                "cache_version": index.cache_version,
                "n_candidates": len(candidates),
                "n_assessed": len(assessments),
                "n_selected": len(selected),
                "new_predictions": dict(predictions),
                "allow_network": ctx.policy.allow_network,
                "disclosure_notes": list(disclosure_notes),
                "numbering_basis": ("residue_atom_mapping.tsv carries both the "
                                    "0-based candidate index and the structure's "
                                    "author numbering; they are different axes"),
            },
            random_seed=ctx.seed_for(self.name),
        )

        result.data.update({
            # Republished in the one serialised form every consumer reads back
            # through ``handoff.as_candidates``, so model_complexes can be wired
            # straight from this step's data mapping.
            CANDIDATES_KEY: serialise_candidates(candidates),
            "structures": {cid: a.record.model_dump(mode="json")
                           for cid, a in selected.items()},
            "selected": {
                cid: {"structure_id": a.entry.structure_id,
                      "path": a.record.path,
                      "chain": a.chain_choice.chain_id,
                      "source": a.record.source,
                      "priority_rank": a.record.priority_rank,
                      "reason": a.selection_reason}
                for cid, a in selected.items()},
            "assessed": [{"candidate_id": a.candidate_id,
                          "structure_id": a.entry.structure_id,
                          "priority_rank": a.record.priority_rank,
                          "selected": a.selected,
                          "reason": a.selection_reason}
                         for a in assessments],
            "missing": list(missing),
            "confidence_caveat": ("mean pLDDT describes the fold; only "
                                  "pocket_plddt describes the residues a "
                                  "catalytic geometry depends on"),
        })


# ---------------------------------------------------------------------------
# helpers used by the interface
# ---------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.1f}"


def _template_lookup(
    templates: Mapping[str, CatalyticTemplate] | Sequence[CatalyticTemplate] | None,
) -> dict[str, CatalyticTemplate]:
    """Index catalytic templates by id and by family name, for lookup either way."""
    out: dict[str, CatalyticTemplate] = {}
    if templates is None:
        return out
    items = templates.values() if isinstance(templates, Mapping) else templates
    for t in items:
        out[t.template_id] = t
        out.setdefault(f"family:{t.family_name}", t)
    return out


def _template_for(cand: Candidate,
                  templates: Mapping[str, CatalyticTemplate]) -> CatalyticTemplate | None:
    """The catalytic template this candidate was annotated against, or ``None``.

    Falls back to the family name only, never to "the only template loaded":
    applying an SDR's catalytic residues to an AKR candidate produces a
    complete and entirely fictional mechanism.
    """
    tid = cand.catalytic_mapping.catalytic_template_id
    if tid and tid in templates:
        return templates[tid]
    family = cand.family.family_name
    if family:
        return templates.get(f"family:{family}")
    return None


def _roles_of(a: StructureAssessment) -> dict[int, str]:
    """Candidate index -> catalytic role label, for the mapping table.

    Several roles can land on one residue (a catalytic lysine that is also the
    cofactor anchor), so the labels are joined rather than one silently
    overwriting the other.
    """
    out: dict[int, list[str]] = {}
    for role, idx in sorted(a.catalytic_roles.items()):
        out.setdefault(int(idx), []).append(role)
    return {i: "+".join(labels) for i, labels in out.items()}


def _unobserved_catalytic_roles(a: StructureAssessment) -> list[str]:
    """Catalytic roles that have no coordinates in this structure.

    Worth its own flag because the consequence is specific: the constraints
    naming those roles come back from
    :func:`eagent.science.geometry.measure_constraint` as ``None`` -- not
    measured -- and a reader skimming a geometry report will otherwise take an
    unmeasured constraint for a failed one, or for a passed one.
    """
    out: list[str] = []
    for role, idx in sorted(a.catalytic_roles.items()):
        try:
            if not a.residue_map.is_observed(int(idx)):
                out.append(f"{role}@candidate_index_{idx}")
        except NumberingError:
            out.append(f"{role}@candidate_index_{idx} (index outside the sequence)")
    return out


def _write_confidence_json(conf_dir: Path, a: StructureAssessment) -> Path:
    """One JSON per assessed structure, so the numbers can be re-read as data."""
    path = conf_dir / f"{a.candidate_id}__{a.entry.structure_id}.json"
    payload = {
        "candidate_id": a.candidate_id,
        "structure_id": a.entry.structure_id,
        "source": a.entry.source.value,
        "b_value_kind": "plddt" if a.entry.source.carries_plddt else "bfactor",
        "mean_plddt": a.confidence.mean_plddt,
        "pocket_plddt": a.confidence.pocket_plddt,
        "pocket_confidence_level": a.record.pocket_confidence.value,
        "n_pocket_residues": a.confidence.n_pocket_residues,
        "pocket_definition": a.confidence.pocket_definition,
        "pae_mean": a.confidence.pae_mean,
        "pae_pocket_mean": a.confidence.pae_pocket_mean,
        "notes": list(a.confidence.notes),
        "caveat": ("a high mean pLDDT with a low pocket pLDDT means the fold is "
                   "confident and the active site is not; only the pocket value "
                   "bears on catalytic geometry"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path
