"""Interface ``model_complexes``: build the whole catalytic system, twice.

Why this module is shaped the way it is
---------------------------------------

**A complex missing the cofactor is not a model of the reaction.** The single
most common structural error in a ketoreductase campaign is to dock the ketone
into the apo pocket, measure its distance to the catalytic tyrosine, and
report a geometry. Hydride comes from the nicotinamide C4 of NAD(P)H; without
the cofactor in the model there is no donor, the substrate relaxes into the
space the cofactor should occupy, and every distance measured is a distance in
a protein that cannot do the chemistry. :func:`validate_assembly` therefore
refuses to start: the requested assembly is checked, component by component,
against the family's :class:`~eagent.schemas.templates.CatalyticTemplate`
(assembly state, substrate, cofactor *in the required oxidation state*,
required metals) and a missing or mis-oxidised component is a **BLOCKER**, not
a warning. An oxidised NAD+ in the pocket is treated as a missing cofactor,
because that is what it is mechanistically.

**Two routes, because they fail differently.** Template-guided pocket docking
(route A) places a ligand into a fixed receptor: it inherits every error in
the receptor, including a side chain rotamer refined against a different
ligand, but the protein is experimental. Joint protein-ligand-cofactor
prediction (route B) builds everything at once: it can close a pocket around a
substrate that does not belong there, and its confidence numbers are about
structural plausibility. Agreement between the two is worth something
precisely because their artefacts are unrelated; a single route's output is a
hypothesis with no second opinion. Both are adapter seams, absent by default,
and both are injected for testing.

**All major poses are kept.** Retaining the single best-scoring pose is how a
modelling artefact becomes a conclusion: the one pose that happened to score
best is then the only pose measured, the only pose whose face of approach is
counted, and the stereochemical call rests on a sample of one. Every pose a
runner returns is written out and indexed here; poses are grouped by
heavy-atom RMSD so that "six poses" and "six copies of one pose" can be told
apart, but nothing is discarded, and the downstream robustness statistic in
:mod:`eagent.science.robustness` is only meaningful over the full set.

**A restrained distance is not evidence.** If the docking run was told to hold
the nicotinamide C4 within 3.5 A of the carbonyl carbon, then measuring that
distance afterwards measures the restraint. Every pose records the restraint
names it was built under in
:attr:`~eagent.schemas.candidate.ComplexPose.restrained_constraints`, which is
what :class:`~eagent.science.robustness.CircularityGuard` reads to exclude
them from the independent evidence count. This is mandatory here:
:func:`assert_restraints_recorded` raises
:class:`~eagent.errors.CircularEvidenceError` rather than emitting a pose
whose restraints were silently forgotten, because a forgotten restraint is
indistinguishable from an honest measurement in every later artifact.

**A ranking score is not an activity ranking.** ``ranking_score``, ``iptm``
and ``pae`` are recorded verbatim, never rescaled, combined, or turned into a
total. They order *structural plausibility* -- how confident the model is that
these coordinates are a real complex -- and they are silent on turnover,
selectivity and ee. A candidate whose pose ranks first and whose catalytic
geometry is wrong is a confident model of an unproductive binding mode.
Likewise a docking score is a scoring-function value, comparable only within
one scoring function, one receptor preparation and one box.

**Licences are checked before the tool runs, not after.** Code, model weights,
the input databases a predictor reads (MSA and template databases) and the
outputs it produces carry *different* terms; several structure predictors ship
permissive code with non-commercial weights. They are therefore registered
separately in :class:`ToolRegistry`, and :func:`check_license` raises
:class:`~eagent.errors.LicenseError` when ``ctx.policy.allow_commercial_use``
conflicts with an entry -- including when the terms are simply unrecorded,
because an unverified licence is not a permission.

Calibration note
----------------
:data:`DEFAULT_BOX_PADDING_A`, :data:`DEFAULT_MIN_BOX_EDGE_A` and
:data:`DEFAULT_POSE_CLUSTER_RMSD_A` are **sampling-scope and pose-grouping
defaults**, not catalytic criteria: they decide where the search looks and how
poses are grouped for reporting, never whether a pose is catalytically
competent. That question is answered only by the sourced
:class:`~eagent.schemas.templates.GeometryConstraint` objects of a catalytic
template, downstream. All three need per-family calibration and are
overridable through :class:`ComplexPolicy`.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..connectors.base import AccessPolicy
from ..context import RunContext
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..errors import (
    CircularEvidenceError,
    EAgentError,
    LicenseError,
    TemplateError,
    ToolUnavailableError,
)
from ..provenance import sha256_file, sha256_obj
from ..schemas import (
    Candidate,
    CatalyticTemplate,
    CofactorState,
    ComplexPose,
    GeometryConstraint,
    LigandSource,
    StructureRecord,
    SubstrateSpec,
    cofactor_state_from_ligand_code,
)
from ..science.geometry import (
    DEFAULT_VDW_OVERLAP_TOLERANCE_A,
    GeometryError,
    clash_pairs,
)
from ..science.numbering import NumberingError, ResidueMap
from ..science.structure_io import (
    Atom,
    Structure,
    StructureParseError,
    read_structure,
)
from .base import ScientificInterface
from .handoff import CANDIDATES_KEY, as_candidates, serialise_candidates
from .prepare_structures import (
    METAL_ELEMENTS,
    StructureSourceError,
    expected_chain_count,
    guard_sequence_submission,
    ligand_inventory,
    metals_in,
    select_chain,
    write_mmcif_text,
)

__all__ = [
    "DEFAULT_BOX_PADDING_A",
    "DEFAULT_MIN_BOX_EDGE_A",
    "DEFAULT_MAX_BOX_EDGE_WARN_A",
    "DEFAULT_POSE_CLUSTER_RMSD_A",
    "RANKING_SCORE_CAVEAT",
    "ToolKind",
    "ToolRegistryEntry",
    "ToolRegistry",
    "ToolRegistryError",
    "check_license",
    "check_modification",
    "Route",
    "ComplexPolicy",
    "ComplexRequest",
    "ComponentRequirement",
    "AssemblyValidation",
    "validate_assembly",
    "SearchBox",
    "trusted_site_atoms",
    "catalytic_site_box",
    "RawPose",
    "DockingJob",
    "JointPredictionJob",
    "DockingRunner",
    "ComplexPredictor",
    "UnavailableDockingRunner",
    "UnavailableComplexPredictor",
    "pose_rmsd",
    "cluster_poses",
    "assert_restraints_recorded",
    "ModelComplexes",
]


# ---------------------------------------------------------------------------
# calibration defaults -- see the module docstring
# ---------------------------------------------------------------------------

#: Padding added around the trusted catalytic site to make the docking box.
#: A sampling-scope default needing per-family calibration: too small and a
#: productive pose cannot be represented, too large and the run is blind
#: docking wearing a pocket-docking label.
DEFAULT_BOX_PADDING_A: float = 4.0

#: Smallest box edge. Guards against a box collapsed onto two anchor atoms,
#: which would leave the substrate no room to adopt any orientation at all.
DEFAULT_MIN_BOX_EDGE_A: float = 12.0

#: Box edge above which the run is flagged as approaching blind docking. A
#: reporting threshold only; it changes no result.
DEFAULT_MAX_BOX_EDGE_WARN_A: float = 30.0

#: Heavy-atom RMSD below which two poses are reported as the same binding mode.
#: A *grouping* parameter for reporting: no pose is ever discarded by it, and
#: it carries no claim about catalytic competence. Needs calibration per
#: substrate size -- 2 A means something different for acetophenone and for a
#: steroid.
DEFAULT_POSE_CLUSTER_RMSD_A: float = 2.0

RANKING_SCORE_CAVEAT: str = (
    "ranking_score / iptm / pae order STRUCTURAL PLAUSIBILITY: how confident "
    "the model is that these coordinates are a real complex. They are not a "
    "catalytic activity ranking, not a binding affinity, and not a "
    "selectivity prediction. A docking score is a scoring-function value, "
    "comparable only within one scoring function, receptor preparation and "
    "box. Catalytic competence is decided downstream by sourced geometry "
    "constraints, pose robustness and experiment."
)


class ToolRegistryError(EAgentError):
    """A tool was about to be invoked without a registry entry.

    Refusing here rather than at licence-audit time is the point: once a
    predictor has run, its weights' terms have already been exercised and its
    outputs already exist.
    """


# ---------------------------------------------------------------------------
# tool registry: code, weights, input databases and outputs register separately
# ---------------------------------------------------------------------------


class ToolKind(str, enum.Enum):
    """What a registry entry describes.

    The four kinds exist because they carry different terms and a single
    "AlphaFold3: licence X" entry hides exactly the conflict this check is for:
    the code may be Apache-2.0 while the weights are non-commercial and
    output-restricted, and the MSA databases have terms of their own.
    """

    CODE = "code"
    MODEL_WEIGHTS = "model_weights"
    INPUT_DATABASE = "input_database"
    OUTPUT = "output"


@dataclass(frozen=True)
class ToolRegistryEntry:
    """One registered artefact a modelling run depends on.

    ``permits_commercial_use`` is tri-state on purpose. ``None`` means nobody
    has recorded the terms, which is different from "non-commercial" and very
    different from "permitted"; a commercial run is blocked by ``None`` exactly
    as it is by ``False``, because an unverified licence is not a permission.
    A stated licence must name where the statement came from, so a claim can be
    traced to whoever made it.
    """

    key: str
    kind: ToolKind
    display_name: str
    version: str | None = None
    license: str | None = None
    license_source: str | None = None
    permits_commercial_use: bool | None = None
    #: Whether the terms allow *modifying and redistributing* this artefact --
    #: vendoring a patched copy, publishing a derivative. Separate from
    #: commercial use because they come apart (a no-derivatives licence can
    #: allow running the code and forbid shipping a changed version of it),
    #: and tri-state for the same reason: ``None`` is unread, not permitted.
    permits_derivative_works: bool | None = None
    #: What the producer's terms say about the coordinates this tool emits.
    #: Registered separately from the weights because output restrictions
    #: outlive the run that produced them: a pose redistributed in a paper is
    #: still an output of the model that made it.
    output_terms: str | None = None
    needs_legal_review: bool = True
    install_hint: str = ""
    citations: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.license is not None and not self.license_source:
            raise ToolRegistryError(
                f"registry entry '{self.key}' states a licence without saying "
                f"where that statement came from")
        if self.permits_commercial_use is not None and self.license is None:
            raise ToolRegistryError(
                f"registry entry '{self.key}' claims a commercial-use "
                f"permission with no licence recorded; the permission would "
                f"rest on nothing")
        if self.permits_derivative_works is not None and self.license is None:
            raise ToolRegistryError(
                f"registry entry '{self.key}' claims a derivative-works "
                f"position with no licence recorded; the position would rest "
                f"on nothing")

    def describe(self) -> str:
        lic = self.license or "licence not recorded"
        return (f"{self.key} [{self.kind.value}] {self.display_name} "
                f"{self.version or 'version unrecorded'} -- {lic}")


class ToolRegistry:
    """Key -> :class:`ToolRegistryEntry`, consulted before any tool is invoked."""

    def __init__(self, entries: Iterable[ToolRegistryEntry] = ()) -> None:
        self._items: dict[str, ToolRegistryEntry] = {}
        for e in entries:
            self.add(e)

    def add(self, entry: ToolRegistryEntry) -> ToolRegistryEntry:
        if entry.key in self._items:
            raise ToolRegistryError(f"duplicate tool registry key: {entry.key}")
        self._items[entry.key] = entry
        return entry

    def get(self, key: str) -> ToolRegistryEntry:
        """Entry by key, or raise naming what must be registered.

        Raising rather than returning ``None`` means a typo in an adapter's
        ``registry_keys`` cannot quietly skip a licence check.
        """
        if key not in self._items:
            raise ToolRegistryError(
                f"tool '{key}' is not in the tool registry; code, model "
                f"weights, input databases and outputs must each be registered "
                f"with their own terms before the tool may be invoked "
                f"(registered: {sorted(self._items) or 'nothing'})")
        return self._items[key]

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def keys(self) -> list[str]:
        return sorted(self._items)

    def of_kind(self, kind: ToolKind) -> list[ToolRegistryEntry]:
        return [e for e in self._items.values() if e.kind is kind]

    @classmethod
    def from_config(cls, config: Any) -> "ToolRegistry":
        """Build from ``ctx.config['tool_registry']``: a list of entry dicts.

        Configuration rather than a hard-coded table of licences: licence terms
        differ between weight releases and change over time, so a table written
        from memory inside this file would be an unsourced legal claim that
        nobody re-checks. The operator records what the installation actually
        ships, with its source.
        """
        items: Iterable[Any]
        if config is None:
            items = ()
        elif isinstance(config, Mapping):
            items = config.get("tool_registry", ()) or ()
        else:
            items = config
        entries: list[ToolRegistryEntry] = []
        for raw in items:
            if isinstance(raw, ToolRegistryEntry):
                entries.append(raw)
                continue
            data = dict(raw)
            data["kind"] = ToolKind(data["kind"])
            data["citations"] = tuple(data.get("citations", ()))
            entries.append(ToolRegistryEntry(**data))
        return cls(entries)


def check_license(registry: ToolRegistry, keys: Sequence[str],
                  allow_commercial_use: bool, invoked_as: str,
                  uses_model_weights: bool = False) -> dict[str, Any]:
    """Resolve every declared entry and refuse a conflicting invocation.

    Two refusals, both before the tool runs:

    * a commercial run (``ctx.policy.allow_commercial_use``) against an entry
      that does not positively permit commercial use -- including an entry
      whose terms are unrecorded, since an unknown licence is not a permission;
    * a weights-based tool whose weights were not registered separately from
      its code, which is how a permissive code licence comes to be cited for
      non-commercial weights.

    Returns the licence facts to put in ``Provenance.parameters``, so the
    record of what was allowed travels with the result.
    """
    entries = [registry.get(k) for k in keys]
    if not entries:
        raise ToolRegistryError(
            f"{invoked_as} declares no registry keys; a tool is invoked only "
            f"after its code, weights, input databases and outputs are "
            f"registered")
    if not any(e.kind is ToolKind.CODE for e in entries):
        raise ToolRegistryError(
            f"{invoked_as} registers no entry of kind 'code'; the software "
            f"itself must be registered")
    if uses_model_weights and not any(e.kind is ToolKind.MODEL_WEIGHTS
                                      for e in entries):
        raise ToolRegistryError(
            f"{invoked_as} uses model weights but registers none of kind "
            f"'model_weights'; weights ship under different terms from code "
            f"and must be registered separately")

    if allow_commercial_use:
        for e in entries:
            if e.permits_commercial_use is not True:
                raise LicenseError(
                    f"{invoked_as}: this run sets allow_commercial_use=True, "
                    f"but {e.describe()} does not record a positive "
                    f"commercial-use permission "
                    f"(permits_commercial_use={e.permits_commercial_use}, "
                    f"source={e.license_source or 'none'}). An unverified "
                    f"licence is not a permission; record the terms or run "
                    f"with allow_commercial_use=False")

    return {
        "invoked_as": invoked_as,
        "allow_commercial_use": allow_commercial_use,
        "entries": [
            {"key": e.key, "kind": e.kind.value, "display_name": e.display_name,
             "version": e.version, "license": e.license,
             "license_source": e.license_source,
             "permits_commercial_use": e.permits_commercial_use,
             "permits_derivative_works": e.permits_derivative_works,
             "output_terms": e.output_terms,
             "needs_legal_review": e.needs_legal_review}
            for e in entries
        ],
    }


def check_modification(registry: ToolRegistry, keys: Sequence[str],
                       invoked_as: str) -> None:
    """Refuse to modify and redistribute an artefact whose terms do not say it may be.

    For a workflow that would patch, vendor or republish a third-party
    artefact -- not for running it. ``permits_derivative_works`` of ``None`` is
    refused exactly as ``False`` is: nobody having read the terms is not a
    permission, and a no-derivatives licence (CC BY-ND, CC BY-NC-ND) is the case
    this exists for.
    """
    for key in keys:
        entry = registry.get(key)
        if entry.permits_derivative_works is not True:
            raise LicenseError(
                f"{invoked_as}: {entry.describe()} does not record a positive "
                f"derivative-works permission "
                f"(permits_derivative_works={entry.permits_derivative_works}, "
                f"source={entry.license_source or 'none'}). Modifying and "
                f"redistributing it, or publishing a derivative, is refused; "
                f"running it as shipped is a separate question answered by "
                f"check_license")


# ---------------------------------------------------------------------------
# requests, policy, routes
# ---------------------------------------------------------------------------


class Route(str, enum.Enum):
    """The two complementary modelling routes.

    Complementary, not interchangeable: they share no failure mode, which is
    the only reason agreement between them carries information (see
    :func:`eagent.science.robustness.cross_method_agreement`).
    """

    TEMPLATE_DOCKING = "template_docking"
    JOINT_PREDICTION = "joint_prediction"


class ComplexPolicy(BaseModel):
    """Sampling-scope and reporting parameters, recorded in provenance."""

    model_config = ConfigDict(extra="forbid")

    box_padding_angstrom: float = Field(DEFAULT_BOX_PADDING_A, ge=0.0)
    min_box_edge_angstrom: float = Field(DEFAULT_MIN_BOX_EDGE_A, gt=0.0)
    max_box_edge_warn_angstrom: float = Field(DEFAULT_MAX_BOX_EDGE_WARN_A, gt=0.0)
    pose_cluster_rmsd_angstrom: float = Field(DEFAULT_POSE_CLUSTER_RMSD_A, gt=0.0)
    vdw_overlap_tolerance_angstrom: float = Field(
        DEFAULT_VDW_OVERLAP_TOLERANCE_A, ge=0.0)
    n_samples: int = Field(5, ge=1, description="Joint-prediction samples per "
                                                "candidate.")
    allow_subunit_deficient_assembly: bool = Field(
        False,
        description="Permit modelling into fewer chains than the template's "
                    "assembly state. Off by default: in most SDRs the pocket "
                    "is walled by the neighbouring subunit.")


class ComplexRequest(BaseModel):
    """One candidate's complex, stated as components rather than as a file.

    Ligand *codes* are required rather than inferred: the question "is the
    substrate in this pose?" has to be answerable by looking at the produced
    coordinates, and it is not answerable if nobody said what the substrate is
    called in them.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    structure_id: str
    receptor_path: str
    chain: str | None = None
    substrate_ligand_code: str = Field(
        ..., min_length=1,
        description="Chemical component id the substrate carries in the pose "
                    "files, e.g. 'LIG'.")
    substrate_conformer_paths: tuple[str, ...] = Field(
        (), description="Pre-generated 3D substrate conformers. Required for "
                        "route A: conformer generation needs a cheminformatics "
                        "toolkit this project does not ship, and docking one "
                        "arbitrary conformer samples one torsional basin.")
    cofactor_ligand_code: str | None = None
    cofactor_smiles: str | None = None
    metals: tuple[str, ...] = ()
    restraint_names: tuple[str, ...] = ()
    n_samples: int | None = None


@dataclass(frozen=True)
class ComponentRequirement:
    """One component of the catalytic system, and whether it is accounted for."""

    name: str
    kind: str                 # protein | substrate | cofactor | metal | assembly
    required: bool
    satisfied: bool
    detail: str


@dataclass
class AssemblyValidation:
    """The verdict on a requested assembly, before anything is modelled."""

    candidate_id: str
    route: Route
    components: list[ComponentRequirement] = field(default_factory=list)
    flags: list[tuple[str, Severity, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only when no required component is missing or mis-stated."""
        return not any(s is Severity.BLOCKER for _c, s, _m in self.flags)

    @property
    def missing(self) -> list[str]:
        return [c.name for c in self.components if c.required and not c.satisfied]


def validate_assembly(request: ComplexRequest, template: CatalyticTemplate,
                      substrate: SubstrateSpec, receptor: Structure,
                      route: Route,
                      policy: ComplexPolicy | None = None) -> AssemblyValidation:
    """Check the requested system against the family's catalytic template.

    The rejections here are the point of the module:

    * **no substrate structure** -- a name is not a molecule, and a docking run
      cannot be defined against one;
    * **cofactor missing** -- for route A the cofactor must already be in the
      receptor, because docking adds only the ligand it is given; a
      protein+substrate pose for a cofactor-dependent reaction is a model of a
      reaction that cannot happen;
    * **cofactor in the wrong oxidation state** -- NAD+/NADP+ (components NAD,
      NAP) cannot donate hydride; this is treated exactly as a missing
      cofactor;
    * **cofactor state not derivable** -- an unrecognised component id is not
      assumed to be the reduced form;
    * **required metal absent** -- the catalytic zinc of an MDR/ADH polarises
      the carbonyl; without it the pose is of a different mechanism;
    * **subunit-deficient assembly** -- a pocket completed by a neighbouring
      subunit is open to solvent in a monomer, and poses found in it are
      artefacts of the missing chain.
    """
    policy = policy or ComplexPolicy()
    v = AssemblyValidation(candidate_id=request.candidate_id, route=route)
    inventory = {c.upper(): n for c, n in ligand_inventory(receptor).items()}
    receptor_metals = set(metals_in(receptor))

    # -- protein ----------------------------------------------------------
    chains = [c for c in receptor.chains if c.polymer_residues()]
    v.components.append(ComponentRequirement(
        "protein", "protein", True, bool(chains),
        f"{len(chains)} polymer chain(s) in the receptor"))
    if not chains:
        v.flags.append(("receptor_has_no_protein", Severity.BLOCKER,
                        "the receptor file contains no polymer chain"))

    want_chains = expected_chain_count(template.assembly_state)
    assembly_ok = True
    if want_chains is not None and len(chains) < want_chains:
        assembly_ok = policy.allow_subunit_deficient_assembly
        message = (f"the template's mechanism needs a {template.assembly_state} "
                   f"({want_chains} chains) and the receptor has {len(chains)}; "
                   f"the pocket may be walled by a subunit that is not in the "
                   f"file, so poses found in it would be artefacts of the "
                   f"missing chain")
        v.flags.append(("assembly_state_incomplete",
                        Severity.WARN if assembly_ok else Severity.BLOCKER,
                        message))
    v.components.append(ComponentRequirement(
        "assembly", "assembly", want_chains is not None, assembly_ok,
        f"template assembly_state={template.assembly_state!r}, "
        f"expected chains={want_chains}, receptor chains={len(chains)}"))

    # -- substrate ---------------------------------------------------------
    defined = substrate.is_structurally_defined
    v.components.append(ComponentRequirement(
        "substrate", "substrate", True, defined,
        f"{substrate.name or 'substrate'} "
        f"{'has an explicit structure' if defined else 'has no SMILES or molfile'}"))
    if not defined:
        v.flags.append(("substrate_not_structurally_defined", Severity.BLOCKER,
                        "the substrate has neither isomeric SMILES nor a "
                        "molfile; a complex cannot be built from a name"))
    if route is Route.TEMPLATE_DOCKING and not request.substrate_conformer_paths:
        v.flags.append(("no_substrate_conformers", Severity.BLOCKER,
                        "route A was asked for with no 3D substrate conformers. "
                        "Conformer generation needs a cheminformatics toolkit "
                        "this project does not ship, and docking a single "
                        "arbitrary conformer samples one torsional basin; "
                        "supply substrate_conformer_paths"))

    # -- cofactor ----------------------------------------------------------
    required_cofactor = template.required_cofactor
    required_state = template.required_cofactor_state
    if required_cofactor:
        code = (request.cofactor_ligand_code or "").strip().upper()
        if not code:
            v.flags.append(("cofactor_missing_from_assembly", Severity.BLOCKER,
                            f"the template requires {required_cofactor} "
                            f"[{required_state.value}] and the request names no "
                            f"cofactor component; a protein+substrate complex "
                            f"for a cofactor-dependent reaction models a "
                            f"reaction that cannot occur"))
            v.components.append(ComponentRequirement(
                "cofactor", "cofactor", True, False, "no component id requested"))
        else:
            state = cofactor_state_from_ligand_code(code)
            allowed = {c.strip().upper()
                       for c in template.cofactor_ligand_codes}
            in_receptor = code in inventory
            detail = (f"component {code} [{state.value}], "
                      f"{'present in' if in_receptor else 'absent from'} the "
                      f"receptor")
            satisfied = True
            if allowed and code not in allowed:
                satisfied = False
                v.flags.append(("cofactor_component_not_in_template",
                                Severity.BLOCKER,
                                f"component {code} is not one of the template's "
                                f"cofactor components {sorted(allowed)}"))
            if state is CofactorState.UNKNOWN:
                satisfied = False
                v.flags.append(("cofactor_state_not_derivable", Severity.BLOCKER,
                                f"the oxidation state of component {code} is not "
                                f"derivable from its id, and a hydride-transfer "
                                f"complex will not be assembled on an assumed "
                                f"state"))
            elif required_state is not CofactorState.UNKNOWN \
                    and state is not required_state:
                satisfied = False
                v.flags.append(("cofactor_wrong_oxidation_state", Severity.BLOCKER,
                                f"the template requires {required_cofactor} "
                                f"[{required_state.value}] and the request names "
                                f"{code}, which is the {state.value} form. "
                                f"NAD(P)+ has no hydride to donate; this is a "
                                f"missing cofactor, not a close-enough one"))
            if route is Route.TEMPLATE_DOCKING and not in_receptor:
                satisfied = False
                v.flags.append(("cofactor_absent_from_receptor", Severity.BLOCKER,
                                f"route A docks the substrate into a fixed "
                                f"receptor, and component {code} is not in it "
                                f"(receptor ligands: {sorted(inventory) or 'none'}). "
                                f"Dock into a holo structure, or transplant the "
                                f"cofactor first and record it as transplanted"))
            v.components.append(ComponentRequirement(
                "cofactor", "cofactor", True, satisfied, detail))
    else:
        v.components.append(ComponentRequirement(
            "cofactor", "cofactor", False, True,
            "the catalytic template declares no cofactor requirement"))

    # -- metals ------------------------------------------------------------
    for metal in template.metals:
        m = metal.strip().upper()
        present = m in receptor_metals or m in {x.strip().upper()
                                                for x in request.metals}
        v.components.append(ComponentRequirement(
            f"metal:{m}", "metal", True, present,
            f"{'present' if present else 'absent'} "
            f"(receptor metals: {sorted(receptor_metals) or 'none'})"))
        if not present:
            v.flags.append(("required_metal_absent", Severity.BLOCKER,
                            f"the template requires {m} and neither the "
                            f"receptor nor the request provides it; without the "
                            f"metal the carbonyl is not polarised and the model "
                            f"is of a different mechanism"))

    # -- restraint names must exist in the template ------------------------
    known = {c.name for c in template.geometry_constraints}
    unknown = [n for n in request.restraint_names if n not in known]
    if unknown:
        raise TemplateError(
            f"{request.candidate_id}: restraint name(s) {unknown} are not "
            f"constraints of catalytic template '{template.template_id}' "
            f"(known: {sorted(known) or 'none'}). A restraint whose name does "
            f"not match a template constraint is invisible to the circularity "
            f"guard")
    return v


# ---------------------------------------------------------------------------
# the search box: route A only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchBox:
    """Where route A is allowed to look, and what defined it."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    anchors: tuple[str, ...]
    basis: str

    @property
    def max_edge(self) -> float:
        return max(self.size)

    def to_dict(self) -> dict[str, Any]:
        return {"center": list(self.center), "size": list(self.size),
                "anchors": list(self.anchors), "basis": self.basis}


def trusted_site_atoms(receptor: Structure, chain_id: str, rmap: ResidueMap,
                       catalytic_roles: Mapping[str, int],
                       cofactor_code: str | None) -> tuple[list[Atom], list[str], str]:
    """Atoms that define the catalytic site, with their provenance.

    "Trusted" means observed or mapped, never inferred from bulk geometry: the
    bound cofactor (observed coordinates) and the catalytic residues reached
    through the candidate's :class:`~eagent.science.numbering.ResidueMap`. The
    largest-cavity heuristic is deliberately absent -- in a two-domain MDR/ADH
    the largest cavity is the interdomain cleft, and a box centred there
    samples a site where no chemistry happens while looking entirely
    reasonable in a figure.

    Raises :class:`GeometryError` when nothing trustworthy is available, so the
    caller must report "the site could not be localised" instead of silently
    docking into the whole protein.
    """
    atoms: list[Atom] = []
    labels: list[str] = []
    bases: list[str] = []

    if cofactor_code:
        code = cofactor_code.strip().upper()
        cofactor = [r for r in receptor.ligands()
                    if r.resname.strip().upper() == code]
        for res in cofactor:
            atoms.extend(res.heavy_atoms())
            labels.append(f"{res.chain}/{res.resname.strip()}{res.resseq}")
        if cofactor:
            bases.append(f"observed cofactor {code}")

    chain = receptor.chain(chain_id)
    if chain is not None:
        for role, index in sorted(catalytic_roles.items()):
            try:
                pos = rmap.to_author(int(index))
            except NumberingError:
                continue
            if pos is None:
                continue
            res = chain.residue(pos.resseq, pos.icode)
            if res is None:
                continue
            atoms.extend(res.heavy_atoms())
            labels.append(f"{role}@{pos}")
        if labels and (not bases or len(labels) > len(bases)):
            bases.append("mapped catalytic residues")

    if not atoms:
        raise GeometryError(
            "no trusted catalytic-site atom: the receptor carries no cofactor "
            "and no catalytic residue could be mapped onto it. A box cannot be "
            "defined, and defining one from the whole protein would turn this "
            "into blind docking reported as pocket docking")
    return atoms, labels, " + ".join(bases)


def catalytic_site_box(anchor_atoms: Sequence[Atom], padding: float,
                       min_edge: float, anchors: Sequence[str],
                       basis: str) -> SearchBox:
    """Axis-aligned box enclosing the anchors, padded to leave room to sample.

    The box is the *scope of the search*, and it is recorded in provenance
    because a pose means nothing without it: the same runner, the same
    receptor and a box twice as wide produce different poses, and a reader
    comparing two candidates must be able to see whether they were searched
    the same way.
    """
    if not anchor_atoms:
        raise GeometryError("cannot build a search box from no atoms")
    xs = [a.x for a in anchor_atoms]
    ys = [a.y for a in anchor_atoms]
    zs = [a.z for a in anchor_atoms]
    center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0,
              (min(zs) + max(zs)) / 2.0)
    size = tuple(max(hi - lo + 2.0 * padding, min_edge)
                 for lo, hi in ((min(xs), max(xs)), (min(ys), max(ys)),
                                (min(zs), max(zs))))
    return SearchBox(center=center, size=(size[0], size[1], size[2]),
                     anchors=tuple(anchors), basis=basis)


# ---------------------------------------------------------------------------
# adapter seams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawPose:
    """What a runner hands back, before this module takes responsibility for it.

    ``model_confidence`` is copied verbatim into
    :attr:`~eagent.schemas.candidate.ComplexPose.model_confidence`; nothing in
    this module rescales it or folds several of its keys into one number.
    ``restrained_constraints`` is what the runner says it enforced, which is
    cross-checked against what the job asked for.
    """

    path: Path
    score: float | None = None
    score_function: str | None = None
    model_confidence: Mapping[str, float] = field(default_factory=dict)
    conformer_id: str | None = None
    sample_index: int | None = None
    restrained_constraints: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DockingJob:
    """Everything route A needs, and everything that must be reproducible."""

    candidate_id: str
    receptor_path: Path
    receptor_chain: str | None
    box: SearchBox
    conformer_paths: tuple[str, ...]
    substrate_ligand_code: str
    cofactor_ligand_code: str | None
    restraints: tuple[GeometryConstraint, ...]
    seed: int
    out_dir: Path
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def restraint_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.restraints)


@dataclass(frozen=True)
class JointPredictionJob:
    """Everything route B needs: the complete system, as components."""

    candidate_id: str
    sequence: str
    substrate_smiles: str | None
    substrate_ligand_code: str
    cofactor_ligand_code: str | None
    cofactor_smiles: str | None
    metals: tuple[str, ...]
    n_samples: int
    seed: int
    out_dir: Path
    restraints: tuple[GeometryConstraint, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def restraint_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.restraints)


@runtime_checkable
class DockingRunner(Protocol):
    """Adapter seam for template-guided pocket docking (route A)."""

    name: str
    version: str
    registry_keys: tuple[str, ...]
    uses_model_weights: bool

    def is_available(self) -> bool: ...

    def dock(self, job: DockingJob) -> Sequence[RawPose]: ...


@runtime_checkable
class ComplexPredictor(Protocol):
    """Adapter seam for joint protein-ligand-cofactor prediction (route B)."""

    name: str
    version: str
    registry_keys: tuple[str, ...]
    uses_model_weights: bool
    runs_remotely: bool

    def is_available(self) -> bool: ...

    def predict(self, job: JointPredictionJob) -> Sequence[RawPose]: ...


@dataclass(frozen=True)
class UnavailableDockingRunner:
    """Default route A adapter: absent, and loudly so.

    No built-in fallback docking exists here on purpose. A hand-rolled scoring
    function would produce poses indistinguishable in the artifacts from a
    real docking run, with none of the sampling.
    """

    name: str = "docking_runner"
    version: str = "absent"
    registry_keys: tuple[str, ...] = ()
    uses_model_weights: bool = False
    install_hint: str = ("install a docking engine (e.g. AutoDock Vina, "
                         "smina, gnina) and pass it as docking_runner=")

    def is_available(self) -> bool:
        return False

    def dock(self, job: DockingJob) -> Sequence[RawPose]:
        raise ToolUnavailableError(self.name, self.install_hint)


@dataclass(frozen=True)
class UnavailableComplexPredictor:
    """Default route B adapter: absent, and loudly so."""

    name: str = "complex_predictor"
    version: str = "absent"
    registry_keys: tuple[str, ...] = ()
    uses_model_weights: bool = True
    runs_remotely: bool = False
    install_hint: str = ("install a co-folding predictor (e.g. Boltz, Chai, "
                         "AlphaFold3 under its own licence) and pass it as "
                         "complex_predictor=")

    def is_available(self) -> bool:
        return False

    def predict(self, job: JointPredictionJob) -> Sequence[RawPose]:
        raise ToolUnavailableError(self.name, self.install_hint)


# ---------------------------------------------------------------------------
# pose comparison
# ---------------------------------------------------------------------------


def _ligand_atoms(structure: Structure, code: str) -> list[Atom]:
    """Heavy atoms of every copy of a chemical component, in file order."""
    want = code.strip().upper()
    return [a for r in structure.ligands() if r.resname.strip().upper() == want
            for a in r.heavy_atoms()]


def _atom_key(a: Atom) -> tuple[str, int, str, str]:
    return (a.resname.strip().upper(), a.resseq, a.icode.strip(),
            a.name.strip().upper())


def pose_rmsd(atoms_a: Sequence[Atom], atoms_b: Sequence[Atom]) -> float | None:
    """Heavy-atom RMSD between two poses of the same ligand, or ``None``.

    Atoms are paired by (component, residue number, insertion code, atom name),
    never by file order: two runs of the same docking engine can emit atoms in
    different orders, and an order-based pairing silently computes the RMSD
    between unrelated atoms, which reads as "a different binding mode".

    ``None`` when the two atom sets do not correspond -- a different number of
    copies, a renamed atom. Returning a large number instead would report a
    naming problem as a conformational difference.

    No superposition is applied: both poses live in the same receptor frame, so
    superposing them would remove exactly the displacement being measured.
    """
    if not atoms_a or len(atoms_a) != len(atoms_b):
        return None
    by_key_b: dict[tuple[str, int, str, str], Atom] = {}
    for a in atoms_b:
        by_key_b.setdefault(_atom_key(a), a)
    total = 0.0
    n = 0
    for a in atoms_a:
        partner = by_key_b.get(_atom_key(a))
        if partner is None:
            return None
        total += ((a.x - partner.x) ** 2 + (a.y - partner.y) ** 2
                  + (a.z - partner.z) ** 2)
        n += 1
    if n == 0:
        return None
    return math.sqrt(total / n)


def cluster_poses(pose_atoms: Mapping[str, Sequence[Atom]],
                  cutoff_angstrom: float) -> dict[str, int]:
    """Group poses into binding modes by RMSD. Nothing is discarded.

    Single-linkage over the pose ids in the order given, so the grouping is
    deterministic. The result is *reporting* structure: it distinguishes "six
    distinct binding modes" from "six copies of one", which is the difference
    between a sampled result and a converged artefact. A pose whose RMSD to
    everything is unmeasurable gets a cluster of its own, because an
    unmeasurable comparison is not a match.
    """
    clusters: dict[str, int] = {}
    representatives: list[tuple[int, str]] = []
    for pose_id, atoms in pose_atoms.items():
        placed = False
        for cid, rep_id in representatives:
            d = pose_rmsd(atoms, pose_atoms[rep_id])
            if d is not None and d <= cutoff_angstrom:
                clusters[pose_id] = cid
                placed = True
                break
        if not placed:
            cid = len(representatives)
            representatives.append((cid, pose_id))
            clusters[pose_id] = cid
    return clusters


def assert_restraints_recorded(poses: Sequence[ComplexPose],
                               expected: Sequence[str]) -> None:
    """Refuse to emit a restrained pose that does not say it was restrained.

    The circularity guard downstream reads
    ``ComplexPose.restrained_constraints`` and nothing else. A pose built under
    restraints but emitted without them would have its enforced distances
    counted as independent corroboration, and no artifact further down the
    pipeline could tell. Hence an exception rather than a warning.
    """
    want = set(expected)
    if not want:
        return
    for pose in poses:
        missing = want - set(pose.restrained_constraints)
        if missing:
            raise CircularEvidenceError(
                f"pose {pose.pose_id} was built under restraint(s) "
                f"{sorted(missing)} but does not record them; the circularity "
                f"guard would count the enforced geometry as independent "
                f"evidence")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

POSE_INDEX_COLUMNS: tuple[str, ...] = (
    "pose_id", "candidate_id", "structure_id", "route", "method", "rank",
    "cluster_id", "conformer_id", "sample_index", "path", "sha256",
    "substrate_present", "substrate_code", "substrate_source",
    "cofactor_present", "cofactor_code", "cofactor_state", "cofactor_source",
    "metals_present", "docking_score", "docking_score_function",
    "ranking_score", "iptm", "pae_interface", "other_confidence",
    "restrained_constraints", "clash_count", "clash_unscreened_atoms",
    "is_valid", "invalid_reason",
)


def _cell(value: Any) -> str:
    """One TSV cell. Empty means not measured, which is not the same as zero."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ";".join(_cell(v) for v in value)
    if isinstance(value, Mapping):
        return ";".join(f"{k}={_cell(v)}" for k, v in sorted(value.items()))
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _write_tsv(path: Path, header: Sequence[str],
               rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(_cell(row.get(c)) for c in header) + "\n")
    return path


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------


@dataclass
class _PoseBundle:
    """A finished pose plus the bookkeeping that does not fit in the schema."""

    pose: ComplexPose
    candidate_id: str
    structure_id: str
    route: Route
    atoms: list[Atom]
    unscreened_atoms: int
    other_confidence: dict[str, Any]
    sha256: str
    substrate_code: str
    cofactor_code: str | None = None
    conformer_id: str | None = None
    sample_index: int | None = None


class ModelComplexes(ScientificInterface):
    """Assemble enzyme-substrate-cofactor complexes by two independent routes."""

    name = "model_complexes"
    description = ("Build and index complete catalytic complexes (protein + "
                   "substrate + correctly-reduced cofactor + metals) by "
                   "template-guided docking and by joint complex prediction")
    required_fields: tuple[str, ...] = ("reaction.substrate.isomeric_smiles",)
    required_approvals: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ("prepare_structures",)
    version = "0.1.0"

    def execute(
        self,
        ctx: RunContext,
        *,
        candidates: Sequence[Candidate] | Sequence[Mapping[str, Any]]
        | Mapping[str, Any] | None = None,
        structures: Mapping[str, StructureRecord] | None = None,
        requests: Sequence[ComplexRequest] | None = None,
        catalytic_templates: Mapping[str, CatalyticTemplate]
        | Sequence[CatalyticTemplate] | None = None,
        substrate: SubstrateSpec | None = None,
        routes: Sequence[Route | str] = (Route.TEMPLATE_DOCKING,),
        docking_runner: DockingRunner | None = None,
        complex_predictor: ComplexPredictor | None = None,
        policy: ComplexPolicy | None = None,
        tool_registry: ToolRegistry | None = None,
        access_policy: AccessPolicy | None = None,
        **_: Any,
    ) -> ToolResult:
        """Validate each requested assembly, then model it by every route asked for.

        ``candidates`` may arrive as models or as the serialised mapping the
        previous step published. :func:`~eagent.tools.handoff.as_candidates`
        resolves that once, here, rather than letting a mapping reach
        ``cand.catalytic_mapping`` and fail as a missing attribute with no hint
        of which step produced the bad payload.
        """
        policy = policy or ComplexPolicy()
        candidates = as_candidates(candidates, source=self.name)
        substrate = substrate or ctx.task.reaction.substrate
        registry = tool_registry or ToolRegistry.from_config(ctx.config)
        wanted_routes = [Route(r) for r in routes]
        if not wanted_routes:
            return ToolResult.failure(self.name, "no modelling route requested",
                                      code="missing_input")
        if not candidates:
            return ToolResult.failure(
                self.name,
                "no candidates were supplied; model_complexes does not invent "
                "the proteins to model",
                code="missing_input")
        by_id = {c.candidate_id: c for c in candidates}
        request_list = list(requests or [])
        if not request_list:
            return ToolResult.failure(
                self.name,
                "no ComplexRequest was supplied. The substrate's chemical "
                "component id, its 3D conformers and the cofactor component "
                "cannot be inferred from the task; state them per candidate",
                code="missing_input")

        templates = _template_lookup(catalytic_templates)
        result = ToolResult(status=Status.SUCCESS)
        complexes_dir = ctx.dir(self.name, "complexes")

        bundles: list[_PoseBundle] = []
        validations: list[AssemblyValidation] = []
        rejected: list[dict[str, Any]] = []
        route_failures: list[EAgentError] = []
        licenses: list[dict[str, Any]] = []
        disclosure_notes: list[str] = []
        per_candidate_poses: dict[str, list[str]] = {}

        for request in request_list:
            cand = by_id.get(request.candidate_id)
            if cand is None:
                rejected.append({"candidate_id": request.candidate_id,
                                 "reason": "no candidate with this id was supplied"})
                continue
            template = _template_for(cand, templates)
            if template is None:
                result.add_flag(
                    "no_catalytic_template", Severity.BLOCKER,
                    f"{cand.candidate_id}: no catalytic template resolved, so "
                    f"the required cofactor, its oxidation state, the metals "
                    f"and the assembly state are unknown. The complex was not "
                    f"built; a complex assembled against an unknown mechanism "
                    f"is not a hypothesis, it is a picture",
                    subject=cand.candidate_id)
                rejected.append({"candidate_id": cand.candidate_id,
                                 "reason": "no catalytic template"})
                continue

            try:
                receptor = read_structure(request.receptor_path,
                                          structure_id=request.structure_id)
            except (StructureParseError, OSError) as exc:
                result.add_flag("receptor_unreadable", Severity.BLOCKER,
                                f"{request.structure_id}: {exc}",
                                subject=cand.candidate_id)
                rejected.append({"candidate_id": cand.candidate_id,
                                 "reason": f"receptor unreadable: {exc}"})
                continue

            for route in wanted_routes:
                validation = validate_assembly(request, template, substrate,
                                               receptor, route, policy)
                validations.append(validation)
                for code, severity, message in validation.flags:
                    result.add_flag(code, severity, message,
                                    subject=f"{cand.candidate_id}/{route.value}")
                if not validation.ok:
                    rejected.append({
                        "candidate_id": cand.candidate_id,
                        "route": route.value,
                        "reason": "incomplete catalytic system: "
                                  + ", ".join(validation.missing),
                    })
                    continue

                try:
                    made, lic = self._run_route(
                        ctx, route, cand, request, template, substrate, receptor,
                        policy, registry, docking_runner, complex_predictor,
                        complexes_dir, access_policy, disclosure_notes, result)
                except (ToolUnavailableError, LicenseError, ToolRegistryError) as exc:
                    route_failures.append(exc)
                    result.add_uncertainty(
                        "route_unavailable",
                        f"Route {route.value} produced nothing for "
                        f"{cand.candidate_id}: {exc}",
                        affects=[cand.candidate_id],
                        resolvable_by=getattr(exc, "hint", "")
                        or "install or license the tool",
                    )
                    continue
                except (GeometryError, StructureSourceError, StructureParseError) as exc:
                    result.add_flag("route_failed", Severity.BLOCKER,
                                    f"{cand.candidate_id}/{route.value}: {exc}",
                                    subject=cand.candidate_id)
                    rejected.append({"candidate_id": cand.candidate_id,
                                     "route": route.value, "reason": str(exc)})
                    continue

                if lic is not None:
                    licenses.append(lic)
                bundles.extend(made)
                per_candidate_poses.setdefault(cand.candidate_id, []).extend(
                    b.pose.pose_id for b in made)
                if len(made) == 1:
                    result.add_flag(
                        "single_pose_returned", Severity.WARN,
                        f"{cand.candidate_id}/{route.value}: the runner returned "
                        f"one pose. A single pose is not a sampling result: pose "
                        f"robustness and the stereochemical call both need the "
                        f"distribution, not the winner",
                        subject=cand.candidate_id)

        if not bundles and route_failures:
            # Nothing was produced and the reason was a missing or unlicensed
            # tool: surface the typed error so the envelope carries its code.
            raise route_failures[0]

        clusters = self._cluster(bundles, policy)
        index_path, notes_path = self._write_outputs(ctx, complexes_dir, bundles,
                                                     clusters)
        self._finalise(ctx, result, candidates, request_list, bundles,
                       validations, rejected, policy, registry, licenses,
                       wanted_routes, clusters, per_candidate_poses,
                       disclosure_notes, index_path, notes_path, complexes_dir)
        return result

    # -- routes ------------------------------------------------------------
    def _run_route(self, ctx: RunContext, route: Route, cand: Candidate,
                   request: ComplexRequest, template: CatalyticTemplate,
                   substrate: SubstrateSpec, receptor: Structure,
                   policy: ComplexPolicy, registry: ToolRegistry,
                   docking_runner: DockingRunner | None,
                   complex_predictor: ComplexPredictor | None,
                   complexes_dir: Path, access_policy: AccessPolicy | None,
                   disclosure_notes: list[str],
                   result: ToolResult) -> tuple[list[_PoseBundle], dict[str, Any] | None]:
        """Dispatch to route A or B, after the availability and licence checks."""
        restraints = tuple(c for c in template.geometry_constraints
                           if c.name in set(request.restraint_names))
        seed = ctx.seed_for(f"{self.name}:{cand.candidate_id}:{route.value}")
        out_dir = ctx.dir(self.name, "runs", cand.candidate_id, route.value)

        if route is Route.TEMPLATE_DOCKING:
            runner = docking_runner or UnavailableDockingRunner()
            if not runner.is_available():
                raise ToolUnavailableError(
                    runner.name, getattr(runner, "install_hint",
                                         "no docking engine configured"))
            if not ctx.policy.allow_external_binaries:
                raise ToolUnavailableError(
                    runner.name,
                    "ctx.policy.allow_external_binaries is False, so no "
                    "external docking binary was run")
            lic = check_license(registry, list(runner.registry_keys),
                                ctx.policy.allow_commercial_use,
                                f"docking runner '{runner.name}'",
                                uses_model_weights=runner.uses_model_weights)
            chain_id = request.chain or self._chain_for(receptor, cand, request)
            rmap = self._residue_map(receptor, cand, chain_id)
            anchors, labels, basis = trusted_site_atoms(
                receptor, chain_id, rmap,
                cand.catalytic_mapping.role_to_index,
                request.cofactor_ligand_code)
            box = catalytic_site_box(anchors, policy.box_padding_angstrom,
                                     policy.min_box_edge_angstrom, labels, basis)
            if box.max_edge > policy.max_box_edge_warn_angstrom:
                result.add_flag(
                    "search_box_large", Severity.WARN,
                    f"{cand.candidate_id}: the box edge reaches "
                    f"{box.max_edge:.1f} A, which is approaching blind docking; "
                    f"poses far from the trusted site are not pocket poses",
                    subject=cand.candidate_id)
            job = DockingJob(
                candidate_id=cand.candidate_id,
                receptor_path=Path(request.receptor_path),
                receptor_chain=chain_id, box=box,
                conformer_paths=tuple(request.substrate_conformer_paths),
                substrate_ligand_code=request.substrate_ligand_code,
                cofactor_ligand_code=request.cofactor_ligand_code,
                restraints=restraints, seed=seed, out_dir=out_dir,
                params={"n_conformers": len(request.substrate_conformer_paths)},
            )
            raw = list(runner.dock(job))
            bundles = self._collect(ctx, raw, route, cand, request, template,
                                    receptor, policy, complexes_dir,
                                    job.restraint_names,
                                    method="template_docking",
                                    runner_name=runner.name,
                                    runner_version=runner.version,
                                    result=result)
            return bundles, lic

        predictor = complex_predictor or UnavailableComplexPredictor()
        if not predictor.is_available():
            raise ToolUnavailableError(
                predictor.name, getattr(predictor, "install_hint",
                                        "no complex predictor configured"))
        if not ctx.policy.allow_gpu_models:
            raise ToolUnavailableError(
                predictor.name,
                "ctx.policy.allow_gpu_models is False, so no co-folding model "
                "was run")
        lic = check_license(registry, list(predictor.registry_keys),
                            ctx.policy.allow_commercial_use,
                            f"complex predictor '{predictor.name}'",
                            uses_model_weights=predictor.uses_model_weights)
        disclosure_notes.extend(
            f"{cand.candidate_id}: {n}" for n in
            guard_sequence_submission(predictor, cand.sequence, ctx, access_policy)
        )
        job = JointPredictionJob(
            candidate_id=cand.candidate_id, sequence=cand.sequence,
            substrate_smiles=substrate.isomeric_smiles,
            substrate_ligand_code=request.substrate_ligand_code,
            cofactor_ligand_code=request.cofactor_ligand_code,
            cofactor_smiles=request.cofactor_smiles,
            metals=tuple(request.metals or tuple(template.metals)),
            n_samples=request.n_samples or policy.n_samples,
            seed=seed, out_dir=out_dir, restraints=restraints,
        )
        raw = list(predictor.predict(job))
        bundles = self._collect(ctx, raw, route, cand, request, template,
                                receptor, policy, complexes_dir,
                                job.restraint_names,
                                method=predictor.name,
                                runner_name=predictor.name,
                                runner_version=predictor.version,
                                result=result)
        return bundles, lic

    # -- pose handling -----------------------------------------------------
    def _collect(self, ctx: RunContext, raw: Sequence[RawPose], route: Route,
                 cand: Candidate, request: ComplexRequest,
                 template: CatalyticTemplate, receptor: Structure,
                 policy: ComplexPolicy, complexes_dir: Path,
                 restraint_names: Sequence[str], method: str,
                 runner_name: str, runner_version: str,
                 result: ToolResult) -> list[_PoseBundle]:
        """Turn raw runner output into indexed, verified poses. Keep them all."""
        bundles: list[_PoseBundle] = []
        for n, item in enumerate(raw, start=1):
            pose_id = (f"{cand.candidate_id}__{request.structure_id}__"
                       f"{route.value}__{n:03d}")
            bundles.append(self._finalise_pose(
                item, pose_id, n, route, cand, request, template, receptor,
                policy, complexes_dir, restraint_names, method, runner_name,
                runner_version, result))
        assert_restraints_recorded([b.pose for b in bundles], restraint_names)
        return bundles

    def _finalise_pose(self, raw: RawPose, pose_id: str, rank: int, route: Route,
                       cand: Candidate, request: ComplexRequest,
                       template: CatalyticTemplate, receptor: Structure,
                       policy: ComplexPolicy, complexes_dir: Path,
                       restraint_names: Sequence[str], method: str,
                       runner_name: str, runner_version: str,
                       result: ToolResult) -> _PoseBundle:
        """Verify one pose contains the system it was supposed to contain.

        A pose that is missing the substrate or the cofactor, or that carries
        the oxidised cofactor, is marked invalid *and kept*: deleting it would
        hide the fact that the runner produced it, and the count of invalid
        poses is itself a result about the modelling run.
        """
        invalid: list[str] = []
        atoms: list[Atom] = []
        substrate_present = cofactor_present = False
        cofactor_state = CofactorState.UNKNOWN
        metals: list[str] = []
        clash_count: int | None = None
        unscreened = 0
        stored_path: Path | None = None
        sha = ""

        try:
            pose_structure = read_structure(raw.path, structure_id=pose_id)
        except (StructureParseError, OSError) as exc:
            invalid.append(f"pose file could not be read: {exc}")
            pose_structure = None

        if pose_structure is not None:
            inventory = {c.upper(): v
                         for c, v in ligand_inventory(pose_structure).items()}
            sub_code = request.substrate_ligand_code.strip().upper()
            atoms = _ligand_atoms(pose_structure, sub_code)
            substrate_present = bool(atoms)
            if not substrate_present:
                invalid.append(
                    f"the substrate component {sub_code} is absent from the "
                    f"produced coordinates (present: {sorted(inventory) or 'none'})")

            cof_code = (request.cofactor_ligand_code or "").strip().upper()
            if template.required_cofactor:
                cofactor_present = bool(cof_code) and cof_code in inventory
                cofactor_state = cofactor_state_from_ligand_code(cof_code or None)
                if not cofactor_present:
                    invalid.append(
                        f"the catalytic template requires "
                        f"{template.required_cofactor} and component "
                        f"{cof_code or '(none requested)'} is absent from the "
                        f"produced coordinates: this is a protein+substrate "
                        f"model of a cofactor-dependent reaction")
                elif template.required_cofactor_state is not CofactorState.UNKNOWN \
                        and cofactor_state is not template.required_cofactor_state:
                    invalid.append(
                        f"the pose carries {cof_code} [{cofactor_state.value}] "
                        f"where the mechanism needs "
                        f"[{template.required_cofactor_state.value}]")
            else:
                cofactor_state = CofactorState.NOT_APPLICABLE

            metals = metals_in(pose_structure)
            missing_metals = [m.strip().upper() for m in template.metals
                              if m.strip().upper() not in set(metals)]
            if missing_metals:
                invalid.append(f"required metal(s) {missing_metals} absent from "
                               f"the produced coordinates")

            if atoms:
                clash_count, unscreened = self._clashes(pose_structure, atoms,
                                                        policy)
            stored_path = complexes_dir / f"{pose_id}.cif"
            stored_path.write_text(write_mmcif_text(pose_structure, pose_id),
                                   encoding="utf-8")
            sha = sha256_file(stored_path)

        restrained = tuple(sorted(set(restraint_names)
                                  | set(raw.restrained_constraints)))
        extra_restraints = set(raw.restrained_constraints) - set(restraint_names)
        if extra_restraints:
            result.add_flag(
                "undeclared_restraints", Severity.WARN,
                f"{pose_id}: the runner reports restraint(s) "
                f"{sorted(extra_restraints)} that the job did not ask for; they "
                f"are recorded so the circularity guard can see them, but a "
                f"restraint the pipeline did not request may also be one it "
                f"cannot name",
                subject=cand.candidate_id)

        confidence, other = _split_confidence(raw.model_confidence)
        pose = ComplexPose(
            pose_id=pose_id,
            method=method,
            rank=rank,
            path=str(stored_path) if stored_path else str(raw.path),
            substrate_present=substrate_present,
            cofactor_present=cofactor_present,
            cofactor_state=cofactor_state,
            substrate_source=(LigandSource.DOCKING_PREDICTED
                              if route is Route.TEMPLATE_DOCKING
                              else LigandSource.JOINT_STRUCTURE_PREDICTION),
            cofactor_source=self._cofactor_source(route, request, receptor),
            metals_present=metals,
            docking_score=raw.score,
            docking_score_function=raw.score_function,
            model_confidence=confidence,
            restrained_constraints=list(restrained),
            clash_count=clash_count,
            is_valid=not invalid,
            invalid_reason="; ".join(invalid) or None,
        )
        if invalid:
            result.add_flag(
                "incomplete_catalytic_system", Severity.BLOCKER,
                f"{pose_id}: {pose.invalid_reason}", subject=cand.candidate_id)
        return _PoseBundle(
            pose=pose, candidate_id=cand.candidate_id,
            structure_id=request.structure_id, route=route, atoms=atoms,
            unscreened_atoms=unscreened, other_confidence=other, sha256=sha,
            substrate_code=request.substrate_ligand_code.strip().upper(),
            cofactor_code=(request.cofactor_ligand_code or "").strip().upper()
            or None,
            conformer_id=raw.conformer_id, sample_index=raw.sample_index,
        )

    def _clashes(self, pose_structure: Structure, ligand_atoms: Sequence[Atom],
                 policy: ComplexPolicy) -> tuple[int, int]:
        """Hard-sphere overlaps between the substrate and everything else.

        A localisation screen, not an energy, and reported with the count of
        atoms that could *not* be screened: a zero from an unscreened set would
        read as "no clashes" when it means "nothing was tested". Metal-to-
        ligand pairs are excluded, because a coordinate bond is expected to sit
        well inside the van der Waals sum and would otherwise be counted as a
        clash in every zinc-dependent candidate.
        """
        ligand_ids = {id(a) for a in ligand_atoms}
        environment = [a for a in pose_structure.atoms() if id(a) not in ligand_ids]

        def not_a_coordinate_bond(protein: Atom, ligand: Atom) -> bool:
            return protein.element.strip().upper() not in METAL_ELEMENTS

        pairs, unscreened = clash_pairs(
            environment, list(ligand_atoms),
            policy.vdw_overlap_tolerance_angstrom,
            pair_filter=not_a_coordinate_bond)
        return len(pairs), len(unscreened)

    def _cofactor_source(self, route: Route, request: ComplexRequest,
                         receptor: Structure) -> LigandSource:
        """Where the cofactor's coordinates came from, which is not where the
        substrate's came from.

        Collapsing the two is how "the cofactor was observed in a 1.6 A
        structure" and "the cofactor was placed by the same model that placed
        the substrate" come to carry the same authority in a report.
        """
        if not request.cofactor_ligand_code:
            return LigandSource.UNKNOWN
        code = request.cofactor_ligand_code.strip().upper()
        in_receptor = code in {c.upper() for c in ligand_inventory(receptor)}
        if route is Route.TEMPLATE_DOCKING:
            return (LigandSource.EXPERIMENTAL_OBSERVED if in_receptor
                    else LigandSource.DOCKING_PREDICTED)
        return LigandSource.JOINT_STRUCTURE_PREDICTION

    def _cluster(self, bundles: Sequence[_PoseBundle],
                 policy: ComplexPolicy) -> dict[str, str]:
        """Cluster poses within each candidate/route, never across them.

        The id is namespaced with the group it was numbered in.
        :func:`cluster_poses` restarts at 0 for every group, so merging the
        per-group results on the bare integer collapsed candidate A's cluster 0
        into candidate B's. The run then reported fewer binding modes than were
        sampled -- understating diversity and overstating convergence, which is
        the reading the cluster count exists to support.
        """
        clusters: dict[str, str] = {}
        groups: dict[tuple[str, str], dict[str, Sequence[Atom]]] = {}
        for b in bundles:
            groups.setdefault((b.candidate_id, b.route.value), {})[
                b.pose.pose_id] = b.atoms
        for (candidate_id, route), poses in groups.items():
            local = cluster_poses(poses, policy.pose_cluster_rmsd_angstrom)
            for pose_id, cid in local.items():
                clusters[pose_id] = f"{candidate_id}:{route}:{cid}"
        return clusters

    @staticmethod
    def _clusters_per_candidate(
        bundles: Sequence[_PoseBundle], clusters: Mapping[str, str]
    ) -> dict[str, int]:
        """Distinct binding modes per candidate, which is where it is read.

        A run-wide count hides the case the comparison needs: two candidates
        with one mode each is convergence, one candidate with two modes is not.
        """
        per: dict[str, set[str]] = {}
        for b in bundles:
            cid = clusters.get(b.pose.pose_id)
            if cid is not None:
                per.setdefault(b.candidate_id, set()).add(cid)
        return {c: len(v) for c, v in sorted(per.items())}

    # -- outputs -----------------------------------------------------------
    def _chain_for(self, receptor: Structure, cand: Candidate,
                   request: ComplexRequest) -> str:
        """Chain to measure and map against, chosen by alignment when unstated."""
        _chain, _rmap, choice = select_chain(receptor, cand.sequence, None)
        return choice.chain_id

    def _residue_map(self, receptor: Structure, cand: Candidate,
                     chain_id: str) -> ResidueMap:
        """Rebuild the candidate <-> author-numbering map for this receptor."""
        _chain, rmap, _choice = select_chain(receptor, cand.sequence, chain_id)
        return rmap

    def _write_outputs(self, ctx: RunContext, complexes_dir: Path,
                       bundles: Sequence[_PoseBundle],
                       clusters: Mapping[str, str]) -> tuple[Path, Path]:
        rows = []
        for b in bundles:
            p = b.pose
            rows.append({
                "pose_id": p.pose_id,
                "candidate_id": b.candidate_id,
                "structure_id": b.structure_id,
                "route": b.route.value,
                "method": p.method,
                "rank": p.rank,
                "cluster_id": clusters.get(p.pose_id),
                "conformer_id": b.conformer_id,
                "sample_index": b.sample_index,
                "path": p.path,
                "sha256": b.sha256 or None,
                "substrate_present": p.substrate_present,
                "substrate_code": b.substrate_code,
                "substrate_source": p.substrate_source.value,
                "cofactor_present": p.cofactor_present,
                "cofactor_code": b.cofactor_code,
                "cofactor_state": p.cofactor_state.value,
                "cofactor_source": p.cofactor_source.value,
                "metals_present": p.metals_present,
                "docking_score": p.docking_score,
                "docking_score_function": p.docking_score_function,
                "ranking_score": p.model_confidence.get("ranking_score"),
                "iptm": p.model_confidence.get("iptm"),
                "pae_interface": p.model_confidence.get("pae_interface"),
                "other_confidence": {k: v for k, v in p.model_confidence.items()
                                     if k not in ("ranking_score", "iptm",
                                                  "pae_interface")}
                or b.other_confidence or None,
                "restrained_constraints": p.restrained_constraints,
                "clash_count": p.clash_count,
                "clash_unscreened_atoms": b.unscreened_atoms,
                "is_valid": p.is_valid,
                "invalid_reason": p.invalid_reason,
            })
        index_path = _write_tsv(complexes_dir / "pose_index.tsv",
                                POSE_INDEX_COLUMNS, rows)
        notes_path = complexes_dir / "POSE_INDEX_NOTES.md"
        notes_path.write_text(
            "# How to read pose_index.tsv\n\n"
            f"{RANKING_SCORE_CAVEAT}\n\n"
            "* `restrained_constraints` lists the geometry that was *enforced* "
            "while the pose was built. Those constraints are self-fulfilling "
            "and are excluded from independent evidence by "
            "`eagent.science.robustness.CircularityGuard`.\n"
            "* `cluster_id` groups poses by heavy-atom RMSD for reporting. No "
            "pose is removed by clustering; several poses in one cluster mean "
            "the sampling converged, not that the mode is correct. It reads "
            "`<candidate_id>:<route>:<local id>` because poses are only ever "
            "compared inside one candidate and one route, so the local number "
            "repeats across groups and is not a run-wide identity.\n"
            "* `is_valid=false` poses are kept deliberately. A runner that "
            "returns complexes missing the cofactor is a fact about the run.\n"
            "* `clash_count` is a hard-sphere screen at a stated tolerance, not "
            "an energy, and is meaningless without `clash_unscreened_atoms`.\n",
            encoding="utf-8")
        return index_path, notes_path

    def _finalise(self, ctx: RunContext, result: ToolResult,
                  candidates: Sequence[Candidate],
                  requests: Sequence[ComplexRequest],
                  bundles: Sequence[_PoseBundle],
                  validations: Sequence[AssemblyValidation],
                  rejected: Sequence[Mapping[str, Any]], policy: ComplexPolicy,
                  registry: ToolRegistry, licenses: Sequence[Mapping[str, Any]],
                  routes: Sequence[Route], clusters: Mapping[str, str],
                  per_candidate: Mapping[str, Sequence[str]],
                  disclosure_notes: Sequence[str], index_path: Path,
                  notes_path: Path, complexes_dir: Path) -> None:
        """Artifacts, status, provenance and the caveats that travel with them."""
        valid = [b for b in bundles if b.pose.is_valid]
        per_candidate_clusters = self._clusters_per_candidate(bundles, clusters)
        result.artifacts.append(Artifact(
            key="complexes_dir", path=str(complexes_dir), kind="object",
            n_records=len(bundles),
            summary="one mmCIF per pose; every pose the runners returned"))
        result.artifacts.append(Artifact(
            key="pose_index", path=str(index_path), kind="table",
            sha256=sha256_file(index_path), n_records=len(bundles),
            summary=("one row per pose: route, cluster, confidence recorded "
                     "verbatim, enforced restraints, clash screen, validity")))
        result.artifacts.append(Artifact(
            key="pose_index_notes", path=str(notes_path), kind="file",
            sha256=sha256_file(notes_path),
            summary="how to read the index, including the ranking-score caveat"))

        n_requested = len(requests)
        n_with_poses = len([c for c, ids in per_candidate.items() if ids])
        if not valid:
            result.status = Status.FAILED
            result.message = (
                f"no valid complex was produced for any of {n_requested} "
                f"request(s); {len(rejected)} were rejected before modelling "
                f"and {len(bundles) - len(valid)} produced poses missing part of "
                f"the catalytic system")
            # Without a blocking code this failure is unclassifiable and the
            # run stops as "unclassified" instead of being routed back to the
            # assembly inputs that are actually missing.
            result.add_flag("no_valid_complex", Severity.BLOCKER,
                            result.message, subject="complexes")
        elif rejected or len(valid) < len(bundles) or n_with_poses < n_requested:
            result.status = Status.PARTIAL
            result.message = (
                f"{len(valid)}/{len(bundles)} pose(s) contain the complete "
                f"catalytic system, covering {n_with_poses}/{n_requested} "
                f"request(s)")
        else:
            result.message = (f"{len(valid)} pose(s) across {n_with_poses} "
                              f"candidate(s) and {len(routes)} route(s)")

        if len(routes) == 1:
            result.add_uncertainty(
                "single_modelling_route",
                f"Only route {routes[0].value} was run, so there is no "
                f"cross-method agreement to check. Two routes with unrelated "
                f"artefacts are what makes agreement informative.",
                affects=sorted(per_candidate),
                resolvable_by="run the other route and compare")
        if valid:
            result.add_next(
                "screen_geometry",
                "Measure the catalytic template's constraints on every pose, "
                "excluding the restrained ones from independent evidence",
                {"pose_index": str(index_path),
                 "restrained": sorted({n for b in bundles
                                       for n in b.pose.restrained_constraints})})

        result.provenance = Provenance(
            tool=self.name, tool_version=self.version,
            inputs_sha256={
                "requests": sha256_obj([r.model_dump(mode="json")
                                        for r in requests]),
                "receptors": sha256_obj(sorted({r.receptor_path
                                                for r in requests})),
            },
            databases={e["key"]: (e["version"] or "unrecorded")
                       for lic in licenses for e in lic["entries"]
                       if e["kind"] == "input_database"},
            models={e["key"]: (e["version"] or "unrecorded")
                    for lic in licenses for e in lic["entries"]
                    if e["kind"] == "model_weights"},
            parameters={
                "routes": [r.value for r in routes],
                "policy": policy.model_dump(mode="json"),
                "licenses": list(licenses),
                "registered_tools": registry.keys(),
                "n_poses": len(bundles),
                "n_valid_poses": len(valid),
                "n_clusters": len(set(clusters.values())),
                "n_clusters_by_candidate": per_candidate_clusters,
                "cluster_id_basis": ("<candidate_id>:<route>:<local id>; "
                                     "clustering never crosses a candidate or "
                                     "a route, so the local id alone is not "
                                     "unique across the run"),
                "restrained_constraints": sorted(
                    {n for b in bundles for n in b.pose.restrained_constraints}),
                "disclosure_notes": list(disclosure_notes),
                "ranking_score_caveat": RANKING_SCORE_CAVEAT,
                "pose_retention": ("every pose returned by a runner is written "
                                   "out and indexed; clustering groups, it does "
                                   "not filter"),
            },
            random_seed=ctx.seed_for(self.name),
        )

        result.data.update({
            CANDIDATES_KEY: serialise_candidates(candidates),
            "poses": [b.pose.model_dump(mode="json") for b in bundles],
            "poses_by_candidate": {c: list(ids)
                                   for c, ids in sorted(per_candidate.items())},
            "clusters": dict(clusters),
            "n_clusters": len(set(clusters.values())),
            "n_clusters_by_candidate": per_candidate_clusters,
            "rejected": list(rejected),
            "assembly_validation": [
                {"candidate_id": v.candidate_id, "route": v.route.value,
                 "ok": v.ok, "missing": v.missing,
                 "components": [{"name": c.name, "kind": c.kind,
                                 "required": c.required, "satisfied": c.satisfied,
                                 "detail": c.detail} for c in v.components]}
                for v in validations],
            "ranking_score_caveat": RANKING_SCORE_CAVEAT,
        })


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _split_confidence(raw: Mapping[str, Any]) -> tuple[dict[str, float],
                                                       dict[str, Any]]:
    """Split a runner's confidence payload into scalars and everything else.

    ``ComplexPose.model_confidence`` is typed as ``dict[str, float]``, so a PAE
    *matrix* cannot live there. Rather than reduce it to a mean and store that
    under the same name -- which would make a summary indistinguishable from a
    reported scalar -- non-scalar entries are kept separately and written to
    the pose index as-is.
    """
    scalars: dict[str, float] = {}
    other: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if isinstance(value, bool):
            other[key] = value
        elif isinstance(value, (int, float)):
            scalars[str(key)] = float(value)
        else:
            other[str(key)] = value
    return scalars, other


def _template_lookup(
    templates: Mapping[str, CatalyticTemplate] | Sequence[CatalyticTemplate] | None,
) -> dict[str, CatalyticTemplate]:
    """Index catalytic templates by id and by family name."""
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

    Never "the only template loaded": assembling an AKR candidate against an
    SDR template would demand the wrong cofactor state and the wrong assembly,
    and would do it confidently.
    """
    tid = cand.catalytic_mapping.catalytic_template_id
    if tid and tid in templates:
        return templates[tid]
    family = cand.family.family_name
    if family:
        return templates.get(f"family:{family}")
    return None
