"""Interface ``normalize_reaction``: turn a loose brief into a falsifiable spec.

Why this step exists
--------------------
The most expensive error in an enzyme-mining campaign is not a bad ranking. It
is searching very efficiently for the wrong reaction. By the time a batch of 96
constructs comes back, nobody can tell whether the hit rate reflects the
enzymes or the fact that the task was written as "reduce the ketone" and the
substrate turned out to be a mixture of two diastereomers, or an aldehyde, or a
symmetric ketone that cannot produce the enantiomer the project was ordered to
make.

So this step does the opposite of what a helpful assistant would do. It does
**not** complete the specification. It reads what the operator supplied,
verifies what can be verified, and produces a precise list of what is still
missing, phrased so that a human can resolve each item. Three refusals in
particular are deliberate:

* **A name is not a structure.** "acetophenone" is resolvable by a chemist and
  by a database; it is not resolvable by this code, and a silently resolved
  name is how the wrong tautomer, salt form or enantiomer enters a run. A name
  without a structure is a blocking QC flag.
* **Whether a stereocentre is created is not inferred from the reaction class.**
  Reduction of an aldehyde, or of a symmetric ketone, creates none. The class
  label is itself an operator assertion and is frequently the thing that is
  wrong. Without a chemistry toolkit the question cannot be answered reliably,
  so :func:`perceive_new_stereocenter` returns ``None`` and the step raises an
  uncertainty instead of guessing. Even when RDKit is installed, its answer is
  offered as a *proposal* requiring confirmation, because
  :class:`~eagent.schemas.reaction.Assumption` admits no "a tool said so"
  authority -- and it is right not to.
* **Mode B refuses to name a best enzyme.** With the reaction fixed and the
  substrate open, there is no such thing as the best enzyme; there is a set of
  chemical sub-spaces the operator has to choose between.
  :data:`SUBSTRATE_CLASS_SCAFFOLD` lays those out as an explicit decision.

Nothing here reaches the network and nothing here reads a database.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Mapping, Sequence

import yaml

from ..context import RunContext
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..provenance import sha256_file, sha256_obj, utc_now
from ..schemas import (
    AtomRef, CofactorState, ReactionClass, Stereochemistry, TaskMode,
    cofactor_state_from_ligand_code,
)
from .base import ScientificInterface

__all__ = [
    "SubstrateChemotype",
    "ChemotypeSubSpace",
    "SUBSTRATE_CLASS_SCAFFOLD",
    "PROCHIRAL_CENTRE_IS_ELECTROPHILE",
    "StereocentreBasis",
    "StereocentreCall",
    "perceive_new_stereocenter",
    "AtomMapIssue",
    "AtomMapReport",
    "parse_atom_map",
    "split_reaction_smiles",
    "validate_atom_map",
    "NormalizeReaction",
]


# ---------------------------------------------------------------------------
# substrate chemotypes (shared with retrieve_evidence)
# ---------------------------------------------------------------------------

class SubstrateChemotype(str, enum.Enum):
    """The sub-spaces a ketone-reduction task is decided between.

    Defined here, next to the step that makes the operator choose, so that the
    evidence matrix downstream is binned on exactly the axis the decision was
    made on. Two modules each inventing their own chemotype vocabulary would
    produce a matrix that does not answer the question that was asked.

    ``UNCLASSIFIED`` is a real member, not a failure mode to be avoided. A
    substrate whose chemotype nobody has recorded belongs in a visible column
    of the evidence matrix, because silently dropping it would make the
    evidence look more complete than it is.
    """

    AROMATIC_KETONE = "aromatic_ketone"
    ALIPHATIC_KETONE = "aliphatic_ketone"
    CYCLIC_KETONE = "cyclic_ketone"
    FUNCTIONALISED_KETONE = "functionalised_ketone"
    UNCLASSIFIED = "unclassified"

    @property
    def is_decided(self) -> bool:
        return self is not SubstrateChemotype.UNCLASSIFIED


@dataclass(frozen=True)
class ChemotypeSubSpace:
    """One sub-space of the reaction space, stated as a decision to be taken.

    Every field is methodological on purpose. Saying "aromatic ketones have
    mature enzymes" would be an unsourced scientific claim made by this file;
    saying "ask the kinetics layer which chemotypes carry confirmed records"
    is a search instruction whose answer comes from evidence retrieval. The
    distinction is the whole point of the scaffold.
    """

    chemotype: SubstrateChemotype
    definition: str
    decision_required: str
    why_it_matters: str
    evidence_to_gather: tuple[str, ...]
    representative_question: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chemotype": self.chemotype.value,
            "definition": self.definition,
            "decision_required": self.decision_required,
            "why_it_matters": self.why_it_matters,
            "evidence_to_gather": list(self.evidence_to_gather),
            "representative_question": self.representative_question,
        }


#: Mode B scaffold: the chemical sub-spaces an operator must decide between
#: before "which enzyme is best" is even a well-formed question.
SUBSTRATE_CLASS_SCAFFOLD: tuple[ChemotypeSubSpace, ...] = (
    ChemotypeSubSpace(
        chemotype=SubstrateChemotype.AROMATIC_KETONE,
        definition="The carbonyl carbon is attached directly to an aromatic ring "
                   "(aryl alkyl ketones and diaryl ketones).",
        decision_required="Decide whether the campaign targets aryl ketones, and "
                          "if so which ring substitution patterns are in scope.",
        why_it_matters="Conjugation to the ring changes the electronics of the "
                       "carbonyl and the steric demand of the two faces, so "
                       "evidence gathered on an aliphatic ketone does not "
                       "transfer. The two substituents also differ most in size "
                       "here, which is what a stereoselectivity model keys on.",
        evidence_to_gather=(
            "kinetics layer: confirmed records on aryl alkyl ketones, with the "
            "detection method that identified the alcohol product",
            "reaction layer: an atom-mapped representative reaction",
            "literature layer: engineering campaigns that changed ring tolerance",
        ),
        representative_question="Which specific aryl ketone will be ordered, in "
                                "what purity, and is an authentic standard of "
                                "both product enantiomers available?",
    ),
    ChemotypeSubSpace(
        chemotype=SubstrateChemotype.ALIPHATIC_KETONE,
        definition="Both carbonyl substituents are saturated open-chain carbon "
                   "chains.",
        decision_required="Decide whether the campaign targets open-chain "
                          "aliphatic ketones, and the chain lengths in scope.",
        why_it_matters="When the two substituents are similar in size, face "
                       "discrimination has little to work with, so an "
                       "enantioselectivity objective may be unattainable for "
                       "part of this sub-space. A symmetric ketone creates no "
                       "stereocentre at all and must be excluded explicitly.",
        evidence_to_gather=(
            "kinetics layer: records distinguishing the two chain lengths",
            "reaction layer: confirmation that the chosen member is prochiral",
            "literature layer: reports of selectivity collapse with similar "
            "substituents",
        ),
        representative_question="Is the chosen member prochiral at the carbonyl "
                                "carbon, or are its two substituents identical?",
    ),
    ChemotypeSubSpace(
        chemotype=SubstrateChemotype.CYCLIC_KETONE,
        definition="The carbonyl carbon is part of a ring.",
        decision_required="Decide whether ring systems are in scope, and whether "
                          "the product alcohol's ring stereochemistry (cis/trans "
                          "relative to existing centres) is part of the target.",
        why_it_matters="A ring constrains the approach geometry, and a ring "
                       "bearing an existing stereocentre turns the task into a "
                       "diastereoselectivity problem, which needs a different "
                       "analytical method from an ee measurement.",
        evidence_to_gather=(
            "kinetics layer: records on ring sizes comparable to the target",
            "reaction layer: whether the ring carbonyl is prochiral at all "
            "(a symmetric ring position is not)",
            "assay planning: whether the available chiral method separates "
            "diastereomers as well as enantiomers",
        ),
        representative_question="Does the target ring already carry a "
                                "stereocentre, making this a "
                                "diastereoselectivity task?",
    ),
    ChemotypeSubSpace(
        chemotype=SubstrateChemotype.FUNCTIONALISED_KETONE,
        definition="A ketone carrying an additional reactive group (halide, "
                   "ester, nitrile, amine, second carbonyl, acidic alpha "
                   "position).",
        decision_required="Decide whether functionalised ketones are in scope, "
                          "and which competing reaction at the second group is "
                          "acceptable.",
        why_it_matters="Chemoselectivity, not just stereoselectivity, becomes "
                       "the primary risk: a second electrophile may be reduced "
                       "instead, an alpha stereocentre may epimerise under the "
                       "reaction conditions, and the substrate may inhibit or "
                       "inactivate the enzyme. An assay that only measures "
                       "cofactor consumption cannot see any of this.",
        evidence_to_gather=(
            "kinetics layer: records for the same functional group combination, "
            "not merely the same carbon skeleton",
            "reaction layer: an atom mapping that shows which carbonyl is "
            "intended to react",
            "assay planning: a product-identifying method, since an indirect "
            "readout cannot distinguish the two possible products",
        ),
        representative_question="Which of the two electrophilic positions must "
                                "react, and how will the assay tell them apart?",
    ),
)


#: Reaction classes in which the atom attacked and the atom that becomes the new
#: stereocentre are necessarily the same atom. This is a definitional property of
#: the transformation, not a tunable threshold: in a carbonyl or imine reduction
#: the hydride adds to the electrophilic carbon, and that carbon is the carbinol
#: or amine centre. A spec naming two different atoms therefore describes a
#: different mechanism, and the geometry layer would measure the wrong distance.
PROCHIRAL_CENTRE_IS_ELECTROPHILE: frozenset[ReactionClass] = frozenset({
    ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
    ReactionClass.IMINE_REDUCTION,
})


# ---------------------------------------------------------------------------
# stereocentre perception
# ---------------------------------------------------------------------------

class StereocentreBasis(str, enum.Enum):
    """Who or what determined whether a new stereocentre is created.

    Recorded because the four origins carry different authority. Only the first
    two may write into the task spec; a toolkit perception is a proposal, and
    an undetermined value is a question for the operator.
    """

    TASK_SPEC = "task_spec"                  # the operator already stated it
    TEMPLATE = "template"                    # a sourced ReactionTemplate said so
    TOOLKIT_PROPOSAL = "toolkit_proposal"    # RDKit perceived it; needs confirming
    UNDETERMINED = "undetermined"            # nothing could answer it


@dataclass(frozen=True)
class StereocentreCall:
    """Whether the reaction creates a new stereocentre, and on whose authority."""

    value: bool | None
    basis: StereocentreBasis
    detail: str = ""
    source: str | None = None        # Assumption-style authority, when writable

    @property
    def writable_to_spec(self) -> bool:
        """Whether this may be resolved into the TaskSpec without a human.

        A toolkit proposal is excluded deliberately: the assumption ledger
        accepts operator, literature, template, database and experiment as
        authorities, and a perception routine is none of those.
        """
        return (self.value is not None
                and self.basis is StereocentreBasis.TEMPLATE
                and bool(self.source))

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "basis": self.basis.value,
                "detail": self.detail, "source": self.source}


def _rdkit() -> tuple[Any, str | None]:
    """Import RDKit if it happens to be installed, otherwise report its absence.

    Guarded because the project runs without a chemistry toolkit by design. The
    fallback is not a hand-rolled substructure matcher -- an approximate
    stereocentre perceiver is exactly the kind of plausible-looking machinery
    this codebase forbids -- it is an honest ``None``.
    """
    try:  # pragma: no cover - exercised only where RDKit is installed
        from rdkit import Chem, rdBase  # type: ignore
        return Chem, str(rdBase.rdkitVersion)
    except Exception:
        return None, None


def _potential_stereocentres(chem: Any, smiles: str) -> int | None:
    """Count potential tetrahedral stereocentres, or ``None`` if unparseable."""
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:  # modern RDKit
        found = chem.FindPotentialStereo(mol)
        return sum(1 for e in found
                   if str(getattr(e, "type", "")).endswith("Atom_Tetrahedral"))
    except Exception:  # pragma: no cover - older RDKit
        centres = chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                            useLegacyImplementation=False)
        return len(centres)


def perceive_new_stereocenter(substrate_smiles: str | None,
                              product_smiles: str | None) -> StereocentreCall:
    """Decide whether the product gains a stereocentre, or admit that you cannot.

    Returns ``value=None`` whenever the question cannot be settled, which
    includes every run without RDKit installed. This is the conservative branch
    the whole module is built around: an aldehyde and a symmetric ketone both
    yield achiral alcohols, a masked carbonyl may not react at the carbon the
    class label implies, and a wrong ``True`` here propagates into an
    enantioselectivity objective, a chiral assay and a synthesis order.

    Even a successful RDKit perception comes back as
    :attr:`StereocentreBasis.TOOLKIT_PROPOSAL`, never as an established fact.
    """
    if not substrate_smiles or not product_smiles:
        return StereocentreCall(
            None, StereocentreBasis.UNDETERMINED,
            "both substrate and product structures are needed before the "
            "question can even be asked")
    chem, version = _rdkit()
    if chem is None:
        return StereocentreCall(
            None, StereocentreBasis.UNDETERMINED,
            "RDKit is not installed; no stereocentre perception was attempted "
            "and none was improvised")
    n_sub = _potential_stereocentres(chem, substrate_smiles)
    n_prod = _potential_stereocentres(chem, product_smiles)
    if n_sub is None or n_prod is None:
        which = "substrate" if n_sub is None else "product"
        return StereocentreCall(
            None, StereocentreBasis.UNDETERMINED,
            f"RDKit could not parse the {which} SMILES")
    return StereocentreCall(
        n_prod > n_sub, StereocentreBasis.TOOLKIT_PROPOSAL,
        f"RDKit {version} counted {n_sub} potential stereocentre(s) in the "
        f"substrate and {n_prod} in the product; confirm before relying on it")


# ---------------------------------------------------------------------------
# atom-mapped reaction SMILES
# ---------------------------------------------------------------------------

_BRACKET_ATOM = re.compile(r"\[([^\[\]]+)\]")
_MAP_SUFFIX = re.compile(r":(\d+)$")
_ELEMENT = re.compile(r"^(?:\d+)?(\*|[A-Z][a-z]?|se|as|[bcnops])")


@dataclass(frozen=True)
class AtomMapIssue:
    """One problem found in an atom mapping, with the severity it deserves."""

    code: str
    severity: Severity
    message: str
    subject: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity.value,
                "message": self.message, "subject": self.subject}


@dataclass
class AtomMapReport:
    """What the mapping contains and what is wrong with it.

    Separated from the interface so it can be unit-tested against strings
    without constructing a run context, and so the checks can be reused by the
    verifier that re-reads a spec later.
    """

    parsed: bool = False
    reactant_ids: dict[int, str | None] = field(default_factory=dict)
    agent_ids: dict[int, str | None] = field(default_factory=dict)
    product_ids: dict[int, str | None] = field(default_factory=dict)
    issues: list[AtomMapIssue] = field(default_factory=list)

    @property
    def left_ids(self) -> dict[int, str | None]:
        """Map ids available before the reaction: reactants plus agents.

        Agents are included because a cofactor is routinely written in the
        agent position, and its hydride-donor atom must still be addressable by
        a :class:`~eagent.schemas.chem.ReactiveAtoms` reference.
        """
        merged = dict(self.agent_ids)
        merged.update(self.reactant_ids)
        return merged

    @property
    def blockers(self) -> list[AtomMapIssue]:
        return [i for i in self.issues if i.severity is Severity.BLOCKER]

    def add(self, code: str, severity: Severity, message: str,
            subject: str | None = None) -> None:
        self.issues.append(AtomMapIssue(code, severity, message, subject))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "reactant_map_ids": sorted(self.reactant_ids),
            "agent_map_ids": sorted(self.agent_ids),
            "product_map_ids": sorted(self.product_ids),
            "issues": [i.to_dict() for i in self.issues],
        }


def parse_atom_map(fragment: str) -> tuple[dict[int, str | None], list[int]]:
    """Map ids and their elements in one side of a reaction SMILES.

    Returns ``(id -> element, duplicates)``. Only bracketed atoms can carry a
    map number in SMILES, so a plain ``C`` is correctly invisible here: an
    unmapped atom is not an error, while the same id appearing twice on one
    side is, because then a reactive-atom reference no longer names one atom.

    The element is parsed best-effort and is ``None`` when the token cannot be
    read; a ``None`` element suppresses the element consistency check rather
    than inventing an element to compare against.
    """
    ids: dict[int, str | None] = {}
    duplicates: list[int] = []
    for token in _BRACKET_ATOM.findall(fragment):
        m = _MAP_SUFFIX.search(token)
        if not m:
            continue
        map_id = int(m.group(1))
        body = token[: m.start()]
        elem_match = _ELEMENT.match(body)
        element = elem_match.group(1) if elem_match else None
        if map_id in ids:
            duplicates.append(map_id)
        else:
            ids[map_id] = element
    return ids, duplicates


def split_reaction_smiles(rxn: str) -> tuple[str, str, str] | None:
    """Split ``reactants>agents>products``, or ``None`` when it is not one.

    Returns ``None`` rather than guessing at a single-component string: a
    "reaction SMILES" with no ``>`` is a molecule, and treating it as a
    reaction would silently validate a mapping that does not exist.
    """
    parts = rxn.split(">")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def validate_atom_map(
    rxn_smiles: str | None,
    referenced: Mapping[str, AtomRef],
    *,
    reaction_class: ReactionClass = ReactionClass.OTHER,
    creates_new_stereocenter: bool | None = None,
    substrate_is_biopolymer: bool = False,
) -> AtomMapReport:
    """Check an atom-mapped reaction SMILES against the declared reactive atoms.

    The point is not to validate SMILES syntax; it is to make the reactive-atom
    specification *checkable*. ``ReactiveAtoms`` addresses atoms by map id, and
    the geometry layer later measures distances to whatever those ids resolve
    to. If an id is absent, duplicated, or sits on an atom of the wrong element,
    every geometric criterion downstream is measured against the wrong atom and
    still produces confident-looking numbers. Each such inconsistency is a
    BLOCKER.

    ``referenced`` maps a role label (``"substrate.electrophile"``,
    ``"cofactor.NADPH.transfer_atom"``) to the :class:`AtomRef` that names it,
    so a failure can say which role is unresolvable rather than only which
    integer is missing.

    ``substrate_is_biopolymer`` changes what a *missing* map means. A peptide
    substrate is identified by its sequence and the residue the reaction
    modifies -- that is what
    :data:`~eagent.schemas.reaction.BIOPOLYMER_GATE_REQUIREMENTS` asks for --
    and nobody writes an atom-mapped reaction SMILES for a thirty-mer. A hard
    ``atom_map_missing`` on such a task blocked it forever on a representation
    it cannot have, while the field gate it is supposed to serve had already
    passed. A map that *is* supplied is still checked, whatever the substrate
    is.
    """
    report = AtomMapReport()
    if not rxn_smiles or not rxn_smiles.strip():
        if substrate_is_biopolymer:
            report.add("atom_map_absent_for_biopolymer", Severity.INFO,
                       "no atom-mapped reaction SMILES, which is expected for "
                       "a biopolymer substrate: it is identified by its "
                       "sequence and the residue the reaction modifies. Any "
                       "map supplied would still be checked")
            return report
        report.add("atom_map_missing", Severity.BLOCKER,
                   "no atom-mapped reaction SMILES; the reactive-atom "
                   "specification cannot be checked and the "
                   "reaction_spec_confirmed gate requires it")
        return report

    sides = split_reaction_smiles(rxn_smiles)
    if sides is None:
        report.add("atom_map_malformed", Severity.BLOCKER,
                   "not a reaction SMILES: expected two '>' separators "
                   "(reactants>agents>products)")
        return report

    report.parsed = True
    reactants, agents, products = sides
    for label, fragment, target in (
        ("reactant", reactants, "reactant_ids"),
        ("agent", agents, "agent_ids"),
        ("product", products, "product_ids"),
    ):
        ids, dupes = parse_atom_map(fragment)
        setattr(report, target, ids)
        for d in sorted(set(dupes)):
            report.add("atom_map_duplicate_id", Severity.BLOCKER,
                       f"map id {d} appears more than once on the {label} side, "
                       f"so a reactive-atom reference to it names no single atom",
                       subject=f"map:{d}")

    left = report.left_ids
    right = report.product_ids

    for pid in sorted(set(right) - set(left)):
        report.add("atom_map_product_only_id", Severity.BLOCKER,
                   f"map id {pid} appears on the product side with no origin on "
                   f"the reactant side; an atom cannot be created by the mapping",
                   subject=f"map:{pid}")
    for rid in sorted(set(left) - set(right)):
        report.add("atom_map_reactant_only_id", Severity.WARN,
                   f"map id {rid} is mapped on the reactant side but does not "
                   f"appear in any product; either the mapping is incomplete or "
                   f"a leaving group was dropped from the equation",
                   subject=f"map:{rid}")

    for role, ref in referenced.items():
        if ref is None:
            continue
        if ref.atom_map_id not in left:
            report.add("atom_map_missing_id", Severity.BLOCKER,
                       f"{role} refers to map id {ref.atom_map_id}, which does "
                       f"not occur on the reactant or agent side of the mapping; "
                       f"every distance defined against it would be measured "
                       f"against nothing",
                       subject=role)
            continue
        element = left[ref.atom_map_id]
        if ref.element and element and element.upper() != ref.element.upper():
            report.add("atom_map_element_mismatch", Severity.BLOCKER,
                       f"{role} declares element {ref.element} for map id "
                       f"{ref.atom_map_id}, but the mapping puts that id on "
                       f"'{element}'",
                       subject=role)

    _check_electrophile_prochiral(report, referenced, reaction_class,
                                  creates_new_stereocenter)
    return report


def _check_electrophile_prochiral(
    report: AtomMapReport,
    referenced: Mapping[str, AtomRef],
    reaction_class: ReactionClass,
    creates_new_stereocenter: bool | None,
) -> None:
    """Cross-check the electrophile against the prochiral centre.

    For a carbonyl or imine reduction these are one atom. When the spec names
    two, either the mechanism is not the declared one or one of the ids is
    wrong, and the stereochemistry call downstream would be made about an atom
    that is not the one being attacked.
    """
    electrophile = referenced.get("substrate.electrophile")
    prochiral = referenced.get("substrate.prochiral_center")

    if creates_new_stereocenter is True and prochiral is None:
        report.add("prochiral_center_unassigned", Severity.WARN,
                   "the task states that a new stereocentre is created but names "
                   "no prochiral centre; the face-of-approach call has no atom "
                   "to be made about",
                   subject="substrate.prochiral_center")

    if electrophile is None or prochiral is None:
        return

    if electrophile.atom_map_id == prochiral.atom_map_id:
        return

    if reaction_class in PROCHIRAL_CENTRE_IS_ELECTROPHILE:
        report.add("electrophile_prochiral_mismatch", Severity.BLOCKER,
                   f"reaction class {reaction_class.value} adds to the "
                   f"electrophilic centre, so the prochiral centre must be that "
                   f"same atom; the spec names map ids "
                   f"{electrophile.atom_map_id} and {prochiral.atom_map_id}",
                   subject="substrate.prochiral_center")
    else:
        report.add("electrophile_prochiral_differ", Severity.WARN,
                   f"the electrophile (map {electrophile.atom_map_id}) and the "
                   f"prochiral centre (map {prochiral.atom_map_id}) are "
                   f"different atoms; confirm that is intended for "
                   f"{reaction_class.value}",
                   subject="substrate.prochiral_center")

    if reaction_class is ReactionClass.ALDEHYDE_TO_PRIMARY_ALCOHOL:
        report.add("aldehyde_with_prochiral_center", Severity.WARN,
                   "the class is aldehyde reduction, yet a prochiral centre is "
                   "named; if that atom is the carbonyl carbon, the product "
                   "carbinol carries two hydrogens and is not a stereocentre",
                   subject="substrate.prochiral_center")


# ---------------------------------------------------------------------------
# template lookup
# ---------------------------------------------------------------------------

def _reaction_template_from_context(ctx: RunContext,
                                    reaction_class: ReactionClass) -> tuple[Any, str]:
    """Find *the* ReactionTemplate for this class, or none at all.

    The template library is owned by the harness and its API is not fixed
    here, so this probes a few plausible shapes and gives up quietly. Giving
    up is safe: the only consequence is that ``creates_new_stereocenter``
    stays unresolved and the operator is asked. Guessing a template would not
    be safe, because a template is an authority that can write into the task
    spec.

    AMBIGUITY IS A REFUSAL, NOT A PICK
    ----------------------------------
    Several templates may legitimately declare one reaction class -- a
    symmetric ketone and an aryl ketone are both
    ``ketone_to_secondary_alcohol`` and differ on whether the reduction
    creates a stereocentre. This used to take the first match, and the order
    is the library's insertion order, which is the order the files were
    globbed in: renaming two template files changed ``acetone`` from
    correctly having no new stereocentre to having one, and the task was then
    blocked for an unresolved configuration it does not have.

    A filename is not evidence. When more than one template claims the class,
    none is returned and the note names them all, so the operator passes the
    right one to the interface explicitly -- which the interface already
    accepts, and which is recorded as a decision rather than taken silently.
    """
    lib = getattr(ctx, "templates", None)
    if lib is None:
        return None, "no template library is attached to the run context"

    for attr in ("reaction_template", "get_reaction_template", "get_reaction"):
        fn = getattr(lib, attr, None)
        if callable(fn):
            try:
                tpl = fn(reaction_class.value)
            except Exception:
                continue
            if tpl is not None:
                return tpl, f"template library .{attr}({reaction_class.value})"

    matches: list[Any] = []
    seen: set[int] = set()
    source_attr = ""
    for attr in ("reaction_templates", "reactions"):
        coll = getattr(lib, attr, None)
        if isinstance(coll, Mapping):
            candidates: Iterable[Any] = coll.values()
        elif isinstance(coll, Sequence):
            candidates = coll
        else:
            continue
        for tpl in candidates:
            if getattr(tpl, "reaction_class", None) != reaction_class.value:
                continue
            if id(tpl) in seen:
                continue
            seen.add(id(tpl))
            matches.append(tpl)
            source_attr = source_attr or attr
    if len(matches) == 1:
        return matches[0], f"template library .{source_attr}"
    if matches:
        ids = ", ".join(sorted(str(getattr(m, "template_id", "unnamed"))
                               for m in matches))
        return None, (
            f"{len(matches)} reaction templates declare class "
            f"'{reaction_class.value}' ({ids}) and nothing here says which "
            f"applies to this substrate. The first match would be whichever "
            f"file sorted first, so none is used: pass the right template to "
            f"normalize_reaction explicitly")
    return None, (f"no ReactionTemplate for reaction class "
                  f"'{reaction_class.value}' in the attached template library")


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------

class NormalizeReaction(ScientificInterface):
    """Produce a ReactionSpec and an explicit list of what is still unresolved.

    Returns ``PARTIAL`` whenever anything is missing or inconsistent, with the
    gaps as uncertainties and the inconsistencies as QC flags, and ``SUCCESS``
    only when the ``reaction_spec_confirmed`` gate's fields are all resolved and
    no blocking inconsistency was found. It never returns ``SUCCESS`` on a spec
    it had to complete itself, because it never completes one.
    """

    name: ClassVar[str] = "normalize_reaction"
    description: ClassVar[str] = (
        "Normalise a reaction brief into a typed ReactionSpec, validate the atom "
        "mapping, and report every field that is still unresolved."
    )
    #: Intentionally empty. This is the step whose job is to discover what is
    #: missing; declaring the fields it checks as prerequisites would make it
    #: refuse to run on exactly the inputs it exists to triage.
    required_fields: ClassVar[tuple[str, ...]] = ()
    required_approvals: ClassVar[tuple[str, ...]] = ()
    depends_on: ClassVar[tuple[str, ...]] = ()
    version: ClassVar[str] = "0.1.0"

    #: The gate this step prepares. Named here so the artifact and the next
    #: action cannot drift apart from the schema's gate table.
    gate: ClassVar[str] = "reaction_spec_confirmed"

    def execute(self, ctx: RunContext, *, reaction_template: Any = None,
                **kwargs: Any) -> ToolResult:
        task = ctx.task
        spec = task.reaction
        result = ToolResult(
            status=Status.PARTIAL,
            provenance=Provenance(tool=self.name, tool_version=self.version),
        )
        checks: dict[str, Any] = {}

        mode_b = task.task_mode is TaskMode.REACTION_SPACE_EXPLORATION

        self._check_structures(result, spec, mode_b)
        stereo = self._resolve_stereocentre(ctx, result, reaction_template)
        checks["stereocentre"] = stereo.to_dict()
        self._check_stereo_consistency(result, spec, stereo)
        self._check_cofactors(ctx, result)

        map_report = validate_atom_map(
            spec.atom_mapped_reaction_smiles,
            self._referenced_atoms(ctx),
            reaction_class=spec.reaction_class,
            creates_new_stereocenter=spec.product.creates_new_stereocenter,
            substrate_is_biopolymer=spec.substrate_kind.is_biopolymer,
        )
        checks["atom_map"] = map_report.to_dict()
        for issue in map_report.issues:
            result.add_flag(issue.code, issue.severity, issue.message, issue.subject)
        if map_report.blockers:
            result.add_uncertainty(
                "atom_map_inconsistent",
                "Which atom-mapped reaction SMILES and reactive-atom ids "
                "describe this transformation correctly?",
                affects=[self.name, "screen_geometry", "call_stereochemistry"],
                resolvable_by="operator input, or an atom mapper whose output a "
                              "chemist has reviewed",
            )

        scaffold = self._mode_b_scaffold(result) if mode_b else None
        if scaffold is not None:
            checks["substrate_class_decision"] = scaffold

        if spec.reaction_class is ReactionClass.OTHER:
            result.add_flag("reaction_class_unset", Severity.WARN,
                            "reaction_class is 'other'; downstream family and "
                            "template lookup have nothing to key on",
                            subject="reaction.reaction_class")
            result.add_uncertainty(
                "reaction_class_unset",
                "Which ReactionClass describes this transformation?",
                affects=[self.name, "retrieve_evidence"],
                resolvable_by="operator input")

        unresolved = task.unresolved_for(self.gate)
        checks["unresolved_for_gate"] = unresolved
        if unresolved:
            result.add_uncertainty(
                "unresolved_fields",
                "Which values should fill: " + ", ".join(unresolved) + "?",
                affects=[self.gate], resolvable_by="operator input")
            result.add_next(
                "resolve_fields",
                f"The {self.gate} gate cannot open while these fields are null; "
                f"this step will not fill them, because a guessed structure or "
                f"stereocentre propagates into a synthesis order",
                {"gate": self.gate, "paths": unresolved}, requires_human=True)
        else:
            result.add_next(
                "request_approval",
                "Every field the gate requires is resolved; a human must still "
                "confirm the spec before mining starts",
                {"gate": self.gate}, requires_human=True)

        artifact = self._write_spec(ctx, checks, scaffold, stereo)
        result.artifacts.append(artifact)

        chem, rdkit_version = _rdkit()
        result.provenance = Provenance(
            tool=self.name,
            tool_version=self.version,
            inputs_sha256={"task_spec": sha256_obj(task.model_dump(mode="json"))},
            databases={},          # this step consults no database, by design
            models={"rdkit": rdkit_version or "absent"},
            parameters={
                "task_mode": task.task_mode.value,
                "reaction_class": spec.reaction_class.value,
                "gate": self.gate,
                "stereocentre_basis": stereo.basis.value,
                "prochiral_equals_electrophile_classes":
                    sorted(c.value for c in PROCHIRAL_CENTRE_IS_ELECTROPHILE),
            },
            random_seed=ctx.seed_for(self.name),
            started_at=None, finished_at=utc_now(),
        )
        result.data.update({
            "unresolved_for_gate": unresolved,
            "checks": checks,
            "mode": task.task_mode.value,
        })

        blockers = result.blockers
        if blockers:
            result.status = Status.PARTIAL
            result.message = (
                f"reaction spec is not usable yet: {len(blockers)} blocking "
                f"problem(s), {len(unresolved)} unresolved field(s)")
        elif unresolved:
            result.status = Status.PARTIAL
            result.message = (f"reaction spec is consistent but incomplete: "
                              f"{len(unresolved)} unresolved field(s)")
        else:
            result.status = Status.SUCCESS
            result.message = ("reaction spec is complete and internally "
                              "consistent; awaiting human confirmation")
        return result

    # -- checks ------------------------------------------------------------
    @staticmethod
    def _check_product(result: ToolResult, prod: Any, mode_b: bool) -> None:
        """The product has to be a structure whatever the substrate is.

        Shared by both substrate paths: a peptide substrate does not make the
        product optional, because the assay still has to detect one specific
        thing and ``confirmed_target_product`` is defined against it.
        """
        if prod.is_structurally_defined:
            return
        if prod.name:
            result.add_flag(
                "product_name_only", Severity.BLOCKER,
                f"product is given only as the name '{prod.name}'; the assay "
                f"has to detect a specific structure, and an ee is undefined "
                f"until the target configuration is attached to one",
                subject="reaction.product")
            result.add_uncertainty(
                "product_structure",
                f"What is the isomeric SMILES of '{prod.name}', including "
                f"the target configuration?",
                affects=["reaction.product.isomeric_smiles"],
                resolvable_by="operator input")
        elif not mode_b:
            result.add_flag(
                "product_unspecified", Severity.BLOCKER,
                "no product was given; without it the assay has no target "
                "and 'confirmed_target_product' cannot be defined",
                subject="reaction.product")

    @staticmethod
    def _check_biopolymer_substrate(result: ToolResult, bio: Any) -> None:
        """What a peptide, protein or nucleic-acid substrate has to carry.

        Its sequence, and the residue the reaction acts on. Those are the two
        things :data:`~eagent.schemas.reaction.BIOPOLYMER_GATE_REQUIREMENTS`
        asks for, and this check now asks for the same two -- a step and the
        gate it serves disagreeing about which object the substrate is in is
        how a fully specified task blocks forever on a SMILES it cannot have.
        """
        if not bio.is_structurally_defined:
            result.add_flag(
                "biopolymer_substrate_unspecified", Severity.BLOCKER,
                f"the substrate is declared as a {bio.kind.value} but carries "
                f"no sequence; a biopolymer substrate is identified by its "
                f"sequence, and a name fixes neither its length nor the "
                f"residue the reaction acts on",
                subject="reaction.biopolymer_substrate.sequence")
            result.add_uncertainty(
                "biopolymer_substrate_sequence",
                f"What is the sequence of the {bio.kind.value} substrate?",
                affects=["reaction.biopolymer_substrate.sequence"],
                resolvable_by="operator input")
            return
        residues = getattr(bio, "reactive_residues", None)
        named = [r for r in (list(getattr(residues, "modified", ()) or ())
                             + list(getattr(residues, "recognised", ()) or ()))]
        if not named:
            result.add_flag(
                "biopolymer_reactive_residue_unspecified", Severity.BLOCKER,
                f"the {bio.kind.value} substrate carries a sequence of "
                f"{bio.length} residue(s) but names none as the one the "
                f"reaction modifies; without it the assay has no position to "
                f"watch and no negative can be stated at a position",
                subject="reaction.biopolymer_substrate.reactive_residues")
            result.add_uncertainty(
                "biopolymer_reactive_residue",
                "Which residue of the substrate does the reaction modify?",
                affects=["reaction.biopolymer_substrate.reactive_residues.modified"],
                resolvable_by="operator input")

    def _check_structures(self, result: ToolResult, spec: Any,
                          mode_b: bool) -> None:
        """Insist on structures, and never resolve a name into one.

        A chemical name is a request to a human or a database, not a structure.
        Resolving "4-chloroacetophenone" here would pick a tautomer, a salt form
        and an implicit stereochemistry that nobody chose, and the rest of the
        run would treat those choices as the operator's.

        WHICH SUBSTRATE OBJECT THIS READS
        ---------------------------------
        The one the spec declares. A biopolymer task carries its substrate in
        ``reaction.biopolymer_substrate`` and leaves ``reaction.substrate``
        empty; reading the small-molecule field regardless produced
        ``substrate_unspecified`` on a task whose substrate was fully
        specified, and whose field gate had already passed. Supplying an
        unrelated small molecule then *removed* the blocker -- the gate read
        the peptide, this check read the small molecule, and giving the wrong
        substrate made the task look better.
        """
        sub, prod = spec.substrate, spec.product
        bio = getattr(spec, "biopolymer_substrate", None)

        if bio is not None:
            self._check_biopolymer_substrate(result, bio)
            self._check_product(result, prod, mode_b)
            return

        if not sub.is_structurally_defined:
            if sub.name:
                result.add_flag(
                    "substrate_name_only", Severity.BLOCKER,
                    f"substrate is given only as the name '{sub.name}'; a name "
                    f"fixes neither tautomer, salt form nor stereochemistry, and "
                    f"this step will not resolve it",
                    subject="reaction.substrate")
                result.add_uncertainty(
                    "substrate_structure", f"What is the isomeric SMILES or "
                    f"molfile of '{sub.name}'?",
                    affects=["reaction.substrate.isomeric_smiles"],
                    resolvable_by="operator input, or a structure lookup a "
                                  "chemist has checked")
            elif not mode_b:
                result.add_flag(
                    "substrate_unspecified", Severity.BLOCKER,
                    "no substrate was given; in enzyme-mining and engineering "
                    "modes the substrate is the index of the whole search",
                    subject="reaction.substrate")
        elif sub.inchikey and not sub.isomeric_smiles:
            result.add_flag(
                "substrate_inchikey_without_structure", Severity.WARN,
                "the substrate is identified by InChIKey but carries no SMILES; "
                "a hash identifies a structure but cannot be docked or mapped",
                subject="reaction.substrate.isomeric_smiles")

        self._check_product(result, prod, mode_b)

        if (sub.isomeric_smiles and prod.isomeric_smiles
                and sub.isomeric_smiles.strip() == prod.isomeric_smiles.strip()):
            result.add_flag(
                "substrate_product_identical", Severity.BLOCKER,
                "substrate and product carry the same SMILES, so the spec "
                "describes no transformation",
                subject="reaction")

    def _resolve_stereocentre(self, ctx: RunContext, result: ToolResult,
                              template: Any) -> StereocentreCall:
        """Settle ``creates_new_stereocenter`` by authority, or leave it open.

        Order of authority: the operator's own statement, then a sourced
        ReactionTemplate, then -- as a proposal only -- a toolkit perception.
        The reaction class is deliberately not consulted: it is an assertion
        about the chemistry that is frequently the very thing that is wrong, and
        inferring the stereocentre from it would make the error unfalsifiable.
        """
        task = ctx.task
        spec = task.reaction

        if spec.product.creates_new_stereocenter is not None:
            return StereocentreCall(
                spec.product.creates_new_stereocenter,
                StereocentreBasis.TASK_SPEC,
                "stated in the task spec by the operator")

        tpl = template
        origin = "passed to the interface call"
        if tpl is None:
            tpl, origin = _reaction_template_from_context(ctx, spec.reaction_class)
        tpl_value = getattr(tpl, "creates_stereocenter", None) if tpl else None
        if tpl_value is not None:
            tpl_id = str(getattr(tpl, "template_id", "unknown"))
            source = f"template:{tpl_id}"
            try:
                task.resolve(
                    "reaction.product.creates_new_stereocenter", bool(tpl_value),
                    source=source,
                    justification=f"ReactionTemplate {tpl_id} ({origin}) states "
                                  f"creates_stereocenter={tpl_value}")
            except Exception as exc:  # a template that cannot be an authority
                result.add_flag("template_assumption_rejected", Severity.WARN,
                                f"ReactionTemplate {tpl_id} could not be recorded "
                                f"as an authority: {exc}",
                                subject="reaction.product.creates_new_stereocenter")
            else:
                return StereocentreCall(
                    bool(tpl_value), StereocentreBasis.TEMPLATE,
                    f"taken from ReactionTemplate {tpl_id} ({origin})",
                    source=source)

        proposal = perceive_new_stereocenter(spec.substrate.isomeric_smiles,
                                             spec.product.isomeric_smiles)
        if proposal.value is not None:
            result.add_flag(
                "stereocentre_toolkit_proposal", Severity.WARN,
                f"RDKit proposes creates_new_stereocenter={proposal.value} "
                f"({proposal.detail}); the field is left unresolved because a "
                f"toolkit is not one of the authorities the assumption ledger "
                f"accepts",
                subject="reaction.product.creates_new_stereocenter")
            result.add_next(
                "confirm_stereocentre",
                "A chemist confirms or rejects the toolkit's perception; it then "
                "enters the spec as an operator assumption",
                {"proposed_value": proposal.value, "detail": proposal.detail},
                requires_human=True)
        else:
            result.add_flag(
                "stereocentre_undetermined", Severity.BLOCKER,
                f"whether the reaction creates a new stereocentre is "
                f"undetermined ({proposal.detail}); it is not inferred from the "
                f"reaction class, because an aldehyde and a symmetric ketone "
                f"both give an achiral alcohol",
                subject="reaction.product.creates_new_stereocenter")
        result.add_uncertainty(
            "stereocentre_undetermined",
            "Does this reaction create a new stereocentre in the product?",
            affects=["reaction.product.creates_new_stereocenter",
                     "call_stereochemistry", "plan_batch"],
            resolvable_by="operator input, or a sourced ReactionTemplate")
        return proposal

    def _check_stereo_consistency(self, result: ToolResult, spec: Any,
                                  stereo: StereocentreCall) -> None:
        """Catch a target configuration that the chemistry cannot deliver."""
        target = spec.product.target_stereochemistry
        value = spec.product.creates_new_stereocenter

        if value is False and target in (Stereochemistry.R, Stereochemistry.S):
            result.add_flag(
                "stereo_target_without_stereocentre", Severity.BLOCKER,
                f"the spec asks for configuration {target.value} but states that "
                f"no new stereocentre is created; one of the two is wrong, and "
                f"an enantioselectivity objective would be unachievable",
                subject="reaction.product")
        if value is True and target is Stereochemistry.ACHIRAL:
            result.add_flag(
                "stereo_achiral_with_stereocentre", Severity.BLOCKER,
                "the spec creates a stereocentre but declares the product "
                "achiral; the assay plan would omit the chiral method",
                subject="reaction.product")
        if value is True and target is Stereochemistry.UNSPECIFIED:
            result.add_uncertainty(
                "target_configuration",
                "Which enantiomer is the target: R, S, or is racemic acceptable?",
                affects=["reaction.product.target_stereochemistry", "plan_batch"],
                resolvable_by="operator input")
        if value is True and spec.product.authentic_standard_available is None:
            result.add_flag(
                "authentic_standard_unknown", Severity.WARN,
                "a stereocentre is created but it is not recorded whether an "
                "authentic standard exists; without one, a chiral method cannot "
                "assign which peak is the target enantiomer",
                subject="reaction.product.authentic_standard_available")

    def _check_cofactors(self, ctx: RunContext, result: ToolResult) -> None:
        """Make the cofactor's oxidation state explicit, or say that it is not.

        NAD(P)+ and NAD(P)H are different molecules. A template or a structure
        carrying the oxidised ligand cannot support a hydride-transfer geometry,
        and the two are routinely confused because their PDB component ids look
        alike. The contradiction check below is the cheap place to catch it.
        """
        options = ctx.task.conditions.cofactor_options
        if not options:
            result.add_flag(
                "cofactor_unspecified", Severity.WARN,
                "no cofactor option is declared; a hydride-transfer geometry "
                "cannot be defined without naming the donor",
                subject="conditions.cofactor_options")
            result.add_uncertainty(
                "cofactor_identity",
                "Which cofactor (and which oxidation state) does the target "
                "reaction use?",
                affects=["conditions.cofactor_options", "build_complex"],
                resolvable_by="operator input, or a sourced CatalyticTemplate")
            return

        for cof in options:
            derived = cofactor_state_from_ligand_code(cof.ligand_code)
            if cof.state is CofactorState.UNKNOWN:
                if derived is CofactorState.UNKNOWN:
                    result.add_flag(
                        "cofactor_state_unknown", Severity.WARN,
                        f"cofactor {cof.name} has no recorded oxidation state; "
                        f"NAD(P)+ and NAD(P)H are not interchangeable in a "
                        f"hydride-transfer model",
                        subject=f"cofactor:{cof.name}")
                else:
                    result.add_next(
                        "confirm_cofactor_state",
                        f"The PDB component id '{cof.ligand_code}' implies "
                        f"{derived.value}; an operator confirms it rather than "
                        f"this step writing it in",
                        {"cofactor": cof.name, "ligand_code": cof.ligand_code,
                         "implied_state": derived.value}, requires_human=True)
            elif derived is not CofactorState.UNKNOWN and derived is not cof.state:
                result.add_flag(
                    "cofactor_state_contradicts_ligand_code", Severity.BLOCKER,
                    f"cofactor {cof.name} is declared {cof.state.value} but its "
                    f"ligand code '{cof.ligand_code}' is the "
                    f"{derived.value} form",
                    subject=f"cofactor:{cof.name}")
            if cof.state is CofactorState.REDUCED and cof.transfer_atom is None:
                result.add_flag(
                    "cofactor_transfer_atom_missing", Severity.WARN,
                    f"{cof.name} is the reduced form but names no transfer atom; "
                    f"the hydride-transfer distance has no donor to measure from",
                    subject=f"cofactor:{cof.name}")

    def _referenced_atoms(self, ctx: RunContext) -> dict[str, AtomRef]:
        """Collect every atom the spec addresses by map id, labelled by role."""
        spec = ctx.task.reaction
        ra = spec.substrate.reactive_atoms
        out: dict[str, AtomRef] = {}
        for label, ref in (("electrophile", ra.electrophile),
                           ("nucleophile", ra.nucleophile),
                           ("leaving_group", ra.leaving_group),
                           ("prochiral_center", ra.prochiral_center)):
            if ref is not None:
                out[f"substrate.{label}"] = ref
        for n, ref in enumerate(ra.stabilised_atoms):
            out[f"substrate.stabilised_atoms[{n}]"] = ref
        for cof in ctx.task.conditions.cofactor_options:
            if cof.transfer_atom is not None:
                out[f"cofactor.{cof.name}.transfer_atom"] = cof.transfer_atom
        return out

    def _mode_b_scaffold(self, result: ToolResult) -> dict[str, Any]:
        """Mode B: enumerate the sub-spaces and refuse to pick a winner.

        "Which enzyme is best for ketone reduction?" has no answer, and
        producing one would be the most damaging thing this step could do:
        every downstream number would then be computed against a substrate
        nobody chose.
        """
        result.add_flag(
            "substrate_class_undecided", Severity.WARN,
            "mode B: the reaction is fixed but the substrate is not; the "
            "chemical sub-space must be chosen before candidates mean anything",
            subject="reaction.substrate")
        result.add_uncertainty(
            "substrate_sub_space",
            "Which substrate sub-space is in scope: "
            + ", ".join(s.chemotype.value for s in SUBSTRATE_CLASS_SCAFFOLD)
            + "? And which representative member will be ordered?",
            affects=["reaction.substrate", "retrieve_evidence", "plan_batch"],
            resolvable_by="operator decision")
        result.add_next(
            "choose_substrate_sub_space",
            "An operator picks one or more sub-spaces and a representative "
            "substrate for each; evidence retrieval then runs per sub-space",
            {"sub_spaces": [s.to_dict() for s in SUBSTRATE_CLASS_SCAFFOLD]},
            requires_human=True)
        result.add_flag(
            "best_enzyme_claim_refused", Severity.INFO,
            "no 'best enzyme' is reported for an unspecified substrate: "
            "selectivity and activity are properties of an enzyme-substrate "
            "pair, so a ranking without a substrate would be a ranking of "
            "nothing",
            subject="reaction.substrate")
        return {
            "sub_spaces": [s.to_dict() for s in SUBSTRATE_CLASS_SCAFFOLD],
            "refusals": [{
                "request": "name the best enzyme for this reaction",
                "refused_because": "activity and stereoselectivity are properties "
                                   "of an enzyme-substrate pair; with the "
                                   "substrate unspecified there is nothing to "
                                   "rank against",
                "unblocked_by": "choose a sub-space and a representative "
                                "substrate structure",
            }],
        }

    # -- artifact ----------------------------------------------------------
    def _write_spec(self, ctx: RunContext, checks: dict[str, Any],
                    scaffold: dict[str, Any] | None,
                    stereo: StereocentreCall) -> Artifact:
        """Write ``reaction_spec.yaml``: the spec plus what is wrong with it.

        The checks travel in the same file as the spec on purpose. A spec file
        that looks complete, read six months later without its QC record, is
        indistinguishable from one that was completed by guesswork.
        """
        task = ctx.task
        path = ctx.path("reaction_spec.yaml")
        document: dict[str, Any] = {
            "task_id": task.task_id,
            "task_mode": task.task_mode.value,
            "generated_by": {"interface": self.name, "version": self.version,
                             "at": utc_now()},
            "reaction": task.reaction.model_dump(mode="json"),
            "conditions": task.conditions.model_dump(mode="json"),
            "objectives": task.objectives.model_dump(mode="json"),
            "assumptions": [a.model_dump(mode="json") for a in task.assumptions],
            "stereocentre_determination": stereo.to_dict(),
            "checks": checks,
            "substrate_class_decision": scaffold,
        }
        header = (
            "# reaction_spec.yaml -- produced by the normalize_reaction interface.\n"
            "#\n"
            "# A null field means 'not yet determined'. Nothing in this file was\n"
            "# filled in by inference: every value either came from the operator,\n"
            "# from a sourced template recorded under 'assumptions', or is absent.\n"
            "# 'checks.unresolved_for_gate' lists exactly what the\n"
            "# reaction_spec_confirmed gate is still waiting for.\n"
        )
        path.write_text(header + yaml.safe_dump(document, sort_keys=False,
                                                allow_unicode=True),
                        encoding="utf-8")
        return Artifact(
            key="reaction_spec",
            path=str(path),
            kind="file",
            sha256=sha256_file(path),
            summary=(f"normalised reaction spec with "
                     f"{len(checks.get('unresolved_for_gate', []))} unresolved "
                     f"field(s) for the {self.gate} gate"),
        )
