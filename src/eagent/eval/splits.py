"""Leakage control: the place retrospective enzyme benchmarks actually fail.

WHAT GOES WRONG, AND WHY THE NUMBER STILL LOOKS GOOD
====================================================
A retrospective evaluation of enzyme function prediction is usually reported
as "trained on these records, tested on those". Four separate mechanisms make
the test rows not new:

* **Re-curation.** One measurement, in one paper, is curated into one
  database, re-integrated by a second, and re-integrated again by a third.
  "Train on database A, test on database B" is then not a split at all: the
  test rows are the training rows wearing a different accession. This is
  stated in :mod:`eagent.datalayer.lineage` and it is the reason the source
  database is *absent* from the grouping key. :func:`audit_leakage` detects
  exactly this case through the ``derived_from`` edges of
  :class:`~eagent.datalayer.registry.SourceRegistry`, so the claim "these are
  two independent resources" can be checked instead of assumed.
* **Variants of one parent.** A parent and its twelve single mutants differ at
  one position each. Put the parent in train and a mutant in test and the
  model is being asked to interpolate inside a sequence it has already seen.
* **One publication, many rows.** Rows from one paper share a construct, an
  assay, a calibration and an analyst. They are not independent samples of the
  world, and splitting them apart measures memorisation of that paper's
  idiosyncrasies.
* **Sequence clusters.** Two sequences at 95% identity are, for every purpose
  a substrate-specificity model cares about, the same protein.

Each of these inflates the test score, and none of them leaves a trace in the
results table. The split therefore has to be grouped *before* it is made and
audited *after* it is made.

THE THREE REGIMES, AND WHAT EACH ONE CLAIMS
===========================================
``novel_enzyme``
    Test sequence clusters absent from training. Substrates are deliberately
    shared: the question is whether the method transfers to an unseen enzyme
    on chemistry it has seen.
``novel_substrate``
    Test substrate scaffolds absent from training. The question is whether the
    method transfers to an unseen substrate.
``dual_extrapolation``
    Both at once. This is the regime a discovery campaign is actually in, and
    it is the one on which published numbers are scarcest.

GROUPING IS NOT OPTIONAL, AND THERE IS ONLY ONE GROUPING KEY
=============================================================
The unit that is assigned to a fold is never a record. It is the
leakage-safe group from :func:`eagent.datalayer.lineage.leakage_safe_groups`:
the transitive closure over publication, experiment activity, parent sequence
lineage and sequence cluster. No second grouping key is defined here -- a
second spelling of "which rows are the same system" is precisely how an audit
comes to disagree with the split it is auditing.

THE PRICE OF THAT INVARIANT, STATED PLAINLY
===========================================
For ``novel_substrate`` and ``dual_extrapolation`` the split unit is the
closure over *both* the lineage group and the substrate scaffold, because a
scaffold that straddles the boundary is scaffold leakage and a lineage group
that straddles it is lineage leakage, and both must hold at once.

A consequence follows that a reader must not discover by surprise: an enzyme
measured on a held-out scaffold is pulled into the test fold along with its
other measurements, so this implementation's ``novel_substrate`` regime is
*stricter* than the common published practice of holding out substrates while
leaving the same enzymes on both sides. The common practice leaks -- the same
paper, the same construct and the same assay appear on both sides -- so the
strict side is the one taken here. The cost is a smaller usable dataset and,
when the closure swallows everything, a split that cannot be formed at all.
That case is **reported short**, with the arithmetic, rather than relaxed.

SCAFFOLDS WITHOUT RDKIT
=======================
rdkit is not installed in this environment, and a scaffold key is still
needed. :func:`scaffold_key` uses rdkit's Murcko scaffold when rdkit is
importable and otherwise computes a pure-Python *ring-and-linker skeleton*
(the 2-core of the molecular graph) hashed with a Weisfeiler-Lehman
refinement. :class:`ScaffoldKey` carries the basis it used and a list of the
limitations that basis has; the two bases produce different key spaces and
must never be mixed inside one split, which :func:`audit_leakage` reports.

NOTHING HERE INVENTS A FACET
============================
A record whose sequence cluster or substrate structure cannot be resolved is
not quietly assigned to a fold. It is excluded from both, named, and paired
with the thing a curator has to supply, because "we could not tell whether
this row leaks" and "this row does not leak" are different statements and only
one of them is a clean split.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..datalayer.lineage import (
    ClusterLookup,
    grouping_key,
    leakage_safe_groups,
    split_leakage,
)
# Imported rather than re-derived: the ids this module reports must be exactly
# the keys ``leakage_safe_groups`` returns. A second local spelling of "which
# field identifies this row" would drift, and the audit would then disagree
# with the split it is auditing -- while still printing "clean".
from ..datalayer.lineage import _record_id as lineage_record_id
from ..errors import EAgentError
from ..provenance import sha256_text

__all__ = [
    "SplitRegime",
    "LeakageCategory",
    "ScaffoldBasis",
    "ScaffoldKey",
    "LeakageOverlap",
    "LeakageAudit",
    "GroupedSplit",
    "SplitNotPossibleError",
    "SCAFFOLD_KEY_ALGORITHM",
    "audit_leakage",
    "grouped_split",
    "rdkit_available",
    "record_ids",
    "scaffold_key",
    "source_tokens_of",
]


class SplitNotPossibleError(EAgentError):
    """A split was requested that cannot be made without leaking.

    Raised only for a *precondition* the caller can fix -- most importantly a
    ``novel_enzyme`` split requested with no sequence clustering, which is a
    claim nobody could check. A dataset that merely cannot be divided is not an
    error: :func:`grouped_split` returns that split short, with the arithmetic
    in ``shortfall_reason``, the way a short batch is reported short.
    """


# ==========================================================================
# Regimes and categories
# ==========================================================================

class SplitRegime(str, enum.Enum):
    """Which extrapolation a test fold is entitled to claim."""

    NOVEL_ENZYME = "novel_enzyme"
    NOVEL_SUBSTRATE = "novel_substrate"
    DUAL_EXTRAPOLATION = "dual_extrapolation"
    SHARED_ENZYME_NOVEL_SUBSTRATE = "shared_enzyme_novel_substrate"

    @property
    def holds_out_sequences(self) -> bool:
        """Whether test sequence clusters must be absent from training."""
        return self in (SplitRegime.NOVEL_ENZYME, SplitRegime.DUAL_EXTRAPOLATION)

    @property
    def holds_out_substrates(self) -> bool:
        """Whether test substrate scaffolds must be absent from training."""
        return self in (SplitRegime.NOVEL_SUBSTRATE,
                        SplitRegime.DUAL_EXTRAPOLATION,
                        SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE)

    @property
    def permits_shared_enzymes(self) -> bool:
        """Whether the same enzyme may appear on both sides by design.

        True only for :attr:`SHARED_ENZYME_NOVEL_SUBSTRATE`, which exists
        because the other three cannot express the commonest substrate-scope
        question. Grouping by lineage, as the leakage-safe regimes must, drags
        every one of an enzyme's measurements into whichever fold its
        held-out substrate landed in, so a substrate-scope split also becomes
        enzyme-disjoint and answers a harder question than the one asked.
        """
        return self is SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE

    @property
    def is_weaker_than_leakage_safe(self) -> bool:
        """Whether a score here supports a narrower claim than the others.

        A regime that deliberately lets an enzyme appear on both sides cannot
        support a generalisation claim about enzymes, and saying so is the
        price of being able to ask the question at all.
        """
        return self.permits_shared_enzymes

    def claim(self) -> str:
        """The sentence a score under this regime is allowed to support."""
        return {
            SplitRegime.NOVEL_ENZYME:
                "performance on enzymes from sequence clusters the model never "
                "saw, using substrates it did see",
            SplitRegime.NOVEL_SUBSTRATE:
                "performance on substrate scaffolds the model never saw",
            SplitRegime.DUAL_EXTRAPOLATION:
                "performance on an unseen enzyme and an unseen substrate at "
                "once, which is the situation a discovery campaign is in",
            SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE:
                "performance on a new substrate scaffold for enzymes the model "
                "has already seen. This is deliberately weaker than "
                "novel_substrate: the same enzyme appears on both sides, so a "
                "score here says nothing about a new enzyme and must not be "
                "quoted as substrate generalisation without that qualifier",
        }[self]


class LeakageCategory(str, enum.Enum):
    """The kinds of overlap an audit distinguishes.

    They are kept apart because the remedies differ: a shared publication is
    fixed by regrouping, a shared cluster by re-clustering at a tighter
    identity, and a re-curated source pair cannot be fixed by splitting at all
    -- one of the two resources has to leave the evaluation.
    """

    SHARED_SEQUENCE_CLUSTER = "shared_sequence_cluster"
    SHARED_SCAFFOLD = "shared_scaffold"
    SHARED_PUBLICATION = "shared_publication"
    SHARED_PARENT_LINEAGE = "shared_parent_lineage"
    SHARED_SPLIT_GROUP = "shared_split_group"
    RECURATED_SOURCE = "recurated_source"

    def is_expected_under(self, regime: "SplitRegime | None") -> bool:
        """Whether this overlap is the regime's design rather than its failure.

        Two cases qualify. A ``novel_enzyme`` split shares substrates on
        purpose, so a shared scaffold there is not leakage. A
        ``shared_enzyme_novel_substrate`` split shares enzymes on purpose, so a
        shared sequence cluster, parent lineage or split group there is the
        design. Nothing excuses a shared publication or a re-curated source
        pair under any regime: those are leakage however the folds were drawn.
        """
        if regime is SplitRegime.NOVEL_ENZYME:
            return self is LeakageCategory.SHARED_SCAFFOLD
        if regime is SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE:
            return self in (LeakageCategory.SHARED_SEQUENCE_CLUSTER,
                            LeakageCategory.SHARED_PARENT_LINEAGE,
                            LeakageCategory.SHARED_SPLIT_GROUP)
        return False


#: Facet prefixes produced by :func:`eagent.datalayer.lineage.grouping_key`.
#: Read rather than recomputed, so the audit and the split cannot drift apart.
_CLUSTER_PREFIX = "cluster:"
_LINEAGE_PREFIX = "lineage:"
_UNRESOLVED_INFIX = "unresolved:"
_PUBLICATION_PREFIXES = ("doi:", "pub:")


# ==========================================================================
# Substrate scaffolds without rdkit
# ==========================================================================

#: Version token embedded in every pure-Python scaffold key. A change to the
#: skeleton rule or to the hash must bump it, so keys written by two versions
#: cannot be compared and silently agree.
SCAFFOLD_KEY_ALGORITHM: str = "wl3-2core-v1"

_ORGANIC_TWO_LETTER: tuple[str, ...] = ("Cl", "Br")
_ORGANIC_ONE_LETTER: frozenset[str] = frozenset("BCNOPSFI")
_AROMATIC_ONE_LETTER: frozenset[str] = frozenset("bcnops")
_BOND_CHARS: frozenset[str] = frozenset("-=#$:/\\~")


class ScaffoldBasis(str, enum.Enum):
    """What a scaffold key was actually computed from.

    Carried on every key because the bases are *different key spaces*. An
    rdkit Murcko scaffold and this module's ring-and-linker skeleton agree
    about which molecules share a core far more often than they agree about
    the string that names it, so comparing a key from one with a key from the
    other would read as "different scaffold" for the same molecule -- the
    direction that leaks.
    """

    RDKIT_MURCKO = "rdkit_murcko"
    RING_AND_LINKER_SKELETON = "ring_and_linker_skeleton"
    ACYCLIC_SKELETON = "acyclic_skeleton"
    INCHIKEY_CONSTITUTION_BLOCK = "inchikey_constitution_block"
    CALLER_SUPPLIED = "caller_supplied"
    UNRESOLVED = "unresolved"

    @property
    def is_structural(self) -> bool:
        """Whether the key came from a structure rather than from a hash block."""
        return self in (ScaffoldBasis.RDKIT_MURCKO,
                        ScaffoldBasis.RING_AND_LINKER_SKELETON,
                        ScaffoldBasis.ACYCLIC_SKELETON,
                        ScaffoldBasis.CALLER_SUPPLIED)


@dataclass(frozen=True)
class ScaffoldKey:
    """A substrate scaffold key, its basis, and what that basis cannot see.

    ``key`` is ``None`` when no structure was available. It is never a hash of
    the compound's *name*: "4-chloroacetophenone" and "2-chloroacetophenone"
    differ by one character and are different molecules, and an "(R)-" that
    should read "(S)-" is the entire objective of an asymmetric reduction.
    :mod:`eagent.datalayer.identity` rules name similarity inadmissible as a
    merge ground, and the same rule holds here.
    """

    key: str | None
    basis: ScaffoldBasis
    limitations: tuple[str, ...] = ()
    needs_curator: str | None = None
    source_smiles: str | None = None

    @property
    def resolved(self) -> bool:
        return self.key is not None

    @property
    def is_true_scaffold(self) -> bool:
        """Whether the key names a *core* rather than a whole constitution.

        ``False`` for the InChIKey block, which changes with every substituent.
        A novel-substrate split built on it holds out exact molecules rather
        than scaffolds and so *under*-states substrate similarity, which is the
        dangerous direction; callers get this flag rather than a footnote.
        """
        return self.basis in (ScaffoldBasis.RDKIT_MURCKO,
                              ScaffoldBasis.RING_AND_LINKER_SKELETON,
                              ScaffoldBasis.CALLER_SUPPLIED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "basis": self.basis.value,
            "limitations": list(self.limitations),
            "needs_curator": self.needs_curator,
            "is_true_scaffold": self.is_true_scaffold,
        }


def rdkit_available() -> bool:
    """Whether rdkit can be imported here. Guarded: it is an optional extra."""
    try:  # pragma: no cover - depends on the environment, not on this code
        import rdkit  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _rdkit_murcko_smiles(smiles: str) -> tuple[str, bool] | None:
    """``(key_body, is_acyclic)`` from rdkit, or ``None`` when rdkit cannot do it.

    Every failure path returns ``None`` rather than raising: an unparseable
    SMILES must fall through to the pure-Python reader (which may also refuse)
    and end as an *unresolved* scaffold, never as an exception that aborts a
    whole evaluation over one malformed row.

    An acyclic molecule has an empty Murcko framework. The canonical SMILES of
    the whole molecule is returned instead, flagged acyclic, because an empty
    key would fuse every acyclic substrate into one scaffold group.
    """
    try:  # pragma: no cover - rdkit is not installed in this environment
        from rdkit import Chem  # type: ignore
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
    except Exception:
        return None
    try:  # pragma: no cover - exercised only where rdkit is installed
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        core = MurckoScaffold.GetScaffoldForMol(mol)
        core_smiles = Chem.MolToSmiles(core) if core is not None else ""
        if core_smiles:
            return core_smiles, False
        return Chem.MolToSmiles(mol), True
    except Exception:
        return None


@dataclass
class _MolGraph:
    """A bond-order-free, stereo-free molecular graph. Enough for a skeleton."""

    elements: list[str] = field(default_factory=list)
    adjacency: list[set[int]] = field(default_factory=list)

    def add_atom(self, element: str) -> int:
        self.elements.append(element)
        self.adjacency.append(set())
        return len(self.elements) - 1

    def add_bond(self, a: int, b: int) -> None:
        if a == b:
            return
        self.adjacency[a].add(b)
        self.adjacency[b].add(a)


def _parse_smiles(smiles: str) -> _MolGraph | None:
    """Parse the SMILES subset this repository actually sees, or refuse.

    Deliberately small and deliberately strict. Bond orders, charges,
    isotopes, explicit hydrogen counts and stereochemistry are all discarded:
    the skeleton is a connectivity statement, and keeping bond orders would
    give a kekulised benzene and an aromatic benzene two different scaffold
    keys -- the same molecule split across the fold boundary, which is the
    failure this module exists to prevent.

    Returns ``None`` on anything it does not understand, so the caller reports
    an unresolved scaffold instead of a key computed from a misparse.
    """
    text = "".join(str(smiles).split())
    if not text:
        return None
    graph = _MolGraph()
    prev: int | None = None
    branch: list[int | None] = []
    ring_open: dict[str, int] = {}
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            branch.append(prev)
            i += 1
            continue
        if ch == ")":
            if not branch:
                return None
            prev = branch.pop()
            i += 1
            continue
        if ch == ".":
            prev = None
            i += 1
            continue
        if ch in _BOND_CHARS:
            i += 1
            continue
        if ch.isdigit() or ch == "%":
            if ch == "%":
                label = text[i + 1:i + 3]
                if len(label) != 2 or not label.isdigit():
                    return None
                i += 3
            else:
                label = ch
                i += 1
            if prev is None:
                return None
            partner = ring_open.pop(label, None)
            if partner is None:
                ring_open[label] = prev
            else:
                graph.add_bond(partner, prev)
            continue
        element: str | None = None
        if ch == "[":
            close = text.find("]", i)
            if close < 0:
                return None
            element = _bracket_element(text[i + 1:close])
            if element is None:
                return None
            i = close + 1
        elif text[i:i + 2] in _ORGANIC_TWO_LETTER:
            element = text[i:i + 2]
            i += 2
        elif ch in _ORGANIC_ONE_LETTER:
            element = ch
            i += 1
        elif ch in _AROMATIC_ONE_LETTER:
            element = ch.upper()
            i += 1
        elif ch == "*":
            element = "*"
            i += 1
        else:
            return None
        idx = graph.add_atom(element)
        if prev is not None:
            graph.add_bond(prev, idx)
        prev = idx
    if ring_open:
        return None            # an unclosed ring digit is a malformed SMILES
    if not graph.elements:
        return None
    return graph


def _bracket_element(body: str) -> str | None:
    """Element symbol inside ``[...]``, ignoring isotope, charge, H count, stereo."""
    i = 0
    while i < len(body) and body[i].isdigit():
        i += 1                                   # isotope
    if i >= len(body):
        return None
    if body[i] == "*":
        return "*"
    if not body[i].isalpha():
        return None
    symbol = body[i]
    if i + 1 < len(body) and body[i + 1].isalpha() and body[i + 1].islower() \
            and symbol.isupper():
        # Two-letter element, but not an aromatic 'H' count like [nH].
        symbol = symbol + body[i + 1]
    return symbol[0].upper() + symbol[1:]


def _largest_component(graph: _MolGraph) -> tuple[list[int], bool]:
    """Atoms of the biggest connected component, and whether others were dropped.

    A salt, a counterion or a solvate would otherwise change the scaffold of
    the molecule it travels with, so the same substrate supplied as the free
    base and as the hydrochloride would land on opposite sides of the split.
    """
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(graph.elements)):
        if start in seen:
            continue
        stack = [start]
        comp: list[int] = []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in graph.adjacency[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(sorted(comp))
    components.sort(key=lambda c: (-len(c), c[0]))
    return components[0], len(components) > 1


def _two_core(graph: _MolGraph, atoms: Sequence[int]) -> list[int]:
    """Ring-and-linker skeleton: repeatedly drop terminal atoms.

    What survives is every atom that lies on a ring or on a path between two
    rings, which is the Murcko framework idea without needing ring perception.
    Exocyclic substituents -- including the carbonyl oxygen of a ketone -- are
    pruned, so acetophenone and 4-chloroacetophenone share a key. That is the
    intended behaviour for a substrate-scaffold holdout, and it is also the
    behaviour that makes this key *coarser* than an exact structure match.
    """
    keep = set(atoms)
    changed = True
    while changed:
        changed = False
        for atom in sorted(keep):
            degree = sum(1 for nb in graph.adjacency[atom] if nb in keep)
            if degree <= 1:
                keep.discard(atom)
                changed = True
    return sorted(keep)


#: Elements treated as peripheral decoration rather than as the functional
#: group a reaction acts on. A chloro analogue of a substrate is the same
#: scaffold; a ketone analogue of a benzene is not.
_PERIPHERAL_ELEMENTS: frozenset[str] = frozenset({"F", "Cl", "Br", "I", "H"})

#: How far from the ring-and-linker core a heteroatom may sit and still count
#: as part of the core's functional group. Two bonds reaches the oxygen of a
#: ring-attached carbonyl, which is the group a carbonyl-reduction campaign is
#: about, without reaching across a linker into a second substituent.
_FUNCTIONAL_SHELL_BONDS = 2


def _functional_signature(graph: _MolGraph, core: Sequence[int]) -> str:
    """Core composition plus the functional heteroatoms hanging off it.

    The two-core alone is not a usable scaffold for this project: it reduces
    acetophenone, 4-chloroacetophenone and plain benzene to one ring, so in a
    carbonyl-reduction campaign the substrate axis of a split collapses and
    cannot hold out a substrate class at all. The reacting group has to
    survive into the key.

    The whole molecule's composition is the wrong fix in the other direction,
    because it makes every substituent a new scaffold and a chloro analogue of
    the training substrate becomes a "novel" substrate. So this counts the
    core's own elements plus the non-halogen heteroatoms within
    :data:`_FUNCTIONAL_SHELL_BONDS` of it: the carbonyl oxygen counts, a ring
    chlorine does not.

    Everything here is counted from element symbols and graph distance, never
    from bond orders, so every spelling of a molecule gives one signature. An
    aromatic and a kekulised benzene must not get two keys, and that property
    is worth more than the resolution a bond-order term would buy.

    What it still cannot do: tell an aromatic ring from a saturated one of the
    same composition. Nothing in this module perceives aromaticity, so
    cyclohexanone and acetophenone's ring system remain indistinguishable
    here. That over-groups, which costs data and never inflates a score.
    """
    members = set(core)
    counts: dict[str, int] = {}
    for a in members:
        counts[graph.elements[a]] = counts.get(graph.elements[a], 0) + 1

    # Breadth-first out from the core, collecting functional heteroatoms.
    frontier = set(members)
    seen = set(members)
    for _ in range(_FUNCTIONAL_SHELL_BONDS):
        nxt: set[int] = set()
        for atom in frontier:
            for nb in graph.adjacency[atom]:
                if nb in seen:
                    continue
                seen.add(nb)
                nxt.add(nb)
                element = graph.elements[nb]
                if element not in _PERIPHERAL_ELEMENTS and element != "C":
                    counts["~" + element] = counts.get("~" + element, 0) + 1
        frontier = nxt
    return ",".join(f"{el}{n}" for el, n in sorted(counts.items()))


def _wl_hash(graph: _MolGraph, atoms: Sequence[int], rounds: int = 3,
             signature: str = "") -> str:
    """Weisfeiler-Lehman hash of an induced subgraph: order-invariant, not canonical.

    Order invariance is the property that matters: the same molecule written
    two ways must give one key, or the split leaks. The price is that WL is not
    a complete graph invariant, so two genuinely different skeletons can
    collide and be held out together. That direction over-separates the folds,
    which costs data and never inflates a score, so it is the safe side to err
    on.
    """
    members = set(atoms)
    labels = {a: graph.elements[a] for a in members}
    for _ in range(rounds):
        nxt: dict[int, str] = {}
        for atom in members:
            neighbours = sorted(labels[nb] for nb in graph.adjacency[atom]
                                if nb in members)
            nxt[atom] = sha256_text(labels[atom] + "|" + ",".join(neighbours))[:16]
        labels = nxt
    n_bonds = sum(1 for a in members for b in graph.adjacency[a] if b in members) // 2
    payload = (f"{signature}|{len(members)}|{n_bonds}|"
               + ",".join(sorted(labels.values())))
    return sha256_text(payload)[:24]


def _substrate_fields(substrate: Any) -> tuple[str | None, str | None]:
    """``(isomeric_smiles, inchikey)`` from a substrate spec, a record, or a string."""
    if substrate is None:
        return None, None
    if isinstance(substrate, str):
        return substrate.strip() or None, None
    spec = getattr(substrate, "substrate", None)
    if spec is not None and hasattr(spec, "isomeric_smiles"):
        substrate = spec
    smiles = getattr(substrate, "isomeric_smiles", None)
    inchikey = getattr(substrate, "inchikey", None)
    return (str(smiles) if smiles else None,
            str(inchikey) if inchikey else None)


def scaffold_key(substrate: Any, *, prefer_rdkit: bool = True) -> ScaffoldKey:
    """Scaffold key for a substrate, with its basis and its limitations attached.

    Accepts a :class:`~eagent.schemas.chem.SubstrateSpec`, anything carrying
    one (an :class:`~eagent.schemas.record.ExperimentRecord`), or a bare SMILES
    string.

    Resolution order, strongest first:

    1. rdkit's Murcko scaffold, when rdkit is importable;
    2. the pure-Python ring-and-linker skeleton of the parsed SMILES;
    3. for an acyclic molecule, the whole skeleton (a molecule with no ring has
       no Murcko framework, and returning the empty framework would fuse every
       acyclic substrate into one group -- conservative, and meaningless);
    4. the first block of an InChIKey, which is a *constitution* hash and not a
       scaffold, flagged as such;
    5. unresolved, with the thing a curator must supply.

    There is no step that derives a key from the substrate's name.
    """
    smiles, inchikey = _substrate_fields(substrate)

    if smiles and prefer_rdkit:
        resolved = _rdkit_murcko_smiles(smiles)
        if resolved is not None:  # pragma: no cover - needs rdkit installed
            body, is_acyclic = resolved
            prefix = "acyclic:rdkit-murcko" if is_acyclic else "scaffold:rdkit-murcko"
            return ScaffoldKey(
                key=f"{prefix}:{body}",
                basis=ScaffoldBasis.RDKIT_MURCKO,
                limitations=(
                    "rdkit's Murcko scaffold keeps exocyclic double bonds and "
                    "perceives aromaticity; keys from this basis are NOT "
                    "comparable with the pure-Python ring-and-linker keys.",
                ),
                source_smiles=smiles,
            )

    if smiles:
        graph = _parse_smiles(smiles)
        if graph is None:
            return ScaffoldKey(
                key=None, basis=ScaffoldBasis.UNRESOLVED,
                limitations=(
                    "the pure-Python SMILES reader did not understand this "
                    "string; no key was guessed from it.",),
                needs_curator=(
                    f"a SMILES this reader can parse, or an rdkit install, for "
                    f"{smiles!r}"),
                source_smiles=smiles,
            )
        atoms, had_other_components = _largest_component(graph)
        shared = [
            "bond orders, charges, isotopes and stereochemistry are discarded, "
            "so cyclohexane and benzene share a key (over-grouping, which "
            "separates folds further than necessary rather than less).",
            "the Weisfeiler-Lehman hash is not a canonical form: distinct "
            "skeletons can collide and be held out together.",
            "the key combines the ring-and-linker skeleton with the core's "
            "composition and the non-halogen heteroatoms within two bonds of "
            "it, so a ring-attached carbonyl survives into the key while a "
            "ring halogen does not; saturated and aromatic rings of the same "
            "composition still collide, because nothing here perceives "
            "aromaticity.",
        ]
        if had_other_components:
            shared.append(
                "only the largest connected component was used; a counterion "
                "or solvate was ignored so that a salt and its free base share "
                "a key.")
        core = _two_core(graph, atoms)
        if core:
            return ScaffoldKey(
                key=f"scaffold:{SCAFFOLD_KEY_ALGORITHM}:{_wl_hash(graph, core, signature=_functional_signature(graph, core))}",
                basis=ScaffoldBasis.RING_AND_LINKER_SKELETON,
                limitations=tuple(shared),
                source_smiles=smiles,
            )
        return ScaffoldKey(
            key=f"acyclic:{SCAFFOLD_KEY_ALGORITHM}:{_wl_hash(graph, atoms, signature=_functional_signature(graph, atoms))}",
            basis=ScaffoldBasis.ACYCLIC_SKELETON,
            limitations=tuple(shared + [
                "the molecule has no ring, so it has no Murcko framework; the "
                "whole skeleton is used, which makes this key finer than a "
                "scaffold and so under-states similarity between acyclic "
                "analogues."]),
            source_smiles=smiles,
        )

    if inchikey:
        block = "".join(str(inchikey).split()).upper().split("-")[0]
        if len(block) == 14 and block.isalpha():
            return ScaffoldKey(
                key=f"constitution:inchikey14:{block}",
                basis=ScaffoldBasis.INCHIKEY_CONSTITUTION_BLOCK,
                limitations=(
                    "the first InChIKey block is a constitution hash, not a "
                    "scaffold: every substituent changes it, so two analogues "
                    "of one core count as two scaffolds and a novel-substrate "
                    "holdout built on it UNDER-states substrate similarity.",),
                needs_curator=(
                    "an isomeric SMILES for this substrate, so a real scaffold "
                    "can be computed"),
            )

    return ScaffoldKey(
        key=None, basis=ScaffoldBasis.UNRESOLVED,
        limitations=("no structure was recorded for this substrate.",),
        needs_curator=("an isomeric SMILES (or molfile) for this substrate; a "
                       "prose name is not a structure"),
    )


# ==========================================================================
# Facet extraction
# ==========================================================================

def record_ids(records: Sequence[Any]) -> list[str]:
    """Record ids under the convention :mod:`eagent.datalayer.lineage` uses."""
    return [lineage_record_id(r, f"row{i}") for i, r in enumerate(records)]


def _facets(obj: Any, lookup: ClusterLookup, prefix: str) -> list[str]:
    """Resolved facets of one kind, read out of the single grouping key."""
    out = []
    for facet in grouping_key(obj, lookup):
        if facet.startswith(prefix) and _UNRESOLVED_INFIX not in facet:
            out.append(facet)
    return out


def _publication_facets(obj: Any, lookup: ClusterLookup) -> list[str]:
    return [f for f in grouping_key(obj, lookup)
            if f.startswith(_PUBLICATION_PREFIXES)]


def _resolve_scaffold(obj: Any, record_id: str,
                      scaffold_lookup: Mapping[str, Any] | Callable[[Any], Any] | None,
                      prefer_rdkit: bool) -> ScaffoldKey:
    """Scaffold for one record, preferring a caller-supplied assignment.

    The override exists so a curator's scaffold assignment -- or one computed
    elsewhere with rdkit -- beats anything this module can derive, without the
    module having to guess which is better.
    """
    supplied: Any = None
    if callable(scaffold_lookup):
        supplied = scaffold_lookup(obj)
    elif isinstance(scaffold_lookup, Mapping):
        supplied = scaffold_lookup.get(record_id)
    if isinstance(supplied, ScaffoldKey):
        return supplied
    if isinstance(supplied, str) and supplied.strip():
        return ScaffoldKey(
            key=supplied.strip(),
            basis=ScaffoldBasis.CALLER_SUPPLIED,
            limitations=("supplied by the caller; this module neither derived "
                         "nor checked it, and it is not comparable with keys "
                         "from another basis.",))
    return scaffold_key(obj, prefer_rdkit=prefer_rdkit)


def source_tokens_of(
    obj: Any,
    source_lookup: Mapping[str, Any] | Callable[[Any], Any] | None = None,
    record_id: str | None = None,
) -> set[str]:
    """Resource ids this row can be traced to, normalised to lowercase tokens.

    Four places are read, because no single field carries the answer: the
    caller's ``source_lookup`` (the connector that fetched the row), the
    ``source_id`` an :class:`~eagent.schemas.record.EvidenceRef` carries for
    the resource it was read from, the ``upstream_sources`` naming what that
    resource re-curated, and a ``source_database`` attribute where one exists.
    Nothing is inferred from a URL or a record-id shape.

    ``source_id`` is what lets the audit catch the subtle case unaided: a split
    that trains on one database and tests on another is not clean when the
    second re-published the first, and before this field the audit could only
    see that if the caller supplied a lookup.
    """
    out: set[str] = set()

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                add(item)
            return
        token = " ".join(str(value).split()).strip().lower()
        if token:
            out.add(token)

    if callable(source_lookup):
        add(source_lookup(obj))
    elif isinstance(source_lookup, Mapping) and record_id is not None:
        add(source_lookup.get(record_id))

    add(getattr(obj, "source_database", None))
    seq_rec = getattr(obj, "sequence_record", None)
    if seq_rec is not None:
        add(getattr(seq_rec, "source_database", None))
    for ev in getattr(obj, "evidence", None) or []:
        add(getattr(ev, "source_id", None))
        add(getattr(ev, "upstream_sources", None))
    return out


# ==========================================================================
# The audit
# ==========================================================================

@dataclass(frozen=True)
class LeakageOverlap:
    """One shared facet, with the rows on each side that share it."""

    category: LeakageCategory
    key: str
    train_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]
    detail: str = ""

    def render(self) -> str:
        return (f"{self.category.value}: {self.key} -- train "
                f"[{', '.join(self.train_record_ids)}] and test "
                f"[{', '.join(self.test_record_ids)}]"
                + (f"; {self.detail}" if self.detail else ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "key": self.key,
            "train_record_ids": list(self.train_record_ids),
            "test_record_ids": list(self.test_record_ids),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LeakageAudit:
    """What a proposed split actually shares across the boundary.

    ``proven_clean`` is deliberately stricter than "no overlaps found": a facet
    that could not be resolved for some rows was not checked for those rows,
    and an unchecked facet is not a clean one. A split that reports
    ``proven_clean == False`` with an empty ``blocking_overlaps`` is telling the
    caller to go and supply a clustering or a structure, not that it leaked.
    """

    regime: SplitRegime | None
    n_train: int
    n_test: int
    overlaps: tuple[LeakageOverlap, ...] = ()
    unchecked: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def blocking_overlaps(self) -> tuple[LeakageOverlap, ...]:
        """Overlaps that are leakage under this regime (see
        :meth:`LeakageCategory.is_expected_under`)."""
        return tuple(o for o in self.overlaps
                     if not o.category.is_expected_under(self.regime))

    @property
    def proven_clean(self) -> bool:
        return not self.blocking_overlaps and not self.unchecked

    @property
    def has_leakage(self) -> bool:
        return bool(self.blocking_overlaps)

    def by_category(self) -> dict[str, list[LeakageOverlap]]:
        """Every category present as a key, including the empty ones.

        Zero-valued categories are kept so a report prints ``shared_scaffold:
        0`` rather than omitting the line, which reads as "not checked".
        """
        out: dict[str, list[LeakageOverlap]] = {c.value: [] for c in LeakageCategory}
        for overlap in self.overlaps:
            out[overlap.category.value].append(overlap)
        return out

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.by_category().items()}

    def render(self) -> str:
        lines = [
            f"leakage audit: {self.n_train} train row(s) vs {self.n_test} test row(s)"
            + (f" under regime {self.regime.value}" if self.regime else ""),
        ]
        for category, items in self.by_category().items():
            expected = LeakageCategory(category).is_expected_under(self.regime)
            suffix = "  (expected under this regime)" if expected and items else ""
            lines.append(f"  {category}: {len(items)}{suffix}")
            for overlap in items:
                lines.append(f"      {overlap.render()}")
        if self.unchecked:
            lines.append("  NOT CHECKED (so the split is not proven clean):")
            for item in self.unchecked:
                lines.append(f"      {item}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        lines.append("  verdict: "
                     + ("proven clean" if self.proven_clean
                        else "LEAKS" if self.has_leakage
                        else "no overlap found, but not proven clean"))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value if self.regime else None,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "counts": self.counts(),
            "overlaps": [o.to_dict() for o in self.overlaps],
            "unchecked": list(self.unchecked),
            "notes": list(self.notes),
            "proven_clean": self.proven_clean,
            "has_leakage": self.has_leakage,
        }


def _facet_overlaps(
    category: LeakageCategory,
    train_facets: Mapping[str, Sequence[str]],
    test_facets: Mapping[str, Sequence[str]],
    detail: str = "",
) -> list[LeakageOverlap]:
    """One overlap per facet value shared by the two sides."""
    train_by_key: dict[str, list[str]] = {}
    for rid, facets in train_facets.items():
        for facet in facets:
            train_by_key.setdefault(facet, []).append(rid)
    test_by_key: dict[str, list[str]] = {}
    for rid, facets in test_facets.items():
        for facet in facets:
            test_by_key.setdefault(facet, []).append(rid)
    out: list[LeakageOverlap] = []
    for facet in sorted(set(train_by_key) & set(test_by_key)):
        out.append(LeakageOverlap(
            category=category,
            key=facet,
            train_record_ids=tuple(sorted(train_by_key[facet])),
            test_record_ids=tuple(sorted(test_by_key[facet])),
            detail=detail,
        ))
    return out


def _recuration_overlaps(
    train_sources: Mapping[str, set[str]],
    test_sources: Mapping[str, set[str]],
    registry: Any | None,
) -> tuple[list[LeakageOverlap], list[str]]:
    """Detect "train on A, test on B" where B re-curated A.

    Two detectors, because the evidence can arrive two ways:

    * **Directly.** Both sides declare the same ``upstream_sources``. One
      measurement re-integrated by two resources is one measurement, so the
      sides are not independent whatever their accessions say.
    * **Through the registry.** With a
      :class:`~eagent.datalayer.registry.SourceRegistry`, the sources on the
      two sides are collapsed with
      :meth:`~eagent.datalayer.registry.SourceRegistry.independent_source_groups`,
      which already follows ``derived_from`` transitively. A group holding a
      train-side source and a test-side source means the split is between two
      spellings of one lineage.
    """
    notes: list[str] = []
    out: list[LeakageOverlap] = []

    direct = _facet_overlaps(
        LeakageCategory.RECURATED_SOURCE,
        {rid: sorted(v) for rid, v in train_sources.items()},
        {rid: sorted(v) for rid, v in test_sources.items()},
        detail=("both sides trace to this resource, so the rows are "
                "re-curations of one lineage rather than two independent "
                "resources"),
    )
    out.extend(direct)

    if registry is None:
        if train_sources or test_sources:
            notes.append(
                "no SourceRegistry was supplied, so derived_from edges between "
                "the two sides' databases were not followed; 'train on A, test "
                "on B' remains unproven as an independent split")
        return out, notes

    known: dict[str, set[str]] = {}
    unknown: set[str] = set()
    for side, mapping in (("train", train_sources), ("test", test_sources)):
        for rid, tokens in mapping.items():
            for token in tokens:
                if token in registry:
                    known.setdefault(token, set()).add(f"{side}:{rid}")
                else:
                    unknown.add(token)
    if unknown:
        notes.append(
            "source id(s) not in the registry, so their lineage was not "
            f"checked: {', '.join(sorted(unknown))}")
    if not known:
        return out, notes

    already = {o.key for o in direct}
    for group in registry.independent_source_groups(sorted(known)):
        train_rows = sorted({m.split(":", 1)[1] for token in group
                             for m in known.get(token, set())
                             if m.startswith("train:")})
        test_rows = sorted({m.split(":", 1)[1] for token in group
                            for m in known.get(token, set())
                            if m.startswith("test:")})
        if not train_rows or not test_rows:
            continue
        key = "+".join(group)
        if len(group) == 1 and group[0] in already:
            continue
        out.append(LeakageOverlap(
            category=LeakageCategory.RECURATED_SOURCE,
            key=key,
            train_record_ids=tuple(train_rows),
            test_record_ids=tuple(test_rows),
            detail=("these resources share a derived_from lineage in the "
                    "registry, so splitting train and test between them is not "
                    "a split: the test rows may be the training rows "
                    "re-curated"),
        ))
    return out, notes


def audit_leakage(
    train: Sequence[Any],
    test: Sequence[Any],
    *,
    regime: SplitRegime | None = None,
    sequence_cluster_lookup: ClusterLookup = None,
    scaffold_lookup: Mapping[str, Any] | Callable[[Any], Any] | None = None,
    registry: Any | None = None,
    source_lookup: Mapping[str, Any] | Callable[[Any], Any] | None = None,
    prefer_rdkit: bool = True,
) -> LeakageAudit:
    """Report every overlap between two folds, by category.

    Categories: shared sequence cluster, shared substrate scaffold, shared
    publication, shared parent lineage, shared leakage-safe group (the
    transitive closure of the first three plus experiment activity), and a
    re-curated source pair found through the registry's ``derived_from`` edges.

    This is the function that lets a split be *proven* clean. It is intended to
    be run on splits this module did not make -- an externally supplied
    train/test division, or the "train on database A, test on database B"
    arrangement that looks like a split and is not one.
    """
    train_list, test_list = list(train), list(test)
    train_ids = record_ids(train_list)
    test_ids = [lineage_record_id(r, f"row{i + len(train_list)}")
                for i, r in enumerate(test_list)]

    overlaps: list[LeakageOverlap] = []
    unchecked: list[str] = []
    notes: list[str] = []

    for side, side_ids in (("train", train_ids), ("test", test_ids)):
        repeated = sorted({i for i in side_ids if side_ids.count(i) > 1})
        if repeated:
            unchecked.append(
                f"{side} carries record id(s) {repeated} more than once, so "
                f"facets were compared for only one row of each; give every "
                f"row a distinct record_id before trusting this audit")

    # -- the one grouping key, closed transitively -------------------------
    verdict = split_leakage(train_list, test_list, sequence_cluster_lookup)
    if not verdict.ok:
        assignment = leakage_safe_groups(train_list + test_list,
                                         sequence_cluster_lookup)
        for group in verdict.shared_group_ids:
            overlaps.append(LeakageOverlap(
                category=LeakageCategory.SHARED_SPLIT_GROUP,
                key=group,
                train_record_ids=tuple(sorted(
                    r for r in train_ids if assignment.get(r) == group)),
                test_record_ids=tuple(sorted(
                    r for r in test_ids if assignment.get(r) == group)),
                detail=("one leakage-safe group spans the boundary: these rows "
                        "share a publication, an experiment activity, a parent "
                        "lineage or a sequence cluster transitively"),
            ))

    # -- the individual facets, read out of the same grouping key ----------
    overlaps.extend(_facet_overlaps(
        LeakageCategory.SHARED_SEQUENCE_CLUSTER,
        {rid: _facets(obj, sequence_cluster_lookup, _CLUSTER_PREFIX)
         for rid, obj in zip(train_ids, train_list)},
        {rid: _facets(obj, sequence_cluster_lookup, _CLUSTER_PREFIX)
         for rid, obj in zip(test_ids, test_list)},
        detail="the same sequence cluster appears on both sides",
    ))
    overlaps.extend(_facet_overlaps(
        LeakageCategory.SHARED_PARENT_LINEAGE,
        {rid: _facets(obj, sequence_cluster_lookup, _LINEAGE_PREFIX)
         for rid, obj in zip(train_ids, train_list)},
        {rid: _facets(obj, sequence_cluster_lookup, _LINEAGE_PREFIX)
         for rid, obj in zip(test_ids, test_list)},
        detail=("a parent sequence and one of its variants, or two variants of "
                "one parent, sit on opposite sides"),
    ))
    overlaps.extend(_facet_overlaps(
        LeakageCategory.SHARED_PUBLICATION,
        {rid: _publication_facets(obj, sequence_cluster_lookup)
         for rid, obj in zip(train_ids, train_list)},
        {rid: _publication_facets(obj, sequence_cluster_lookup)
         for rid, obj in zip(test_ids, test_list)},
        detail=("rows from one paper share its construct, assay, calibration "
                "and analyst, so they are not independent samples"),
    ))

    # -- scaffolds ---------------------------------------------------------
    bases: set[ScaffoldBasis] = set()
    scaffolds: dict[str, list[str]] = {}
    for side_ids, side_objs in ((train_ids, train_list), (test_ids, test_list)):
        for rid, obj in zip(side_ids, side_objs):
            key = _resolve_scaffold(obj, rid, scaffold_lookup, prefer_rdkit)
            if key.resolved:
                scaffolds[rid] = [key.key]           # type: ignore[list-item]
                bases.add(key.basis)
            else:
                scaffolds[rid] = []
                message = (f"{rid}: substrate scaffold unresolved -- "
                           f"{key.needs_curator or 'no structure recorded'}")
                if regime is not None and not regime.holds_out_substrates:
                    # This regime shares substrates on purpose, so an unknown
                    # scaffold does not weaken the claim it makes; it is still
                    # reported, as a note rather than as an unchecked facet.
                    notes.append(message)
                else:
                    unchecked.append(message)
    overlaps.extend(_facet_overlaps(
        LeakageCategory.SHARED_SCAFFOLD,
        {rid: scaffolds[rid] for rid in train_ids},
        {rid: scaffolds[rid] for rid in test_ids},
        detail="the same substrate scaffold appears on both sides",
    ))
    structural = {b for b in bases if b.is_structural}
    if len(structural) > 1 or (structural and
                               ScaffoldBasis.INCHIKEY_CONSTITUTION_BLOCK in bases):
        notes.append(
            "scaffold keys in this split come from more than one basis "
            f"({', '.join(sorted(b.value for b in bases))}); keys from "
            "different bases are different key spaces and an overlap between "
            "them cannot be detected by comparing strings")

    # -- re-curation -------------------------------------------------------
    train_sources = {rid: source_tokens_of(obj, source_lookup, rid)
                     for rid, obj in zip(train_ids, train_list)}
    test_sources = {rid: source_tokens_of(obj, source_lookup, rid)
                    for rid, obj in zip(test_ids, test_list)}
    recuration, recuration_notes = _recuration_overlaps(
        train_sources, test_sources, registry)
    overlaps.extend(recuration)
    notes.extend(recuration_notes)

    if regime is not None and regime.holds_out_sequences \
            and sequence_cluster_lookup is None:
        unchecked.append(
            "no sequence clustering was supplied, so 'test clusters absent "
            "from training' was not checked; a novel-enzyme claim needs one")

    return LeakageAudit(
        regime=regime,
        n_train=len(train_list),
        n_test=len(test_list),
        overlaps=tuple(overlaps),
        unchecked=tuple(unchecked),
        notes=tuple(notes),
    )


# ==========================================================================
# The split
# ==========================================================================

@dataclass(frozen=True)
class GroupedSplit:
    """One grouped train/test division, with the audit that proves it.

    The audit is a field rather than something the caller is trusted to run,
    because a split that is never audited is a split that is assumed clean --
    and the whole point of this module is that the assumption is usually false.
    """

    regime: SplitRegime
    train_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]
    excluded_record_ids: tuple[str, ...]
    exclusion_reasons: Mapping[str, str]
    unit_of_record: Mapping[str, str]
    test_units: tuple[str, ...]
    held_out_sequence_clusters: tuple[str, ...]
    held_out_scaffolds: tuple[str, ...]
    audit: LeakageAudit
    shortfall_reason: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def n_train(self) -> int:
        return len(self.train_record_ids)

    @property
    def n_test(self) -> int:
        return len(self.test_record_ids)

    @property
    def is_usable(self) -> bool:
        """Both folds non-empty and no overlap that this regime forbids."""
        return bool(self.train_record_ids and self.test_record_ids
                    and not self.audit.has_leakage)

    def partition(self, records: Sequence[Any]) -> tuple[list[Any], list[Any]]:
        """``(train_rows, test_rows)`` for the same sequence the split was made from.

        The excluded rows appear in neither list. They are not a third fold to
        be quietly folded into training: each was excluded because a facet
        could not be resolved, and training on it could leak into the test
        fold through the facet nobody could check.
        """
        ids = record_ids(records)
        train_set, test_set = set(self.train_record_ids), set(self.test_record_ids)
        train = [r for r, rid in zip(records, ids) if rid in train_set]
        test = [r for r, rid in zip(records, ids) if rid in test_set]
        return train, test

    def render(self) -> str:
        lines = [
            f"{self.regime.value} split: {self.n_train} train / {self.n_test} test"
            f" / {len(self.excluded_record_ids)} excluded",
            f"claim this supports: {self.regime.claim()}",
            f"test units held out: {len(self.test_units)}",
        ]
        if self.held_out_sequence_clusters:
            lines.append("held-out sequence clusters: "
                         + ", ".join(self.held_out_sequence_clusters))
        if self.held_out_scaffolds:
            lines.append("held-out scaffolds: " + ", ".join(self.held_out_scaffolds))
        for rid in self.excluded_record_ids:
            lines.append(f"excluded {rid}: {self.exclusion_reasons.get(rid, '')}")
        if self.shortfall_reason:
            lines.append(f"SHORT: {self.shortfall_reason}")
        for note in self.notes:
            lines.append(f"note: {note}")
        lines.append(self.audit.render())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "claim": self.regime.claim(),
            "train_record_ids": list(self.train_record_ids),
            "test_record_ids": list(self.test_record_ids),
            "excluded_record_ids": list(self.excluded_record_ids),
            "exclusion_reasons": dict(self.exclusion_reasons),
            "test_units": list(self.test_units),
            "held_out_sequence_clusters": list(self.held_out_sequence_clusters),
            "held_out_scaffolds": list(self.held_out_scaffolds),
            "shortfall_reason": self.shortfall_reason,
            "notes": list(self.notes),
            "audit": self.audit.to_dict(),
            "is_usable": self.is_usable,
        }


class _Union:
    """Union-find over split units. Deterministic merge direction, as in lineage."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self._parent[hi] = lo


def grouped_split(
    records: Sequence[Any],
    regime: SplitRegime,
    *,
    test_fraction: float = 0.25,
    sequence_cluster_lookup: ClusterLookup = None,
    scaffold_lookup: Mapping[str, Any] | Callable[[Any], Any] | None = None,
    registry: Any | None = None,
    source_lookup: Mapping[str, Any] | Callable[[Any], Any] | None = None,
    prefer_rdkit: bool = True,
    seed: int = 0,
) -> GroupedSplit:
    """Split records into folds whose unit is a group, never a row.

    The unit is the leakage-safe group of
    :func:`eagent.datalayer.lineage.leakage_safe_groups`; for the substrate
    regimes it is further closed over the substrate scaffold, so neither a
    lineage group nor a scaffold can straddle the boundary (see the module
    docstring for why that is stricter than common practice, and what it
    costs).

    Determinism: units are ordered by ``sha256(seed:unit)``, so the same inputs
    and seed give the same folds and a different seed gives a genuinely
    different draw -- without the first-listed or the largest unit being
    systematically the test fold.

    Parameters
    ----------
    test_fraction:
        Target share of *placeable* rows in the test fold. Whole units are
        taken, so the realised share is rarely exact and is never achieved by
        breaking a unit.
    sequence_cluster_lookup:
        Sequence identity (``sequence_sha256``, or whatever
        :func:`~eagent.datalayer.lineage.grouping_key` resolves for the row) to
        cluster id, as a mapping or a callable.
    scaffold_lookup:
        Record id to a :class:`ScaffoldKey` or key string, as a mapping, or a
        callable taking the record. A supplied key wins over anything this
        module can derive, so a curator's assignment is not overridden.
    registry, source_lookup:
        Passed through to :func:`audit_leakage` for the re-curation check:
        ``source_lookup`` maps a record id (or the record, if callable) to the
        registry id of the resource it came from.

    Raises
    ------
    SplitNotPossibleError
        When ``test_fraction`` is outside ``(0, 1)``, or when a regime that
        claims unseen sequence clusters is requested with no clustering: that
        claim would be uncheckable, and an uncheckable claim is the thing this
        module exists to stop.
    """
    if not 0.0 < test_fraction < 1.0:
        raise SplitNotPossibleError(
            f"test_fraction must lie strictly between 0 and 1, got {test_fraction}")
    if regime.holds_out_sequences and sequence_cluster_lookup is None:
        raise SplitNotPossibleError(
            f"regime '{regime.value}' claims that test sequence clusters are "
            f"absent from training, and no sequence_cluster_lookup was "
            f"supplied. Without a clustering the claim cannot be checked, and "
            f"an unchecked leakage claim is worse than no claim: supply the "
            f"clustering (for example mmseqs2 cluster ids) or use "
            f"SplitRegime.NOVEL_SUBSTRATE.")

    rows = list(records)
    ids = record_ids(rows)
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise SplitNotPossibleError(
            f"record id(s) {repeated} appear more than once. A split is "
            f"addressed by record id, so a repeated id silently drops one row "
            f"from the fold map and the audit would then check a row that is "
            f"not the one in the fold")
    by_id = dict(zip(ids, rows))
    assignment = leakage_safe_groups(rows, sequence_cluster_lookup)

    notes: list[str] = []
    excluded: list[str] = []
    reasons: dict[str, str] = {}

    # -- 1. resolve the facets each regime needs, excluding what it cannot --
    scaffolds: dict[str, ScaffoldKey] = {}
    bases: set[ScaffoldBasis] = set()
    for rid in ids:
        key = _resolve_scaffold(by_id[rid], rid, scaffold_lookup, prefer_rdkit)
        scaffolds[rid] = key
        if key.resolved:
            bases.add(key.basis)
    if regime.holds_out_substrates:
        for rid in ids:
            if not scaffolds[rid].resolved:
                excluded.append(rid)
                reasons[rid] = (
                    "substrate scaffold unresolved, so this row cannot be shown "
                    "absent from the other fold. A curator must supply "
                    + (scaffolds[rid].needs_curator or "an isomeric SMILES"))
    if regime.holds_out_sequences:
        for rid in ids:
            if any(f.startswith(_CLUSTER_PREFIX) and _UNRESOLVED_INFIX in f
                   for f in grouping_key(by_id[rid], sequence_cluster_lookup)):
                reason = ("sequence cluster unresolved, so this row cannot be "
                          "shown absent from the other fold. A curator must "
                          "supply a cluster id for its sequence")
                if rid in reasons:
                    reasons[rid] = reasons[rid] + " ALSO: " + reason
                else:
                    excluded.append(rid)
                    reasons[rid] = reason
    placeable = [rid for rid in ids if rid not in reasons]

    # -- 2. build the split units ------------------------------------------
    # The unit is normally the lineage group closed over the scaffold, so
    # neither can straddle the boundary. The shared-enzyme regime is the one
    # exception: its unit is the scaffold alone, which is precisely what lets
    # one enzyme's measurements land on both sides. That is the weaker
    # question, chosen deliberately, and the regime's own claim() says so.
    if regime.permits_shared_enzymes:
        unit_of_record: dict[str, str] = {}
        for rid in placeable:
            scaffold = scaffolds[rid].key
            # A placeable row under this regime always has a scaffold: the
            # unresolved ones were excluded above, since holds_out_substrates
            # is true here.
            unit_of_record[rid] = f"scaffold::{scaffold}"
        notes.append(
            "regime 'shared_enzyme_novel_substrate': units are substrate "
            "scaffolds alone, so an enzyme measured on both a training and a "
            "test scaffold appears on both sides. That is this regime's "
            "design and the reason its claim is narrower; a score here is "
            "about a new substrate for a known enzyme and is not evidence "
            "about a new enzyme")
    else:
        union = _Union()
        for rid in placeable:
            group = assignment[rid]
            union.add(group)
            if regime.holds_out_substrates:
                scaffold = scaffolds[rid].key
                if scaffold:
                    union.union(group, f"scaffold::{scaffold}")
        unit_of_record = {rid: union.find(assignment[rid]) for rid in placeable}
    members: dict[str, list[str]] = {}
    for rid in placeable:
        members.setdefault(unit_of_record[rid], []).append(rid)

    # -- 3. deterministic unit order, then whole units into the test fold ---
    order = sorted(members, key=lambda u: (sha256_text(f"{seed}:{u}"), u))
    target = max(1, int(math.ceil(test_fraction * len(placeable)))) \
        if placeable else 0
    test_units: list[str] = []
    test_ids: list[str] = []
    for unit in order:
        if len(test_ids) >= target:
            break
        if len(test_ids) + len(members[unit]) >= len(placeable):
            # Taking this unit would leave no training rows at all.
            continue
        test_units.append(unit)
        test_ids.extend(sorted(members[unit]))
    test_set = set(test_ids)
    train_ids = [rid for rid in placeable if rid not in test_set]

    shortfall: str | None = None
    if not test_ids or not train_ids:
        shortfall = (
            f"no usable split: {len(placeable)} placeable row(s) collapsed into "
            f"{len(members)} unit(s) under the {regime.value} closure, and no "
            f"combination of whole units leaves rows on both sides. The folds "
            f"are reported empty rather than formed by breaking a unit -- "
            f"breaking one would put a publication, a parent lineage or a "
            f"scaffold on both sides of the boundary.")
    elif len(test_ids) < target:
        shortfall = (
            f"test fold holds {len(test_ids)} of the {target} row(s) asked for: "
            f"units are taken whole, and the remaining units are too large to "
            f"add without emptying the training fold.")

    if len(bases) > 1:
        notes.append(
            "substrate scaffolds in this dataset were computed on more than "
            f"one basis ({', '.join(sorted(b.value for b in bases))}); keys "
            "from different bases are different key spaces and were not "
            "compared across bases")
    if regime.holds_out_substrates:
        weak = sorted({rid for rid in placeable
                       if not scaffolds[rid].is_true_scaffold})
        if weak:
            notes.append(
                f"{len(weak)} row(s) were held out on a constitution hash "
                f"rather than a scaffold, which under-states substrate "
                f"similarity; first: {weak[0]}")
    if regime.holds_out_substrates and regime.holds_out_sequences is False:
        notes.append(
            "the lineage-group invariant means this novel-substrate split is "
            "also enzyme-disjoint: an enzyme measured on a held-out scaffold "
            "moved to the test fold with its other measurements. That is "
            "stricter than the common practice of keeping the same enzymes on "
            "both sides, and it is strict in the direction that does not "
            "inflate the score")

    train_rows = [by_id[rid] for rid in train_ids]
    test_rows = [by_id[rid] for rid in test_ids]
    audit = audit_leakage(
        train_rows, test_rows,
        regime=regime,
        sequence_cluster_lookup=sequence_cluster_lookup,
        scaffold_lookup=scaffold_lookup,
        registry=registry,
        source_lookup=source_lookup,
        prefer_rdkit=prefer_rdkit,
    )

    held_clusters = sorted({f for rid in test_ids
                            for f in _facets(by_id[rid], sequence_cluster_lookup,
                                             _CLUSTER_PREFIX)})
    held_scaffolds = sorted({scaffolds[rid].key for rid in test_ids
                             if scaffolds[rid].key})

    return GroupedSplit(
        regime=regime,
        train_record_ids=tuple(train_ids),
        test_record_ids=tuple(test_ids),
        excluded_record_ids=tuple(sorted(set(excluded))),
        exclusion_reasons=dict(reasons),
        unit_of_record=unit_of_record,
        test_units=tuple(test_units),
        held_out_sequence_clusters=tuple(held_clusters),
        held_out_scaffolds=tuple(held_scaffolds),
        audit=audit,
        shortfall_reason=shortfall,
        notes=tuple(notes),
    )
