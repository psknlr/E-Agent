"""Geometric measurement. Measurement only -- never judgement.

Every function here answers "what is the number?" and none of them answers
"is the number good?". That separation is the point of the module. The
moment a distance helper also owns a cutoff, the cutoff becomes invisible: it
stops being a property of a sourced
:class:`~eagent.schemas.templates.CatalyticTemplate`, calibrated on known
systems, and becomes a magic number inside a utility that nobody reviews.

So:

* Windows come from :class:`~eagent.schemas.templates.GeometryConstraint`
  objects, which carry their own ``calibrated_on`` provenance and decide
  pass/fail through ``GeometryConstraint.satisfied_by``. Nothing in this
  module compares a measured value against a literature number.
* The two module-level numbers that *are* here --
  :data:`DEFAULT_VDW_OVERLAP_TOLERANCE_A` and the van der Waals radius table
  -- exist only to localise a steric problem for QC. They are a screen, not
  an energy, and both are documented as needing per-family calibration.
* A quantity that cannot be measured comes back as ``None``. It never comes
  back as a default, an average, or a value from the nearest resolvable atom.
  :func:`measure_constraint` is the load-bearing example: an unresolvable
  role token yields ``None``, and the caller must then report the constraint
  as *unevaluated*, which is a different thing from *failed*.

Pure Python throughout: numpy is not installed, and for the handful of atoms
involved in a catalytic measurement the vector maths is cheap anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..errors import EAgentError, TemplateError
from ..schemas.templates import GeometryConstraint
from .structure_io import Atom, Residue, Structure

__all__ = [
    "GeometryError",
    "Point",
    "as_point",
    "distance",
    "angle",
    "dihedral",
    "centroid",
    "min_distance",
    "VDW_RADII_A",
    "DEFAULT_VDW_OVERLAP_TOLERANCE_A",
    "ClashPair",
    "clash_pairs",
    "clash_count",
    "pocket_shell",
    "ResolvedAtom",
    "RoleResolver",
    "resolve_role",
    "measure_constraint",
    "measure_all",
    "hydride_transfer_distance",
    "burgi_dunitz_angle",
    "BURGI_DUNITZ_LITERATURE_DEG",
]


class GeometryError(EAgentError):
    """A measurement cannot be made because the inputs are malformed.

    Distinct from "cannot be measured because an atom is missing", which is
    reported as ``None``. This is for a caller mistake -- degenerate
    coordinates, an unknown element in a clash screen, a point that is not a
    point -- where returning ``None`` would hide a bug behind a plausible
    "unresolvable" result.
    """


Point = tuple[float, float, float]


def as_point(obj: Atom | Point | Sequence[float] | Any) -> Point:
    """Coerce an :class:`~eagent.science.structure_io.Atom` or an (x, y, z) to a tuple.

    Accepting both keeps call sites honest: a measurement against a template
    atom and a measurement against an arbitrary probe point are the same
    operation, and forcing the caller to wrap a tuple in a fake ``Atom`` would
    invite fabricating the other ``Atom`` fields (element, occupancy) that it
    does not know.
    """
    if obj is None:
        raise GeometryError("cannot measure against a missing point (None)")
    if isinstance(obj, Atom):
        return (obj.x, obj.y, obj.z)
    x = getattr(obj, "x", None)
    if x is not None and hasattr(obj, "y") and hasattr(obj, "z"):
        return (float(x), float(obj.y), float(obj.z))
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        try:
            return (float(obj[0]), float(obj[1]), float(obj[2]))
        except (TypeError, ValueError):
            raise GeometryError(f"not a numeric 3-vector: {obj!r}") from None
    raise GeometryError(
        f"expected an Atom or an (x, y, z) triple, got {type(obj).__name__}"
    )


# ---------------------------------------------------------------------------
# primitive vector maths
# ---------------------------------------------------------------------------


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Point) -> float:
    return math.sqrt(_dot(a, a))


def _scale(a: Point, k: float) -> Point:
    return (a[0] * k, a[1] * k, a[2] * k)


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------


def distance(a: Atom | Point, b: Atom | Point) -> float:
    """Euclidean distance in angstroms.

    Angstroms because every coordinate file this project reads is in
    angstroms; no unit conversion happens anywhere in this module, so a
    nanometre-scale input would produce a silently wrong number. Callers that
    might see nanometres must convert before they get here.
    """
    pa, pb = as_point(a), as_point(b)
    return math.sqrt(
        (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
    )


def angle(a: Atom | Point, b: Atom | Point, c: Atom | Point) -> float:
    """Angle a-b-c in degrees, with ``b`` at the vertex. Range 0..180.

    The cosine is clamped to [-1, 1] before ``acos``. Without the clamp,
    floating-point error on a perfectly linear arrangement gives a cosine of
    -1.0000000000000002 and raises ``ValueError`` -- which in a catalytic
    geometry check would turn an ideal collinear hydride-transfer trajectory
    into a crash or, worse, into a swallowed exception reported as "angle
    unmeasurable".
    """
    pa, pb, pc = as_point(a), as_point(b), as_point(c)
    v1 = _sub(pa, pb)
    v2 = _sub(pc, pb)
    n1, n2 = _norm(v1), _norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        raise GeometryError(
            "angle is undefined: two of the three points are coincident"
        )
    cos_t = _dot(v1, v2) / (n1 * n2)
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


def dihedral(a: Atom | Point, b: Atom | Point, c: Atom | Point,
             d: Atom | Point) -> float:
    """Torsion a-b-c-d in degrees, IUPAC sign convention, range (-180, 180].

    Sign convention matters here in a way it does not for a distance: the
    difference between +90 and -90 is the difference between the two
    prochiral faces of a ketone, which is the difference between the (R) and
    the (S) alcohol. A helper that returned ``abs(dihedral)`` would make
    stereochemistry unmeasurable while still returning a confident number.

    Looking from b towards c, a positive value means d is rotated clockwise
    relative to a.
    """
    pa, pb, pc, pd = as_point(a), as_point(b), as_point(c), as_point(d)
    b1 = _sub(pb, pa)
    b2 = _sub(pc, pb)
    b3 = _sub(pd, pc)
    nb2 = _norm(b2)
    if nb2 == 0.0:
        raise GeometryError(
            "dihedral is undefined: the central atoms b and c are coincident"
        )
    n1 = _cross(b1, b2)
    n2 = _cross(b2, b3)
    if _norm(n1) == 0.0 or _norm(n2) == 0.0:
        raise GeometryError(
            "dihedral is undefined: three consecutive points are collinear"
        )
    m1 = _cross(n1, _scale(b2, 1.0 / nb2))
    x = _dot(n1, n2)
    y = _dot(m1, n2)
    return math.degrees(math.atan2(y, x))


def centroid(atoms: Iterable[Atom | Point]) -> Point:
    """Unweighted mean position.

    Unweighted, and named ``centroid`` rather than ``center_of_mass``, because
    no mass table is applied. A centroid is a convenient handle for a pocket
    or a ring; it is explicitly *not* a catalytic reference point. A distance
    from a substrate centroid to a protein centroid tells you nothing about
    whether a reaction can occur -- see
    :class:`~eagent.schemas.chem.ReactiveAtoms`, which exists so that
    distances are defined between the atoms the chemistry touches.
    """
    pts = [as_point(a) for a in atoms]
    if not pts:
        raise GeometryError("centroid of an empty atom set is undefined")
    n = float(len(pts))
    return (
        sum(p[0] for p in pts) / n,
        sum(p[1] for p in pts) / n,
        sum(p[2] for p in pts) / n,
    )


def min_distance(set_a: Iterable[Atom | Point],
                 set_b: Iterable[Atom | Point]) -> float:
    """Closest approach between two atom sets, in angstroms.

    Raises on an empty set rather than returning ``inf``. ``inf`` compares as
    "very far away" and would quietly pass a "no clash" or "outside the
    shell" test for a set that is empty because the cofactor failed to parse.
    """
    pa = [as_point(a) for a in set_a]
    pb = [as_point(b) for b in set_b]
    if not pa or not pb:
        raise GeometryError(
            "min_distance needs two non-empty atom sets; an empty set usually "
            "means a selection matched nothing (check chain ids and ligand codes)"
        )
    best = float("inf")
    for p in pa:
        for q in pb:
            d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
            if d2 < best:
                best = d2
    return math.sqrt(best)


# ---------------------------------------------------------------------------
# steric screen
# ---------------------------------------------------------------------------

#: Van der Waals radii in angstroms, from Bondi, *J. Phys. Chem.* 68 (1964)
#: 441-451, with the hydrogen value left at Bondi's 1.20 rather than the
#: 1.09 of Rowland & Taylor (1996) -- mixing the two sources would make the
#: screen inconsistent, and most structures here carry no hydrogens anyway.
#:
#: SOURCE ASSUMPTION, and the reason this table is deliberately incomplete:
#: Bondi derived radii from crystal packing of *non-bonded* contacts and
#: explicitly did not publish values for most transition metals. Fe, Mn and
#: Co are therefore absent. Rather than substituting a plausible 2.0 A, an
#: atom whose element is not in this table is reported as *unscreened* by
#: :func:`clash_pairs` and makes :func:`clash_count` raise unless the caller
#: opts in. An invented metal radius would decide, silently, whether a
#: metalloenzyme's active site looks occupied or clashing.
#:
#: THIS IS A SCREEN, NOT AN ENERGY. A "clash" here means two hard spheres
#: overlap by more than a tolerance. It does not account for a covalent or
#: coordinate bond, a hydrogen bond (whose heavy-atom separation of ~2.8 A is
#: well inside the O/N sphere sum of ~3.07 A), or any attractive term. In
#: particular, a metal and its coordinating ligands will register as clashing
#: under this screen and must be excluded by the caller via ``pair_filter``.
#: The output localises a steric problem for a human or for a real force
#: field; it is never evidence that a pose is energetically bad.
VDW_RADII_A: dict[str, float] = {
    "H": 1.20, "D": 1.20,
    "HE": 1.40,
    "LI": 1.82, "BE": 1.53, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "NE": 1.54,
    "NA": 2.27, "MG": 1.73, "AL": 1.84, "SI": 2.10, "P": 1.80, "S": 1.80,
    "CL": 1.75, "AR": 1.88,
    "K": 2.75, "CA": 2.31,
    "NI": 1.63, "CU": 1.40, "ZN": 1.39, "GA": 1.87, "GE": 2.11, "AS": 1.85,
    "SE": 1.90, "BR": 1.85, "KR": 2.02,
    "PD": 1.63, "AG": 1.72, "CD": 1.58, "IN": 1.93, "SN": 2.17, "TE": 2.06,
    "I": 1.98, "XE": 2.16,
    "PT": 1.75, "AU": 1.66, "HG": 1.55, "TL": 1.96, "PB": 2.02,
}

#: Default overlap a pair must exceed before it counts as a clash, in
#: angstroms. 0.5 A is the common soft-clash convention used by structure
#: validation tools; it is *not* derived from this project's data.
#:
#: NEEDS PER-FAMILY CALIBRATION. The right tolerance depends on the structure
#: source (a 1.5 A crystal structure tolerates far less overlap than a
#: predicted model with a 70-pLDDT loop) and on whether hydrogens are present.
#: Treat it as a starting point for localising steric problems, and record
#: the value actually used in provenance.
DEFAULT_VDW_OVERLAP_TOLERANCE_A: float = 0.5


@dataclass(frozen=True)
class ClashPair:
    """One overlapping atom pair, kept with its numbers so it can be checked."""

    atom_a: Atom
    atom_b: Atom
    distance_A: float
    sum_radii_A: float

    @property
    def overlap_A(self) -> float:
        """How far inside the hard-sphere contact the pair sits."""
        return self.sum_radii_A - self.distance_A

    def describe(self) -> str:
        """Report line carrying both atoms and both numbers, so a reader can
        re-derive the verdict instead of trusting the word "clash"."""
        return (
            f"{self.atom_a.label()} -- {self.atom_b.label()}: "
            f"{self.distance_A:.2f} A vs vdW sum {self.sum_radii_A:.2f} A "
            f"(overlap {self.overlap_A:.2f} A)"
        )


def _radius(a: Atom) -> float | None:
    return VDW_RADII_A.get(a.element.strip().upper())


def clash_pairs(
    structure_atoms: Iterable[Atom],
    ligand_atoms: Iterable[Atom],
    vdw_overlap_tolerance: float = DEFAULT_VDW_OVERLAP_TOLERANCE_A,
    pair_filter: Callable[[Atom, Atom], bool] | None = None,
) -> tuple[list[ClashPair], list[Atom]]:
    """Return (clashing pairs, atoms that could not be screened).

    The second element is the honest part. An atom whose element has no Bondi
    radius is listed rather than skipped, so a caller cannot read "0 clashes"
    from a pose whose catalytic iron was never tested.

    Pairs are skipped when the same atom appears in both sets (compared by
    identity and then by serial), because a ligand that is also part of the
    structure would otherwise clash with itself at distance 0. ``pair_filter``
    lets the caller drop further pairs -- most importantly metal-to-ligating-
    atom pairs, which are coordinate bonds and are expected to sit well inside
    the van der Waals sum.
    """
    sa = list(structure_atoms)
    la = list(ligand_atoms)
    if vdw_overlap_tolerance < 0.0:
        raise GeometryError(
            "vdw_overlap_tolerance must be >= 0; a negative tolerance would "
            "count non-contacting pairs as clashes"
        )

    unscreened: list[Atom] = []
    seen_unscreened: set[int] = set()
    for a in sa + la:
        if _radius(a) is None and id(a) not in seen_unscreened:
            seen_unscreened.add(id(a))
            unscreened.append(a)

    ligand_ids = {id(a) for a in la}
    ligand_serials = {a.serial for a in la if a.serial >= 0}

    out: list[ClashPair] = []
    for lig in la:
        rl = _radius(lig)
        if rl is None:
            continue
        for prot in sa:
            if id(prot) in ligand_ids:
                continue
            if prot.serial >= 0 and prot.serial in ligand_serials:
                continue
            rp = _radius(prot)
            if rp is None:
                continue
            if pair_filter is not None and not pair_filter(prot, lig):
                continue
            limit = rp + rl - vdw_overlap_tolerance
            d = distance(prot, lig)
            if d < limit:
                out.append(ClashPair(prot, lig, d, rp + rl))
    return out, unscreened


def clash_count(
    structure_atoms: Iterable[Atom],
    ligand_atoms: Iterable[Atom],
    vdw_overlap_tolerance: float = DEFAULT_VDW_OVERLAP_TOLERANCE_A,
    allow_unscreened: bool = False,
    pair_filter: Callable[[Atom, Atom], bool] | None = None,
) -> int:
    """Number of hard-sphere overlaps between a ligand and its surroundings.

    A screen for localising steric problems, not an energy -- see the note on
    :data:`VDW_RADII_A`. The returned integer is only meaningful alongside the
    tolerance that produced it, so record both in provenance.

    Raises :class:`GeometryError` when any atom's element has no tabulated
    radius, unless ``allow_unscreened=True``. Defaulting to silence here would
    let a count of 0 mean either "no clashes" or "nothing was tested"; the
    caller has to say which it is willing to accept, and if it accepts
    unscreened atoms it should use :func:`clash_pairs` to report them.
    """
    pairs, unscreened = clash_pairs(
        structure_atoms, ligand_atoms, vdw_overlap_tolerance, pair_filter
    )
    if unscreened and not allow_unscreened:
        elements = sorted({a.element.strip().upper() for a in unscreened})
        raise GeometryError(
            f"no tabulated van der Waals radius for element(s) {elements} "
            f"({len(unscreened)} atom(s), e.g. {unscreened[0].label()}); these "
            f"atoms were NOT screened. Pass allow_unscreened=True to accept a "
            f"partial screen, and report the gap."
        )
    return len(pairs)


# ---------------------------------------------------------------------------
# pocket scope
# ---------------------------------------------------------------------------


def pocket_shell(
    structure: Structure,
    ligand_atoms: Iterable[Atom],
    min_angstrom: float,
    max_angstrom: float,
    include_water: bool = False,
    include_hetatm: bool = False,
) -> list[Residue]:
    """Residues with at least one heavy atom in the shell around a ligand.

    THIS IS A SEARCH SCOPE, NOT A CLAIM. The returned residues are the set a
    mutation-proposal step is allowed to consider. Membership says only "this
    residue has an atom at a distance between ``min_angstrom`` and
    ``max_angstrom`` from the ligand in this one model". It does not say the
    residue contacts the substrate, contributes to selectivity, or matters at
    all; proximity alone is explicitly rejected as a reason to mutate a
    position by :class:`~eagent.schemas.variant.SiteEvidence`.

    Semantics worth being precise about, because the obvious reading is wrong:
    a residue is included when *some* heavy atom falls inside the window. A
    residue whose every atom is closer than ``min_angstrom`` is therefore
    **excluded**. That is intended -- ``min_angstrom`` is how an engineering
    template carves out the frozen first shell it does not want touched (see
    ``EngineeringTemplate.default_shell_min_angstrom``). To get "everything
    within ``max``", pass ``min_angstrom=0.0``.

    Heavy atoms only, because most deposited structures have no hydrogens and
    a hydrogen-aware shell would silently be a different shell for predicted
    models than for crystal structures.

    Waters and other HETATM groups are excluded by default: a shell is a list
    of mutable protein positions, and "mutate HOH 501" is not a proposal. The
    ligand's own residues are always excluded.
    """
    lig = list(ligand_atoms)
    lig_heavy = [a for a in lig if a.is_heavy]
    if not lig_heavy:
        raise GeometryError(
            "pocket_shell needs at least one heavy ligand atom; an empty or "
            "hydrogen-only ligand selection usually means the ligand code did "
            "not match anything in the structure"
        )
    if min_angstrom < 0.0 or max_angstrom <= min_angstrom:
        raise GeometryError(
            f"invalid shell window [{min_angstrom}, {max_angstrom}]: need "
            f"0 <= min < max"
        )

    lig_ids = {id(a) for a in lig}
    lig_residue_keys = {a.residue_key for a in lig}
    lig_points = [as_point(a) for a in lig_heavy]

    out: list[Residue] = []
    for res in structure.residues():
        if res.key in lig_residue_keys:
            continue
        if res.is_water and not include_water:
            continue
        if res.is_hetatm and not include_hetatm:
            continue
        for atom in res.atoms:
            if not atom.is_heavy or id(atom) in lig_ids:
                continue
            p = as_point(atom)
            for q in lig_points:
                d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
                d = math.sqrt(d2)
                if min_angstrom <= d <= max_angstrom:
                    out.append(res)
                    break
            else:
                continue
            break

    out.sort(key=lambda r: (r.chain, r.resseq, r.icode))
    return out


# ---------------------------------------------------------------------------
# role resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedAtom:
    """The outcome of looking up a template role token.

    A result object rather than ``Atom | None`` because the *reason* a role
    did not resolve is diagnostic information the pipeline needs. "The
    cofactor namespace was never supplied" and "the cofactor is present but
    has no atom named hydride_donor_C4" lead to different fixes, and both are
    different from "the constraint was evaluated and failed".

    ``coords`` is populated for either an :class:`Atom` or a bare (x, y, z)
    lookup result, so a probe point can play a role without a fabricated Atom.
    """

    token: str
    namespace: str
    key: str
    atom: Atom | None = None
    coords: Point | None = None
    reason: str = ""

    @property
    def found(self) -> bool:
        """Whether coordinates were obtained. False carries a ``reason``, and
        a constraint naming an unfound role is unevaluated, not failed."""
        return self.coords is not None

    def require(self) -> Point:
        """Coordinates, or raise. For callers that treat absence as a bug."""
        if self.coords is None:
            raise GeometryError(f"role '{self.token}' unresolved: {self.reason}")
        return self.coords


#: A namespace is either a name -> atom mapping or a callable lookup.
Namespace = Mapping[str, Any] | Callable[[str], Any]
RoleContext = Mapping[str, Namespace]


def resolve_role(token: str, context: RoleContext) -> ResolvedAtom:
    """Resolve a template role token such as ``"cofactor.hydride_donor_C4"``.

    Templates name atoms by *role*, not by element or serial, because the same
    mechanistic role sits on a different atom name in every family and on a
    different serial in every structure. ``context`` maps a namespace
    (``"substrate"``, ``"cofactor"``, ``"protein"``) to either a dict of
    role-key -> atom, or a callable that performs the lookup.

    The token is split on the *first* dot only, so a namespace may use dotted
    keys of its own (``"protein.A.155.OH"`` -> namespace ``protein``, key
    ``A.155.OH``).

    Two different failures, handled differently on purpose:

    * A malformed token (no namespace) raises :class:`TemplateError`. That is
      a bug in a template, and a template bug must be fixed, not routed
      around as missing data.
    * A well-formed token that nothing can satisfy returns a
      :class:`ResolvedAtom` with ``found=False`` and a reason. That is missing
      data, and the correct downstream behaviour is to report the constraint
      as unevaluated.
    """
    if not isinstance(token, str) or "." not in token:
        raise TemplateError(
            f"role token {token!r} is malformed: expected "
            f"'<namespace>.<atom_key>', e.g. 'cofactor.hydride_donor_C4'"
        )
    namespace, _, key = token.partition(".")
    namespace = namespace.strip()
    key = key.strip()
    if not namespace or not key:
        raise TemplateError(
            f"role token {token!r} is malformed: namespace and atom key must "
            f"both be non-empty"
        )

    if not hasattr(context, "get"):
        raise GeometryError(
            f"resolver context must be a mapping of namespace -> lookup, got "
            f"{type(context).__name__}"
        )
    source = context.get(namespace)
    if source is None:
        return ResolvedAtom(
            token, namespace, key,
            reason=f"namespace '{namespace}' was not supplied to the resolver "
                   f"(available: {sorted(context) if context else 'none'})",
        )

    try:
        if callable(source):
            hit = source(key)
        elif hasattr(source, "get"):
            hit = source.get(key)
        else:
            raise TemplateError(
                f"resolver namespace '{namespace}' is neither a mapping nor a "
                f"callable (got {type(source).__name__})"
            )
    except TemplateError:
        raise
    except Exception as exc:  # a broken lookup is missing data, not a crash
        return ResolvedAtom(
            token, namespace, key,
            reason=f"lookup of '{key}' in namespace '{namespace}' raised "
                   f"{type(exc).__name__}: {exc}",
        )

    if hit is None:
        return ResolvedAtom(
            token, namespace, key,
            reason=f"namespace '{namespace}' has no atom for role '{key}'",
        )
    try:
        coords = as_point(hit)
    except GeometryError as exc:
        return ResolvedAtom(
            token, namespace, key,
            reason=f"namespace '{namespace}' returned something that is not an "
                   f"atom or a point for role '{key}': {exc}",
        )
    return ResolvedAtom(
        token, namespace, key,
        atom=hit if isinstance(hit, Atom) else None,
        coords=coords,
    )


class RoleResolver:
    """Caching role lookup that remembers what it could not resolve.

    Kept as an object rather than a bare function so that a geometry report
    can state *which* roles were unresolvable. Without that list, a report
    showing three constraints measured out of eight is indistinguishable from
    a report where five constraints failed -- and the two mean opposite things
    about the candidate.
    """

    def __init__(self, context: RoleContext | None = None, **namespaces: Namespace):
        ctx: dict[str, Namespace] = dict(context or {})
        ctx.update(namespaces)
        self.context: dict[str, Namespace] = ctx
        self._cache: dict[str, ResolvedAtom] = {}

    def resolve(self, token: str) -> ResolvedAtom:
        """Look a role token up once and remember the outcome, so repeated
        constraints stay consistent and the unresolved list stays complete."""
        hit = self._cache.get(token)
        if hit is None:
            hit = resolve_role(token, self.context)
            self._cache[token] = hit
        return hit

    def __call__(self, token: str) -> ResolvedAtom:
        return self.resolve(token)

    @property
    def unresolved(self) -> list[str]:
        """Role tokens asked for and not found, in the order first requested."""
        return [t for t, r in self._cache.items() if not r.found]

    def reasons(self) -> dict[str, str]:
        """Token -> why it did not resolve. Report material, not a log line."""
        return {t: r.reason for t, r in self._cache.items() if not r.found}


Resolver = RoleResolver | Callable[[str], ResolvedAtom] | RoleContext


def _as_resolver(resolver: Resolver) -> Callable[[str], ResolvedAtom]:
    """Accept a RoleResolver, a plain callable, or a raw context mapping."""
    if isinstance(resolver, RoleResolver):
        return resolver.resolve
    if callable(resolver):
        return resolver
    if hasattr(resolver, "get"):
        return RoleResolver(resolver).resolve
    raise GeometryError(
        f"cannot use {type(resolver).__name__} as a role resolver; pass a "
        f"RoleResolver, a callable token -> ResolvedAtom, or a context mapping"
    )


# ---------------------------------------------------------------------------
# constraint measurement
# ---------------------------------------------------------------------------

_KIND_ARITY: dict[str, int] = {"distance": 2, "angle": 3, "dihedral": 4}


def measure_constraint(
    constraint: GeometryConstraint, resolver: Resolver
) -> float | None:
    """Measure one template constraint, or return ``None`` if it cannot be.

    ``None`` is the whole reason this function exists. A constraint whose
    atoms could not be resolved has *not been tested*. If this returned a
    large distance, or 0.0, or the nearest resolvable alternative, the
    constraint would appear to have failed (or passed) and a reviewer could
    not tell the difference between "the geometry is wrong" and "the cofactor
    was never in the model". ``GeometryConstraint.satisfied_by(None)`` likewise
    returns ``None``, so the distinction survives all the way into
    :class:`~eagent.schemas.candidate.GeometryReport`.

    A constraint of an unknown ``kind``, or one missing an atom slot its kind
    requires, raises :class:`TemplateError`: that is a malformed template, not
    missing structural data, and it must be fixed at the template.

    The unit is taken from the geometry: angstroms for a distance, degrees for
    an angle or dihedral. The constraint's ``unit`` field is checked against
    that and a mismatch raises, because comparing a value in degrees against a
    window written in angstroms produces a confident wrong verdict.
    """
    lookup = _as_resolver(resolver)
    kind = (constraint.kind or "").strip().lower()
    arity = _KIND_ARITY.get(kind)
    if arity is None:
        raise TemplateError(
            f"constraint '{constraint.name}': unknown kind {constraint.kind!r}; "
            f"supported kinds are {sorted(_KIND_ARITY)}"
        )

    expected_unit = "angstrom" if kind == "distance" else "degree"
    declared = (constraint.unit or "").strip().lower()
    # GeometryConstraint's own validator rewrites unit "angstrom" -> "degree"
    # for kind="angle" but not for kind="dihedral", so a dihedral written with
    # the pydantic default arrives here still saying "angstrom". That one
    # combination is treated as "unit not stated" rather than as an error; it
    # is a gap in the schema validator, not a claim by the template author.
    # Every other dimensional mismatch raises, because measuring degrees
    # against a window written in angstroms yields a confident wrong verdict.
    unit_unset = (kind == "dihedral" and declared == "angstrom")
    if declared and not unit_unset \
            and declared not in (expected_unit, expected_unit + "s"):
        raise TemplateError(
            f"constraint '{constraint.name}' is a {kind} but declares unit "
            f"{constraint.unit!r}; a {kind} is measured in {expected_unit}s and "
            f"its window must be written in the same unit"
        )

    tokens = [constraint.atom_a, constraint.atom_b,
              constraint.atom_c, constraint.atom_d][:arity]
    for slot, tok in zip("abcd", tokens):
        if not tok:
            raise TemplateError(
                f"constraint '{constraint.name}' of kind {kind} needs atom_{slot}"
            )

    points: list[Point] = []
    for tok in tokens:
        hit = lookup(tok)
        if not hit.found:
            return None
        points.append(hit.coords)  # type: ignore[arg-type]

    try:
        if kind == "distance":
            return distance(points[0], points[1])
        if kind == "angle":
            return angle(points[0], points[1], points[2])
        return dihedral(points[0], points[1], points[2], points[3])
    except GeometryError:
        # Degenerate coordinates (coincident or collinear atoms) mean the
        # quantity does not exist for this pose. Unmeasurable, not failed.
        return None


def measure_all(
    constraints: Iterable[GeometryConstraint], resolver: Resolver
) -> dict[str, float | None]:
    """Measure every constraint by name, keeping ``None`` for the unmeasurable.

    The returned dict is shaped to drop straight into
    ``GeometryReport.measurements``, whose companion ``satisfied`` map is
    built by ``GeometryConstraint.satisfied_by`` -- so the pass/fail decision
    is made by the sourced template object and never by this module.

    Duplicate constraint names raise, because a dict silently keeps the last
    one and a template with two differently-windowed constraints called
    ``hydride_distance`` would then be evaluated against only one of them.
    """
    lookup = _as_resolver(resolver)
    out: dict[str, float | None] = {}
    for c in constraints:
        if c.name in out:
            raise TemplateError(
                f"duplicate geometry constraint name '{c.name}'; names are the "
                f"key of the measurement report and must be unique"
            )
        out[c.name] = measure_constraint(c, lookup)
    return out


# ---------------------------------------------------------------------------
# named mechanistic measurements
# ---------------------------------------------------------------------------


def hydride_transfer_distance(donor_atom: Atom | Point,
                              acceptor_atom: Atom | Point) -> float:
    """Heavy-atom separation between a hydride donor and its acceptor, in A.

    READ THIS BEFORE USING THE NUMBER. This is the distance between two
    *heavy* atoms -- typically the nicotinamide C4 of NAD(P)H and the carbonyl
    carbon of the ketone. It is **not** the path length the transferring
    hydride travels, and it is not a donor-H...acceptor distance.

    Why the distinction is load-bearing:

    * Almost no structure in this pipeline has hydrogens. A "hydride transfer
      distance" that silently meant C-H...C would be unmeasurable for a
      crystal structure and measurable for a protonated model, so the same
      constraint would mean two different things depending on the file.
    * The C4-H bond is about 1.1 A long and points in a direction this
      function knows nothing about. A C4...C separation of 3.5 A is therefore
      consistent with a near-ideal transfer geometry *or* with a hydride
      pointing the wrong way entirely. Distance alone cannot distinguish
      them; that is what the accompanying angle constraints (and
      :func:`burgi_dunitz_angle`) are for.

    No window is applied. The acceptable separation belongs to a calibrated
    :class:`~eagent.schemas.templates.GeometryConstraint` in the family's
    catalytic template.
    """
    return distance(donor_atom, acceptor_atom)


#: The Burgi-Dunitz angle observed in small-molecule crystal structures of
#: nucleophile-carbonyl contacts, in degrees (Burgi, Dunitz & Shefter,
#: *J. Am. Chem. Soc.* 95 (1973) 5065).
#:
#: THIS IS A LITERATURE EXPECTATION, NOT A CUTOFF. No function in this package
#: compares a measured angle against it; it is exported so that a report can
#: quote the expectation next to a measured value. The window a candidate is
#: actually judged against must come from a ``GeometryConstraint`` whose
#: ``calibrated_on`` lists the systems it was fitted to, because the
#: distribution in an enzyme active site is broader than in small-molecule
#: crystals and differs between families.
BURGI_DUNITZ_LITERATURE_DEG: tuple[float, float] = (105.0, 107.0)


def burgi_dunitz_angle(donor: Atom | Point, carbonyl_C: Atom | Point,
                       carbonyl_O: Atom | Point) -> float:
    """Nucleophile approach angle to a carbonyl: donor-C=O, in degrees.

    For a ketoreductase this is the nicotinamide C4 (the hydride donor) to the
    substrate carbonyl carbon to the carbonyl oxygen. Small-molecule
    crystallography puts the preferred trajectory near
    :data:`BURGI_DUNITZ_LITERATURE_DEG` -- the nucleophile approaches above the
    carbonyl plane and tilted away from the oxygen, towards the forming
    tetrahedral geometry.

    That range is an expectation to calibrate against, **not** a pass/fail
    cutoff, and nothing in this module tests against it. Two reasons it would
    be wrong to hard-code:

    * The small-molecule value describes an isolated contact. Enzyme active
      sites hold substrates in geometries spread well beyond a 2-degree
      window, and the spread differs by family and by substrate size.
    * The angle is only half the story: it is a scalar, so it does not say
      which prochiral face the donor is on. Face assignment needs the signed
      :func:`dihedral` (see its note on +90 versus -90). An angle in the
      "ideal" range is fully compatible with producing the wrong enantiomer.

    The window a candidate is judged against must come from the family's
    catalytic template.
    """
    return angle(donor, carbonyl_C, carbonyl_O)
