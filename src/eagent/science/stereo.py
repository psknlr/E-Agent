"""Which prochiral face is attacked, and therefore which enantiomer forms.

SCOPE -- read this before calling anything here
===============================================
"The substrate fits the pocket" and "the target enantiomer is produced" are
two separate questions. This module answers only the second one, and only
*conditional on a pose that has already been accepted as geometrically
competent by the geometry/gating layer*. A face call computed from a pose that
never satisfied the catalytic distance and angle windows is a statement about
an arrangement that cannot react; it is arithmetic, not chemistry.

The separation matters because the two questions fail independently and in
opposite directions. A pocket that accommodates the substrate beautifully can
still deliver hydride to the wrong face, and an enzyme whose docking scores are
mediocre can be perfectly enantioselective. Fusing the two into one "it looks
good" judgement is the standard way a stereochemical prediction becomes
unfalsifiable, so nothing in this module reads a docking score, a pocket
volume or a clash count.

What this module does NOT do
----------------------------
* It does not perceive CIP priorities. See :class:`CIPPriority`.
* It does not estimate an enantiomeric excess. Pose counts are an artefact of
  how the modelling protocol sampled, not a Boltzmann population; see
  :func:`call_stereochemistry`.
* It does not decide whether a pose is valid. That is the caller's gate.

Geometry dependency
-------------------
``eagent.science.geometry`` exposes distance / angle / dihedral / centroid.
None of those yields a *signed* quantity, and the sign is the entire content of
a face call: an unsigned angle cannot distinguish re from si. This module
therefore carries its own four-line vector helpers rather than taking a
dependency it cannot actually use. Atom objects from
``eagent.science.structure_io`` are accepted directly: every coordinate
argument is duck-typed on ``.x/.y/.z`` as well as on a 3-sequence, so no import
is needed and no particular Atom class is imposed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence

from ..errors import FabricationGuardError, TemplateError
from ..schemas import StereoCall, Stereochemistry
from ..schemas.candidate import DEFAULT_COMPETING_FACE_FRACTION

__all__ = [
    "FaceLabel",
    "Configuration",
    "PointLike",
    "DEFAULT_COMPETING_FACE_FRACTION",
    "DEFAULT_IN_PLANE_TOLERANCE_DEG",
    "COLLINEARITY_SIN_FLOOR",
    "CIP_CONFIG_KEY",
    "CIPPriority",
    "cip_from_template",
    "face_of_approach",
    "face_of_approach_with_cip",
    "out_of_plane_angle_deg",
    "centre_planarity_deviation",
    "face_to_configuration",
    "call_stereochemistry",
]


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

#: Which face of the trigonal carbonyl the hydride donor sits on.
FaceLabel = Literal["re", "si", "in_plane_ambiguous"]

#: Configuration of the stereocentre created at the former carbonyl carbon.
Configuration = Literal["R", "S"]


class _HasXYZ(Protocol):
    """Structural type of anything carrying Cartesian coordinates."""

    x: float
    y: float
    z: float


#: A point: a 3-sequence, or any object with ``.x/.y/.z`` (e.g. ``structure_io.Atom``).
PointLike = "Sequence[float] | _HasXYZ"

_Vec = tuple[float, float, float]


# --------------------------------------------------------------------------
# Numerical guards
#
# Neither constant below is a catalytic threshold. Constraint windows for
# catalysis come from a sourced CatalyticTemplate's GeometryConstraint objects
# and never from this module. These two numbers exist only to stop the code
# reporting a confident sign that the input coordinates do not support, and
# both are overridable per call / per protocol.
# --------------------------------------------------------------------------

#: Donor elevation above the sp2 plane, in degrees, below which the face call is
#: refused as ``"in_plane_ambiguous"``.
#:
#: CALIBRATION: this is a *coordinate-precision* parameter, not a chemical one.
#: Set it from the positional noise of the modelling protocol. A docking
#: ensemble with ~0.5 A positional scatter on a ~1.23 A C=O vector leaves the
#: plane normal uncertain by several degrees, so 10 degrees is a deliberately
#: conservative starting value for a docked pose. A crystallographic complex
#: refined at high resolution justifies a much smaller value; a coarse
#: homology-model pose justifies a larger one. Re-set it per protocol and
#: record the value in provenance.
DEFAULT_IN_PLANE_TOLERANCE_DEG: float = 10.0

#: Smallest |sin| of the angle between the two in-plane edges for which a plane
#: normal is considered defined. Below this the three sp2 ligands are collinear
#: within numerical noise and the pose is broken, not ambiguous.
COLLINEARITY_SIN_FLOOR: float = 1e-6

#: Config key under which CIP ranks must be supplied. See :func:`cip_from_template`.
CIP_CONFIG_KEY: str = "cip_ranks"

_UNDETERMINED_TOKENS: frozenset[str] = frozenset({
    "", "none", "null", "undetermined", "unknown", "insufficient_evidence",
    "in_plane_ambiguous", "ambiguous",
})


# --------------------------------------------------------------------------
# Minimal vector algebra
# --------------------------------------------------------------------------

def _as_point(value: Any, name: str) -> _Vec:
    """Coerce an Atom-like or 3-sequence argument to a float triple.

    Duck-typed rather than typed against a concrete Atom class so that this
    module does not have to import the structure parser. A wrong *kind* of
    argument raises instead of being coerced to zeros: a silently zeroed
    coordinate would place an atom at the origin and invert face calls without
    any visible error.
    """
    if value is None:
        raise TypeError(f"{name}: coordinate is None; a missing atom is not a point")
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return (float(value.x), float(value.y), float(value.z))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise TypeError(f"{name}: expected 3 coordinates, got {len(value)}")
        return (float(value[0]), float(value[1]), float(value[2]))
    raise TypeError(
        f"{name}: expected a 3-sequence or an object with .x/.y/.z, "
        f"got {type(value).__name__}"
    )


def _sub(p: _Vec, q: _Vec) -> _Vec:
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _cross(u: _Vec, v: _Vec) -> _Vec:
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def _dot(u: _Vec, v: _Vec) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _norm(u: _Vec) -> float:
    return math.sqrt(_dot(u, u))


def _plane_normal(o: _Vec, a: _Vec, b: _Vec) -> _Vec:
    """Normal of the plane through the three sp2 ligands, O -> A -> B.

    Right-hand rule: viewed from the side the returned vector points toward,
    the sequence O -> A -> B runs *counterclockwise*. That orientation
    convention is the whole basis of the re/si assignment below, so it is
    computed here once and never re-derived at the call sites.

    Raises when the three ligands are collinear within
    :data:`COLLINEARITY_SIN_FLOOR`: a collapsed ligand triangle is a broken
    pose, and returning an arbitrary normal would turn it into a confident
    stereochemical claim.
    """
    e1 = _sub(a, o)
    e2 = _sub(b, o)
    n1, n2 = _norm(e1), _norm(e2)
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError(
            "degenerate carbonyl ligand set: two of the three sp2 ligands occupy "
            "the same point"
        )
    normal = _cross(e1, e2)
    if _norm(normal) <= COLLINEARITY_SIN_FLOOR * n1 * n2:
        raise ValueError(
            "degenerate carbonyl ligand set: the three sp2 ligands are collinear, "
            "so no face can be defined"
        )
    return normal


# --------------------------------------------------------------------------
# CIP priorities -- supplied, never perceived
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CIPPriority:
    """CIP ranks for the three sp2 ligands of one prochiral carbonyl carbon.

    WHY THIS IS A CONTAINER AND NOT AN ALGORITHM
    --------------------------------------------
    CIP perception is a graph algorithm with duplicated-atom rules, hierarchical
    digraph exploration, isotope and like/unlike tie-breakers, and it is famous
    for being got wrong by implementations far more serious than this one. A
    wrong priority assignment does not degrade the answer, it *inverts* it:
    swapping ranks 2 and 3 turns every R into an S, silently, with every
    distance, angle and docking score still looking perfect. There is no
    downstream check that would catch it, because the geometry is unchanged.

    So this module refuses to improvise. Ranks are data the caller supplies from
    a sourced template or an operator-reviewed config, carrying the provenance
    of whoever asserted them. Guessing from element alone is explicitly not
    allowed either: O beats C at a carbonyl carbon, true, but the relative rank
    of the two *carbon* substituents is exactly the hard part, and "both are
    carbon" is not an answer.

    Attributes
    ----------
    ranks:
        Atom key -> CIP rank, 1 being the highest priority. These are the three
        ligands of the trigonal carbonyl carbon: the carbonyl oxygen and the two
        carbon substituents. Distinctness is enforced; two ligands with equal
        rank means the centre is not prochiral and no face call is meaningful.
    source:
        Who asserted these ranks, e.g. ``"template:rt_ketone_reduction_v1"`` or
        ``"operator:jdoe"``. Empty is rejected: an unsourced CIP assignment is
        precisely the silent inverter described above.
    incoming_rank:
        Optional convenience slot for the rank the *incoming* group takes in the
        product. It lives apart from ``ranks`` because it is a property of the
        reaction (hydride, alkyl, cyanide ...), not of the substrate.
    scope:
        ``"product"`` or ``"reactant"``. See :func:`face_to_configuration` for
        why the distinction is load-bearing.
    """

    ranks: Mapping[str, int]
    source: str
    incoming_rank: int | None = None
    scope: str = "product"

    def __post_init__(self) -> None:
        if not self.source or not str(self.source).strip():
            raise TemplateError(
                "CIPPriority requires a source; an unsourced CIP assignment "
                "silently inverts every stereochemical call downstream"
            )
        if not self.ranks:
            raise TemplateError("CIPPriority requires at least one rank")
        cleaned: dict[str, int] = {}
        for key, rank in dict(self.ranks).items():
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TemplateError(
                    f"CIP rank for '{key}' must be an int, got {type(rank).__name__}"
                )
            if rank < 1:
                raise TemplateError(f"CIP rank for '{key}' must be >= 1, got {rank}")
            cleaned[str(key)] = rank
        if len(set(cleaned.values())) != len(cleaned):
            raise TemplateError(
                f"CIP ranks must be distinct; got {cleaned}. Equal ranks mean the "
                f"centre is not prochiral and no face assignment applies."
            )
        if self.incoming_rank is not None:
            if isinstance(self.incoming_rank, bool) or not isinstance(self.incoming_rank, int):
                raise TemplateError("incoming_rank must be an int")
            if self.incoming_rank in cleaned.values():
                raise TemplateError(
                    f"incoming group rank {self.incoming_rank} collides with a "
                    f"retained-ligand rank in {cleaned}"
                )
        if self.scope not in ("product", "reactant"):
            raise TemplateError(f"scope must be 'product' or 'reactant', got {self.scope!r}")
        object.__setattr__(self, "ranks", MappingProxyType(cleaned))

    # -- queries ----------------------------------------------------------
    def rank_of(self, key: str) -> int | None:
        """Rank for one ligand, or None when it was not supplied."""
        return self.ranks.get(key)

    def missing_keys(self, expected: Iterable[str]) -> list[str]:
        """Expected ligand keys with no rank; the caller reports these as gaps."""
        return [k for k in expected if k not in self.ranks]

    @property
    def is_complete(self) -> bool:
        """True when all three sp2 ligands of the carbonyl carbon are ranked.

        Two ranks are not enough: the face call needs the full rotational order
        of the trigonal face, and a missing third ligand means the sense of
        rotation is undefined rather than merely uncertain.
        """
        return len(self.ranks) == 3

    def keys_in_priority_order(self) -> tuple[str, ...]:
        """Ligand keys sorted by descending CIP priority (rank 1 first)."""
        return tuple(k for k, _ in sorted(self.ranks.items(), key=lambda kv: kv[1]))

    def as_dict(self) -> dict[str, int]:
        return dict(self.ranks)


def _extract_rank_block(source: Any, key: str) -> tuple[Mapping[str, Any] | None, str]:
    """Locate a rank block and a provenance label on a template or config object."""
    label = ""
    if isinstance(source, Mapping):
        label = str(source.get("template_id") or source.get("id") or "config")
        if key in source and isinstance(source[key], Mapping):
            return source[key], f"config:{label}.{key}"
        nested = source.get("stereo")
        if isinstance(nested, Mapping) and isinstance(nested.get(key), Mapping):
            return nested[key], f"config:{label}.stereo.{key}"
        return None, label
    block = getattr(source, key, None)
    label = str(getattr(source, "template_id", None) or type(source).__name__)
    if isinstance(block, Mapping):
        return block, f"template:{label}"
    nested = getattr(source, "stereo", None)
    if isinstance(nested, Mapping) and isinstance(nested.get(key), Mapping):
        return nested[key], f"template:{label}.stereo.{key}"
    return None, label


def cip_from_template(
    reaction_template_or_config: Any,
    *,
    key: str = CIP_CONFIG_KEY,
    scope: str = "product",
) -> CIPPriority:
    """Read supplied CIP ranks off a reaction template or a task config.

    WHY IT RAISES INSTEAD OF RETURNING A DEFAULT
    --------------------------------------------
    There is no safe default here. Any fallback -- element order, input order,
    "oxygen first then whichever substituent was listed first" -- produces a
    complete, confident, frequently inverted answer. :class:`TemplateError` is
    raised so the run stops at the point where a human can supply the ranks,
    instead of a wrong configuration reaching a synthesis order. Callers that
    want a soft failure catch this and let :func:`face_to_configuration` return
    ``None``, which propagates as ``"insufficient_evidence"`` in
    :class:`~eagent.schemas.candidate.StereoCall`.

    Accepted shapes, in order:
      * a :class:`CIPPriority`, returned unchanged;
      * a mapping with ``{key: {ligand: rank, ...}}`` at the top level or under
        a ``"stereo"`` sub-mapping;
      * an object carrying the same block as an attribute.

    The block may additionally carry ``"incoming_group_rank"`` (or
    ``"incoming_group"``) and ``"source"``. ``"source"`` is preferred over the
    derived label, because the person who asserted the ranks is more
    informative provenance than the file they ended up in.
    """
    if isinstance(reaction_template_or_config, CIPPriority):
        return reaction_template_or_config
    if reaction_template_or_config is None:
        raise TemplateError(
            f"no source object supplied for CIP ranks; expected a mapping or "
            f"template carrying '{key}'"
        )

    block, label = _extract_rank_block(reaction_template_or_config, key)
    if block is None:
        raise TemplateError(
            f"{label}: no CIP priority block '{key}' found. CIP ranks for the "
            f"three carbonyl substituents must be supplied explicitly; this "
            f"module does not perceive them, because a wrong assignment "
            f"silently inverts every R/S call."
        )

    raw = dict(block)
    declared_source = raw.pop("source", None)
    incoming = raw.pop("incoming_group_rank", None)
    alias = raw.pop("incoming_group", None)
    if incoming is None:
        incoming = alias
    if not raw:
        raise TemplateError(
            f"{label}: CIP priority block '{key}' carries no ligand ranks"
        )
    if incoming is not None and (isinstance(incoming, bool) or not isinstance(incoming, int)):
        raise TemplateError(
            f"{label}: incoming group rank must be an int, got {incoming!r}"
        )
    return CIPPriority(
        ranks=raw,
        source=str(declared_source) if declared_source else label,
        incoming_rank=incoming,
        scope=scope,
    )


# --------------------------------------------------------------------------
# Face assignment
# --------------------------------------------------------------------------

def _elevation_sin(
    donor: _Vec, centre: _Vec, o: _Vec, a: _Vec, b: _Vec
) -> float:
    """Signed sine of the donor's elevation above the (O, A, B) plane."""
    normal = _plane_normal(o, a, b)
    d = _sub(donor, centre)
    dn = _norm(d)
    if dn == 0.0:
        raise ValueError(
            "donor coincides with the carbonyl carbon; no approach vector exists"
        )
    s = _dot(d, normal) / (dn * _norm(normal))
    return max(-1.0, min(1.0, s))


def face_of_approach(
    donor_xyz: Any,
    carbonyl_c: Any,
    substituent_a: Any,
    substituent_b: Any,
    carbonyl_o: Any,
    *,
    in_plane_tolerance_deg: float = DEFAULT_IN_PLANE_TOLERANCE_DEG,
) -> FaceLabel:
    """Decide whether the hydride donor sits on the re or the si face.

    METHOD
    ------
    The three sp2 ligands define a plane whose oriented normal
    ``N = (A - O) x (B - O)`` points, by the right-hand rule, toward the side
    from which the sequence ``O -> A -> B`` appears *counterclockwise*. The
    donor's position relative to the carbonyl carbon is projected onto ``N``:

      * ``d . N > 0``  -- donor on the counterclockwise side -> ``"si"``
      * ``d . N < 0``  -- donor on the clockwise side        -> ``"re"``

    which is the IUPAC definition (clockwise 1 -> 2 -> 3 seen from the viewer
    means that viewer is on the re face).

    ORDERING TRAP -- the one thing to get right
    -------------------------------------------
    The ordered triple whose handedness is measured is
    ``(carbonyl_o, substituent_a, substituent_b)``, in that order, regardless of
    the positional order of this function's arguments. The returned label is the
    IUPAC face only when that ordering is the *descending CIP priority* order.
    The oxygen is safe -- O outranks C at every carbonyl carbon -- so the risk
    lies entirely in whether ``substituent_a`` outranks ``substituent_b``. If you
    are not certain, call :func:`face_of_approach_with_cip`, which sorts the
    three ligands by supplied ranks and removes the trap.

    AMBIGUITY RATHER THAN A FORCED CALL
    -----------------------------------
    When the donor lies within ``in_plane_tolerance_deg`` of the sp2 plane, the
    sign of ``d . N`` is dominated by coordinate noise rather than by chemistry,
    so ``"in_plane_ambiguous"`` is returned instead of a coin flip. Such a pose
    is also not a productive hydride-transfer geometry: in-plane attack on a
    carbonyl has no orbital overlap with the pi* system, so a pose at near-zero
    elevation is evidence of a modelling artefact, not of a real trajectory.

    Raises
    ------
    ValueError
        If the three ligands are collinear or coincident (a broken pose), if the
        donor coincides with the carbonyl carbon, or if the tolerance is outside
        ``[0, 90)``.
    """
    if not (0.0 <= float(in_plane_tolerance_deg) < 90.0):
        raise ValueError(
            f"in_plane_tolerance_deg must be in [0, 90), got {in_plane_tolerance_deg}"
        )
    donor = _as_point(donor_xyz, "donor_xyz")
    centre = _as_point(carbonyl_c, "carbonyl_c")
    a = _as_point(substituent_a, "substituent_a")
    b = _as_point(substituent_b, "substituent_b")
    o = _as_point(carbonyl_o, "carbonyl_o")

    s = _elevation_sin(donor, centre, o, a, b)
    elevation = math.degrees(math.asin(abs(s)))
    if elevation < float(in_plane_tolerance_deg):
        return "in_plane_ambiguous"
    return "si" if s > 0.0 else "re"


def face_of_approach_with_cip(
    donor_xyz: Any,
    carbonyl_c: Any,
    substituents: Mapping[str, Any],
    cip_ranks: CIPPriority,
    *,
    in_plane_tolerance_deg: float = DEFAULT_IN_PLANE_TOLERANCE_DEG,
) -> FaceLabel:
    """Face call with the ligand ordering resolved from supplied CIP ranks.

    Preferred over :func:`face_of_approach` in production code: it removes the
    one way the primitive can be misused, which is passing the two carbon
    substituents in the wrong order and getting a confident, inverted answer.

    ``substituents`` maps the same ligand keys used in ``cip_ranks`` to their
    coordinates, and must contain exactly the three sp2 ligands of the carbonyl
    carbon. A key without a rank raises :class:`TemplateError` rather than being
    ordered by dictionary insertion.
    """
    if len(substituents) != 3:
        raise ValueError(
            f"a trigonal carbonyl carbon has exactly three ligands; got "
            f"{len(substituents)}: {sorted(substituents)}"
        )
    missing = cip_ranks.missing_keys(substituents.keys())
    if missing:
        raise TemplateError(
            f"no CIP rank supplied for ligand(s) {missing}; refusing to order "
            f"them by element or by insertion order"
        )
    ordered = sorted(substituents.keys(), key=lambda k: cip_ranks.ranks[k])
    k1, k2, k3 = ordered
    return face_of_approach(
        donor_xyz,
        carbonyl_c,
        substituents[k2],
        substituents[k3],
        substituents[k1],
        in_plane_tolerance_deg=in_plane_tolerance_deg,
    )


def out_of_plane_angle_deg(
    donor_xyz: Any,
    carbonyl_c: Any,
    substituent_a: Any,
    substituent_b: Any,
    carbonyl_o: Any,
) -> float:
    """Unsigned elevation of the donor above the sp2 plane, in degrees.

    Exposed so the caller can record *how far* a face call was from the
    ambiguity boundary. A call at 11 degrees with a 10 degree tolerance passed a
    threshold by a hair and deserves a QC flag; a call at 70 degrees did not.
    Reporting only the label would hide that difference from a reviewer.
    """
    donor = _as_point(donor_xyz, "donor_xyz")
    centre = _as_point(carbonyl_c, "carbonyl_c")
    a = _as_point(substituent_a, "substituent_a")
    b = _as_point(substituent_b, "substituent_b")
    o = _as_point(carbonyl_o, "carbonyl_o")
    return math.degrees(math.asin(abs(_elevation_sin(donor, centre, o, a, b))))


def centre_planarity_deviation(
    carbonyl_c: Any,
    substituent_a: Any,
    substituent_b: Any,
    carbonyl_o: Any,
) -> float:
    """Distance in angstrom of the carbonyl carbon from its ligands' plane.

    A diagnostic, deliberately *not* a gate inside :func:`face_of_approach`.
    A genuine sp2 carbon sits within a few hundredths of an angstrom of the
    plane of its three ligands; a pose where it sits 0.4 A out has either been
    pyramidalised by the model or has the wrong atoms assigned as substituents.
    Either way the right response is a QC flag raised by the caller with the
    number in hand, not a silent threshold buried in the face call.
    """
    centre = _as_point(carbonyl_c, "carbonyl_c")
    a = _as_point(substituent_a, "substituent_a")
    b = _as_point(substituent_b, "substituent_b")
    o = _as_point(carbonyl_o, "carbonyl_o")
    normal = _plane_normal(o, a, b)
    return abs(_dot(_sub(centre, o), normal)) / _norm(normal)


# --------------------------------------------------------------------------
# Face -> product configuration
# --------------------------------------------------------------------------

def face_to_configuration(
    face: str | None,
    cip_ranks: CIPPriority | None,
    incoming_group_rank: int | None = None,
) -> Configuration | None:
    """Translate an attacked face into the configuration of the new stereocentre.

    DERIVATION (so a reviewer can check it rather than trust it)
    -----------------------------------------------------------
    After addition, the incoming group occupies the side the donor came from,
    and the three retained ligands pyramidalise away from it. Consider the
    ordered 4-tuple ``T = (r1, r2, r3, incoming)`` where ``r1 > r2 > r3`` is the
    retained ligands in descending priority.

    1. Viewed from the incoming group's side, ``r1 -> r2 -> r3`` runs clockwise
       exactly when that side is the re face (the IUPAC definition).
    2. The R/S rule inspects the tuple with its *last* member pointing away from
       the viewer. Viewing instead from that member's side reverses the apparent
       rotation. So, reading ``T`` as if it were the priority order, re gives S
       and si gives R.
    3. ``T`` is the true priority order only when the incoming group ranks last.
       In general the true order is obtained by moving the incoming group from
       position 4 to position ``g`` (its rank), a cycle of length ``5 - g``. That
       permutation is odd for ``g`` in {1, 3} and even for ``g`` in {2, 4}, and an
       odd permutation of the priority order inverts the descriptor.

    Hence ``base = S if face == "re" else R``, inverted when ``g`` is 1 or 3. For
    hydride transfer ``g = 4``, so re gives S and si gives R -- which is the
    familiar result that re-face hydride delivery to acetophenone yields
    (S)-1-phenylethanol. The general form is kept because the same machinery is
    used for cyanide and alkyl additions, where the incoming group is not last.

    PRODUCT-SIDE RANKS
    ------------------
    ``cip_ranks`` must rank the three retained ligands *as they stand in the
    product*. In the overwhelmingly common case the relative order of the three
    is unchanged by the reaction (C=O -> C-OH does not reorder the two carbon
    substituents against each other), so reactant-side ranks are usually safe --
    but "usually" is the caller's call, not this function's, and
    :attr:`CIPPriority.scope` records which was supplied.

    Returns
    -------
    ``"R"``, ``"S"``, or ``None``.

    ``None`` is returned -- never a guess -- when the face is ambiguous or
    absent, when no ranks were supplied, when fewer than three ligands are
    ranked, or when no incoming-group rank is available. Downstream that becomes
    an undetermined pose and, if no pose resolves, a
    ``"insufficient_evidence"`` :class:`~eagent.schemas.candidate.StereoCall`.

    Raises
    ------
    ValueError
        If the supplied ranks are internally inconsistent -- more than three
        retained ligands, or a rank set that is not a permutation of 1..4 once
        the incoming group is included. That is a contradiction in the input,
        not a gap in it, and silently picking an interpretation would be the
        same inversion risk :class:`CIPPriority` exists to prevent.
    """
    if face is None or face not in ("re", "si"):
        return None
    if cip_ranks is None:
        return None
    if len(cip_ranks.ranks) > 3:
        raise ValueError(
            f"cip_ranks must carry exactly the three retained ligands of the "
            f"carbonyl carbon; got {len(cip_ranks.ranks)}: {cip_ranks.as_dict()}. "
            f"The incoming group's rank is passed separately."
        )
    if not cip_ranks.is_complete:
        return None

    g = incoming_group_rank if incoming_group_rank is not None else cip_ranks.incoming_rank
    if g is None:
        return None
    if isinstance(g, bool) or not isinstance(g, int):
        raise ValueError(f"incoming_group_rank must be an int, got {g!r}")

    full = set(cip_ranks.ranks.values()) | {g}
    if full != {1, 2, 3, 4}:
        raise ValueError(
            f"the three retained ranks {sorted(cip_ranks.ranks.values())} plus the "
            f"incoming group rank {g} must form exactly {{1, 2, 3, 4}}; got "
            f"{sorted(full)}"
        )

    base: Configuration = "S" if face == "re" else "R"
    if g in (1, 3):
        return "R" if base == "S" else "S"
    return base


# --------------------------------------------------------------------------
# Aggregation over poses
# --------------------------------------------------------------------------

def _normalise_configuration(value: Any, where: str) -> Configuration | None:
    """Map one per-pose entry to ``"R"``, ``"S"`` or undetermined.

    An unrecognised label raises rather than being swept into the undetermined
    bucket: a typo that quietly reduces the opposite-face count would bias the
    stereochemical call in the direction the pipeline is hoping for.
    """
    if value is None:
        return None
    if isinstance(value, Stereochemistry):
        if value is Stereochemistry.R:
            return "R"
        if value is Stereochemistry.S:
            return "S"
        return None
    if isinstance(value, str):
        v = value.strip()
        if v.upper() in ("R", "S"):
            return v.upper()  # type: ignore[return-value]
        if v.lower() in _UNDETERMINED_TOKENS:
            return None
    raise ValueError(
        f"{where}: {value!r} is not a configuration label. Supply 'R', 'S', or "
        f"None/'undetermined'; per-pose entries come from face_to_configuration."
    )


def _iter_configurations(poses_faces: Any) -> list[Configuration | None]:
    """Normalise the several shapes a caller may hand in, without guessing."""
    if poses_faces is None:
        return []
    items: list[tuple[str, Any]]
    if isinstance(poses_faces, Mapping):
        items = [(str(k), v) for k, v in poses_faces.items()]
    else:
        items = []
        for i, entry in enumerate(poses_faces):
            if (isinstance(entry, (tuple, list)) and len(entry) == 2
                    and isinstance(entry[0], str)):
                items.append((str(entry[0]), entry[1]))
            else:
                items.append((f"pose[{i}]", entry))
    return [_normalise_configuration(v, k) for k, v in items]


def _normalise_target(target_configuration: Any) -> Configuration | None:
    if target_configuration is None:
        return None
    if isinstance(target_configuration, Stereochemistry):
        if target_configuration is Stereochemistry.R:
            return "R"
        if target_configuration is Stereochemistry.S:
            return "S"
        return None
    if isinstance(target_configuration, str):
        v = target_configuration.strip()
        if v.upper() in ("R", "S"):
            return v.upper()  # type: ignore[return-value]
        if v.lower() in ("racemic", "achiral", "unspecified", "") or v.lower() in _UNDETERMINED_TOKENS:
            return None
    raise ValueError(
        f"target_configuration {target_configuration!r} is not 'R', 'S', or a "
        f"Stereochemistry member"
    )


def call_stereochemistry(
    poses_faces: Any,
    target_configuration: Any,
    *,
    basis: str = "",
    competing_face_fraction: float = DEFAULT_COMPETING_FACE_FRACTION,
) -> StereoCall:
    """Aggregate per-pose configurations into a directional stereochemical call.

    POSE COUNTS ARE NOT A POPULATION
    --------------------------------
    The counts in the returned :class:`~eagent.schemas.candidate.StereoCall` are
    an artefact of the modelling protocol: how many seeds were run, what the
    clustering radius was, which poses survived the validity filter, whether the
    sampler happens to over-populate one basin. They are not a Boltzmann
    distribution and they carry no free-energy information. Seven re poses and
    three si poses does not mean 40% ee, it means the sampler produced seven and
    three. For that reason this function **never** populates
    ``predicted_ee_pct``, and a final guard raises
    :class:`~eagent.errors.FabricationGuardError` if anything ever manages to set
    it. A numeric ee requires a model calibrated on this reaction class, and
    :class:`StereoCall` already demands a named calibration source for one.

    What the call *is* good for is direction and conflict: whether every
    surviving pose points the same way, and whether any minority points the
    other way. ``"competing_poses"`` is the honest answer when it does, and it
    is deliberately not resolved by majority vote: by default *any* genuine
    split is reported as competing, and the only way to get a majority call
    out of a split ensemble is for the caller to pass a cut it has calibrated,
    which is then recorded in ``basis``.

    Parameters
    ----------
    poses_faces:
        Per-pose *product configurations*, as ``{pose_id: "R"|"S"|None}``, an
        iterable of such labels, or an iterable of ``(pose_id, label)`` pairs.
        Note that these are configurations, not raw face labels: the same face
        gives opposite configurations for different CIP orderings, so faces must
        be passed through :func:`face_to_configuration` first, per pose, with
        that pose's ranks. Entries of ``None`` (ambiguous face, missing ranks)
        are counted as undetermined and never as agreement.
    target_configuration:
        ``"R"``, ``"S"``, or a :class:`~eagent.schemas.chem.Stereochemistry`.
        ``ACHIRAL`` and ``RACEMIC`` yield ``"not_applicable"`` -- there is no
        target face to favour. ``UNSPECIFIED``/``None`` yields
        ``"insufficient_evidence"``: the model may have an opinion, but with no
        declared target there is nothing to compare it against, and picking the
        majority as "the target" would make the prediction unfalsifiable.
    basis:
        Free-text provenance of how the per-pose calls were produced (method,
        template id, CIP source). Appended to a generated count summary.
    competing_face_fraction:
        Minority-face share at or above which a split is reported as
        ``competing_poses`` rather than resolved by majority. Defaults to
        :data:`~eagent.schemas.candidate.DEFAULT_COMPETING_FACE_FRACTION`
        (0.0), i.e. *any* genuine split is reported as competing. It is a
        parameter -- not a literal inside the schema -- so that a caller who
        has calibrated the cut on their own pose generator can set it, and so
        that the value which decided the call is written into the returned
        call's ``basis`` and from there into provenance. See that constant for
        what calibrating it would have to mean.

    Raises
    ------
    ValueError
        When ``competing_face_fraction`` is outside ``[0, 0.5]``; see
        :meth:`~eagent.schemas.candidate.StereoCall.from_counts`.
    """
    configurations = _iter_configurations(poses_faces)
    n_r = sum(1 for c in configurations if c == "R")
    n_s = sum(1 for c in configurations if c == "S")
    n_undet = sum(1 for c in configurations if c is None)

    if isinstance(target_configuration, Stereochemistry) and target_configuration in (
        Stereochemistry.ACHIRAL, Stereochemistry.RACEMIC
    ):
        target = None
        not_applicable = True
    elif isinstance(target_configuration, str) and target_configuration.strip().lower() in (
        "achiral", "racemic"
    ):
        target = None
        not_applicable = True
    else:
        target = _normalise_target(target_configuration)
        not_applicable = False

    detail = (f"counts: R={n_r}, S={n_s}, undetermined={n_undet}; "
              f"competing-face reporting cut={competing_face_fraction:.3f}"
              + (" (any genuine split is reported as competing_poses)"
                 if competing_face_fraction <= 0.0 else
                 " (calibrated by the caller)")
              + "; pose counts are a sampling artefact of the modelling "
                "protocol, not a population distribution")
    full_basis = f"{basis}; {detail}" if basis else detail

    if not_applicable:
        call = StereoCall(
            call="not_applicable",
            target_face_poses=0,
            opposite_face_poses=0,
            undetermined_poses=len(configurations),
            basis=f"no single target configuration (achiral or racemic target); {full_basis}",
        )
    elif target is None:
        call = StereoCall(
            call="insufficient_evidence",
            target_face_poses=0,
            opposite_face_poses=0,
            undetermined_poses=len(configurations),
            basis=(f"target configuration is unspecified, so no pose can be "
                   f"classified as favouring it; {full_basis}"),
        )
    else:
        on_target = n_r if target == "R" else n_s
        off_target = n_s if target == "R" else n_r
        call = StereoCall.from_counts(
            target=on_target,
            opposite=off_target,
            undetermined=n_undet,
            basis=f"target={target}; {full_basis}",
            competing_face_fraction=competing_face_fraction,
        )

    if call.predicted_ee_pct is not None:  # pragma: no cover - guard, must stay unreachable
        raise FabricationGuardError(
            "call_stereochemistry produced a numeric ee; pose counts cannot "
            "support one"
        )
    return call
