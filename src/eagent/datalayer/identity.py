"""Layered chemical and biological identity, resolved without over-merging.

Why this module exists
----------------------
Two records are merged when somebody decides they describe "the same thing".
In an enzyme project that decision is made many hundreds of times, usually
implicitly, and almost every irrecoverable error in the resulting dataset is
one of these:

* a prose substrate name was turned into a structure, a stereocentre appeared
  somewhere along the way, and the pipeline then optimised for the wrong
  enantiomer;
* "NADH" and "NADPH" were both written down as "NAD-type cofactor", and a
  hydride-transfer geometry was then checked against the wrong molecule, or
  against an oxidised cofactor that cannot donate a hydride at all;
* two different sequences carrying one accession were fused into one row,
  because an accession looked like a primary key;
* two records were merged because the names were similar, the structures
  looked alike, or the substrate strings nearly matched.

So this module keeps identity **layered and explicit**:

* :class:`ChemicalIdentityLadder` retains the author's words, the normalised
  structure, the stereo-defined structure, the charge/protonation state and the
  structure actually modelled as five *separate* rungs, and
  :meth:`ChemicalIdentityLadder.is_consistent` says where information was added
  rather than letting the final rung masquerade as the source.
* :class:`NameResolver` maps names to identifiers only through registered
  sources. When no source answers, it returns an unresolved marker carrying a
  question for the operator; it never guesses and never matches fuzzily.
* :class:`CofactorIdentity` stores the specific species and oxidation state, and
  refuses to answer "is this a hydride donor" when the state is unknown.
* :class:`ProteinIdentity` makes the sequence hash primary, accessions
  secondary and versioned, and the expressed construct a separate field;
  :func:`same_protein` answers only on hash equality.
* :class:`EntityMergePolicy` is a documented, testable object that answers
  whether two records may be merged, and returns the reason when they may not.

Nothing here computes chemistry. Structure comparisons are textual, because no
cheminformatics toolkit is available in this environment; the module therefore
reports "this changed, a curator must say why" instead of silently deciding
that two differently written SMILES are the same molecule.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping

try:  # harness error base; guarded so this module imports standalone
    from ..errors import EAgentError
except Exception:  # pragma: no cover - only when eagent.errors is unavailable
    class EAgentError(Exception):  # type: ignore[no-redef]
        """Fallback base error used when :mod:`eagent.errors` cannot be imported."""

try:  # canonical sequence hash; never re-implemented locally
    from ..provenance import sequence_hash as _sequence_hash
except Exception:  # pragma: no cover - only when eagent.provenance is unavailable
    _sequence_hash = None  # type: ignore[assignment]


__all__ = [
    # errors
    "IdentityError",
    "IdentityRefusedError",
    "NameGuessRefusedError",
    "CofactorStateUnknownError",
    "UnresolvedIdentityError",
    "SequenceHashUnavailableError",
    # chemical ladder
    "IdentityRung",
    "RUNG_ORDER",
    "Representation",
    "StereoDescriptors",
    "stereo_descriptors",
    "net_charge_of",
    "RungValue",
    "InformationAddition",
    "LadderConsistency",
    "ChemicalIdentityLadder",
    # name resolution
    "ResolutionStatus",
    "IdentifierHit",
    "NameResolution",
    "ResolverSource",
    "NameResolver",
    "normalise_query_name",
    # cofactors
    "RedoxState",
    "CofactorSpecies",
    "LOOSE_COFACTOR_LABELS",
    "HydrideDonorAnswer",
    "CofactorRequirementCheck",
    "CofactorIdentity",
    # proteins
    "AccessionRef",
    "ProteinIdentity",
    "ProteinRelation",
    "ProteinComparison",
    "same_protein",
    # merge policy
    "MergeGround",
    "RejectedGround",
    "MergeDecision",
    "EntityMergePolicy",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class IdentityError(EAgentError):
    """Base class for every identity failure raised by this module.

    Exists so a caller can distinguish "I could not establish identity" from a
    generic exception and route it to an operator instead of a retry loop.
    """


class IdentityRefusedError(IdentityError):
    """A question was asked that the available information cannot answer.

    Distinct from a negative answer. Prevents the single worst failure mode in
    this module, which is a confident ``False`` standing in for "unknown".
    """


class NameGuessRefusedError(IdentityRefusedError):
    """Someone asked the resolver to guess an identifier from a name.

    Prevents a trivial name being silently turned into a specific compound.
    "ADH substrate 3", "the ketone" and "acetophenone derivative" resolve to
    nothing, and inventing a CID for them puts a fabricated structure into
    every downstream geometry check.
    """


class CofactorStateUnknownError(IdentityRefusedError):
    """A redox-dependent question was asked about a cofactor of unknown state.

    Prevents the NAD+/NADH confusion that turns an oxidation structure into
    claimed evidence for a reduction mechanism.
    """


class UnresolvedIdentityError(IdentityError):
    """A caller demanded an identifier that no registered source supplied.

    Prevents an unresolved marker being unwrapped with ``or ""`` and travelling
    on as an empty-but-present identifier.
    """


class SequenceHashUnavailableError(IdentityError):
    """A sequence hash was required but :mod:`eagent.provenance` is unavailable.

    Raised rather than computing a locally normalised hash, because a hash with
    different normalisation silently fails to match identical proteins, which is
    worse than having no hash at all.
    """


# ---------------------------------------------------------------------------
# Chemical identity ladder
# ---------------------------------------------------------------------------

class IdentityRung(str, enum.Enum):
    """The five layers of a small-molecule identity, deliberately kept apart.

    Each rung answers a different question and carries a different authority.
    Collapsing them into one "substrate" field is how a prose name becomes a
    stereo-defined, protonated, docked structure with nobody able to say which
    step introduced which assumption.
    """

    def __new__(cls, value: str, order: int, question: str, doc: str,
                cannot_substitute: str) -> "IdentityRung":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.order = order                           # type: ignore[attr-defined]
        obj.question = question                     # type: ignore[attr-defined]
        obj.cannot_substitute = cannot_substitute   # type: ignore[attr-defined]
        obj.__doc__ = doc
        return obj

    AS_WRITTEN = (
        "as_written", 0,
        "What did the author actually write?",
        "The author's original description, verbatim: a trivial name, a catalogue "
        "entry, a phrase from a table header. It is the only rung that is "
        "evidence rather than interpretation, and it is kept unedited so a "
        "curator can always re-read the source.",
        "A name is not a structure; it does not fix stereochemistry, salt form "
        "or protonation.",
    )
    NORMALISED_STRUCTURE = (
        "normalised_structure", 1,
        "Which connectivity does that description denote?",
        "A structure with defined connectivity, obtained by resolving the "
        "description through a registered source. Stereochemistry may still be "
        "absent here.",
        "A connectivity-only structure is not the assayed material and does not "
        "distinguish enantiomers.",
    )
    STEREO_DEFINED_STRUCTURE = (
        "stereo_defined_structure", 2,
        "Which stereoisomer is meant?",
        "The structure with stereochemistry stated. For an asymmetric-reduction "
        "project this rung carries the entire objective, so it is never inferred "
        "from the reaction class or filled in by a toolkit default.",
        "A stereo-defined structure does not state the charge or protonation "
        "state used in modelling.",
    )
    CHARGE_AND_PROTONATION_STATE = (
        "charge_and_protonation_state", 3,
        "Which charge and protonation state, at which pH?",
        "The specific microspecies: net charge and protonation. Docking and "
        "geometry are run on one microspecies, and the choice changes hydrogen "
        "bonding and therefore the measured geometry.",
        "A protonation assignment is a modelling choice, not a measurement of "
        "the compound in the flask.",
    )
    MODELLED_STRUCTURE = (
        "modelled_structure", 4,
        "What was actually fed to the modelling tool?",
        "The structure that entered docking or co-folding, after any tool-side "
        "edit: added hydrogens, a chosen tautomer, a stripped counter-ion, a "
        "substituted analogue. It is kept separate because this is the molecule "
        "every geometric number actually describes.",
        "The modelled structure is not the author's compound; differences "
        "between this rung and the ones below it are assumptions, not data.",
    )

    @property
    def doc(self) -> str:
        """The member docstring, exposed for CLI and report rendering."""
        return self.__doc__ or ""

    def describe(self) -> str:
        """One-line rendering used in consistency reports."""
        return (f"{self.value}: {self.question} "
                f"Not a substitute: {self.cannot_substitute}")


#: Rungs in ascending order of interpretation. Reports always list all five.
RUNG_ORDER: tuple[IdentityRung, ...] = tuple(
    sorted(IdentityRung, key=lambda r: r.order)
)


class Representation(str, enum.Enum):
    """How a rung's value is written down.

    Needed because stereochemistry and charge can be read from a SMILES or an
    InChI string and cannot be read from a trivial name. Marking a name as
    ``FREE_TEXT`` is what lets the ladder answer "unknown" instead of "absent",
    and those two are not the same claim.
    """

    FREE_TEXT = "free_text"
    SMILES = "smiles"
    INCHI = "inchi"
    MOLBLOCK = "molblock"
    OTHER = "other"
    UNKNOWN = "unknown"

    @property
    def is_structural(self) -> bool:
        """Whether the representation encodes a structure rather than prose."""
        return self in (Representation.SMILES, Representation.INCHI,
                        Representation.MOLBLOCK)

    @property
    def stereo_readable(self) -> bool:
        """Whether stereochemistry can be read from the string itself."""
        return self in (Representation.SMILES, Representation.INCHI)


@dataclass(frozen=True)
class StereoDescriptors:
    """Counts of stereo markers found in a structure string.

    Tetrahedral and double-bond markers are counted separately because only the
    first is a stereocentre. The ladder's stereocentre warning must not fire on
    an E/Z marker, and an E/Z addition must not be hidden either.
    """

    tetrahedral: int = 0
    double_bond: int = 0

    @property
    def any(self) -> bool:
        """Whether any stereochemistry at all is encoded."""
        return bool(self.tetrahedral or self.double_bond)

    def as_dict(self) -> dict[str, int]:
        """Plain mapping for serialisation into a report."""
        return {"tetrahedral": self.tetrahedral, "double_bond": self.double_bond}


_SMILES_TETRAHEDRAL = re.compile(r"@@|@")
_SMILES_DOUBLE_BOND = re.compile(r"[/\\]")
_INCHI_TET_LAYER = re.compile(r"/[tms][^/]*")
_INCHI_DB_LAYER = re.compile(r"/b[^/]*")
_SMILES_BRACKET_ATOM = re.compile(r"\[([^\]]*)\]")
_BRACKET_CHARGE = re.compile(r"(\+{1,4}|-{1,4})(\d*)\s*$")


def stereo_descriptors(value: str | None,
                       representation: Representation) -> StereoDescriptors | None:
    """Read stereo markers out of a structure string, or return ``None``.

    ``None`` means "cannot be determined from this representation" -- for a
    trivial name, a molblock (whose parity flags need a parser) or an unknown
    format. Returning ``None`` rather than ``StereoDescriptors()`` is the whole
    point: "no stereochemistry stated" and "we cannot see the stereochemistry"
    lead to different actions, and only the second needs a curator.
    """
    if value is None or not str(value).strip():
        return None
    if not representation.stereo_readable:
        return None
    text = str(value)
    if representation is Representation.SMILES:
        return StereoDescriptors(
            tetrahedral=len(_SMILES_TETRAHEDRAL.findall(text)),
            double_bond=len(_SMILES_DOUBLE_BOND.findall(text)),
        )
    # InChI: stereochemistry lives in the /t, /m, /s and /b layers.
    body = text.split("/", 1)[1] if text.lower().startswith("inchi=") else text
    body = "/" + body
    return StereoDescriptors(
        tetrahedral=len(_INCHI_TET_LAYER.findall(body)),
        double_bond=len(_INCHI_DB_LAYER.findall(body)),
    )


def net_charge_of(value: str | None,
                  representation: Representation) -> int | None:
    """Sum the formal charges written in a structure string, or ``None``.

    This is a textual read of bracket atoms in a SMILES (or the ``/q`` layer of
    an InChI), not a chemistry calculation. Any token it cannot parse makes the
    whole answer ``None``, because a partially parsed charge is a fabricated
    number, and the charge rung exists precisely so that nobody has to guess.
    """
    if value is None or not str(value).strip():
        return None
    text = str(value)
    if representation is Representation.SMILES:
        total = 0
        for body in _SMILES_BRACKET_ATOM.findall(text):
            m = _BRACKET_CHARGE.search(body)
            if not m:
                continue
            signs, digits = m.group(1), m.group(2)
            sign = 1 if signs[0] == "+" else -1
            if digits:
                try:
                    total += sign * int(digits)
                except ValueError:  # pragma: no cover - regex forbids this
                    return None
            else:
                total += sign * len(signs)
        return total
    if representation is Representation.INCHI:
        q = re.search(r"/q([^/]*)", text)
        if not q:
            return None
        total = 0
        for token in re.findall(r"[+-]\d+", q.group(1)):
            try:
                total += int(token)
            except ValueError:  # pragma: no cover
                return None
        return total
    return None


@dataclass(frozen=True)
class RungValue:
    """One rung of a chemical identity ladder, with how it was obtained.

    ``produced_by`` and ``source`` are separate on purpose: a value produced by
    a toolkit with no cited source is an assumption, and the consistency report
    says so rather than treating the tool's output as a reading of the paper.
    """

    rung: IdentityRung
    value: str
    representation: Representation = Representation.UNKNOWN
    produced_by: str | None = None
    source: str | None = None
    declared_stereo: bool | None = None
    declared_charge: int | None = None
    is_assumption: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.value).strip():
            raise IdentityError(
                f"rung '{self.rung.value}' was given an empty value; omit the "
                f"rung instead, so the ladder can report it as absent"
            )
        object.__setattr__(self, "rung", IdentityRung(self.rung))
        object.__setattr__(self, "representation",
                           Representation(self.representation))

    # -- reads -------------------------------------------------------------
    def descriptors(self) -> StereoDescriptors | None:
        """Stereo markers readable from this rung's string, or ``None``."""
        return stereo_descriptors(self.value, self.representation)

    def has_defined_stereocentre(self) -> bool | None:
        """Tri-state: stereocentre present, absent, or not determinable here."""
        if self.declared_stereo is not None:
            return bool(self.declared_stereo)
        d = self.descriptors()
        if d is None:
            return None
        return d.tetrahedral > 0

    def has_defined_double_bond_stereo(self) -> bool | None:
        """Tri-state answer for E/Z geometry, kept apart from stereocentres."""
        d = self.descriptors()
        if d is None:
            return None
        return d.double_bond > 0

    def net_charge(self) -> int | None:
        """Declared charge if given, else one read from the string, else ``None``."""
        if self.declared_charge is not None:
            return int(self.declared_charge)
        return net_charge_of(self.value, self.representation)

    @property
    def is_sourced(self) -> bool:
        """Whether this rung cites where its value came from."""
        return bool(self.source and str(self.source).strip())

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form used in reports and manifests."""
        return {
            "rung": self.rung.value,
            "value": self.value,
            "representation": self.representation.value,
            "produced_by": self.produced_by,
            "source": self.source,
            "is_assumption": self.is_assumption,
            "stereocentre": self.has_defined_stereocentre(),
            "net_charge": self.net_charge(),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class InformationAddition:
    """A place where a rung states something the rung below it did not.

    Every addition is an assumption until a source is cited. Recording them
    individually is what lets a reviewer ask "who decided the substrate was the
    (S) enantiomer?" and get an answer, instead of finding a stereo-defined
    SMILES with no history.
    """

    from_rung: IdentityRung | None
    to_rung: IdentityRung
    kind: str
    detail: str
    is_assumption: bool = True
    needs_curation: bool = True
    question: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form used in reports."""
        return {
            "from_rung": self.from_rung.value if self.from_rung else None,
            "to_rung": self.to_rung.value,
            "kind": self.kind,
            "detail": self.detail,
            "is_assumption": self.is_assumption,
            "needs_curation": self.needs_curation,
            "question": self.question,
        }


@dataclass(frozen=True)
class LadderConsistency:
    """What a ladder contains, what it added, and what must be confirmed.

    Returned by :meth:`ChemicalIdentityLadder.is_consistent`. ``consistent`` is
    deliberately *not* a quality score: a ladder can be perfectly consistent and
    still rest entirely on assumptions, which is why the additions travel with
    the verdict.
    """

    subject: str
    rungs_present: tuple[IdentityRung, ...]
    rungs_absent: tuple[IdentityRung, ...]
    additions: tuple[InformationAddition, ...]
    stereocentre_introduced_during_normalisation: bool
    stereocentre_notes: tuple[str, ...]
    problems: tuple[str, ...]
    questions: tuple[str, ...]
    needs_curation: bool

    @property
    def consistent(self) -> bool:
        """True when no rung contradicts another. Says nothing about evidence."""
        return not self.problems

    @property
    def assumptions(self) -> tuple[InformationAddition, ...]:
        """Additions that cite no source, i.e. the unsupported decisions."""
        return tuple(a for a in self.additions if a.is_assumption)

    def additions_of_kind(self, kind: str) -> tuple[InformationAddition, ...]:
        """Additions filtered by kind, e.g. ``'stereochemistry'``."""
        return tuple(a for a in self.additions if a.kind == kind)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the run manifest."""
        return {
            "subject": self.subject,
            "consistent": self.consistent,
            "rungs_present": [r.value for r in self.rungs_present],
            "rungs_absent": [r.value for r in self.rungs_absent],
            "additions": [a.as_dict() for a in self.additions],
            "stereocentre_introduced_during_normalisation":
                self.stereocentre_introduced_during_normalisation,
            "stereocentre_notes": list(self.stereocentre_notes),
            "problems": list(self.problems),
            "questions": list(self.questions),
            "needs_curation": self.needs_curation,
        }

    def report_lines(self) -> list[str]:
        """Human-readable rendering that always lists all five rungs."""
        lines = [f"chemical identity ladder for {self.subject}"]
        present = set(self.rungs_present)
        for rung in RUNG_ORDER:
            mark = "have " if rung in present else "ABSENT"
            lines.append(f"  {mark} {rung.value}")
        for add in self.additions:
            src = "assumption" if add.is_assumption else "sourced"
            frm = add.from_rung.value if add.from_rung else "-"
            lines.append(f"  added [{add.kind}/{src}] {frm} -> "
                         f"{add.to_rung.value}: {add.detail}")
        for note in self.stereocentre_notes:
            lines.append(f"  STEREO {note}")
        for p in self.problems:
            lines.append(f"  PROBLEM {p}")
        for q in self.questions:
            lines.append(f"  curator question: {q}")
        return lines

    def describe(self) -> str:
        """Report lines joined for printing."""
        return "\n".join(self.report_lines())


@dataclass
class ChemicalIdentityLadder:
    """Five rungs of chemical identity, retained separately and never collapsed.

    There is deliberately no ``.structure`` property. A caller must say which
    rung it wants, because "the substrate" means the author's name to a curator,
    the stereo-defined structure to a chemist and the protonated, hydrogen-added
    ligand to a docking program, and silently returning the last of those when
    the first was meant is how an ee target gets attached to the wrong molecule.
    """

    subject: str
    rungs: dict[IdentityRung, RungValue] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- construction ------------------------------------------------------
    def set_rung(self, value: RungValue) -> "ChemicalIdentityLadder":
        """Attach (or replace) one rung. Returns ``self`` for chaining."""
        if not isinstance(value, RungValue):
            raise IdentityError(
                "set_rung expects a RungValue carrying its own provenance; a "
                "bare string would lose how the value was obtained"
            )
        self.rungs[value.rung] = value
        return self

    def add(self, rung: IdentityRung | str, value: str, **kwargs: Any) \
            -> "ChemicalIdentityLadder":
        """Convenience wrapper around :meth:`set_rung` for one rung."""
        return self.set_rung(RungValue(IdentityRung(rung), value, **kwargs))

    # -- reads -------------------------------------------------------------
    def get(self, rung: IdentityRung | str) -> RungValue | None:
        """The rung if present, else ``None``. Never falls back to another rung."""
        return self.rungs.get(IdentityRung(rung))

    def require(self, rung: IdentityRung | str) -> RungValue:
        """The rung, or :class:`IdentityError` naming what is missing."""
        r = IdentityRung(rung)
        got = self.rungs.get(r)
        if got is None:
            raise IdentityError(
                f"{self.subject}: rung '{r.value}' is not populated "
                f"({r.question}); no other rung may stand in for it"
            )
        return got

    def present(self) -> tuple[IdentityRung, ...]:
        """Populated rungs, in ladder order."""
        return tuple(r for r in RUNG_ORDER if r in self.rungs)

    def absent(self) -> tuple[IdentityRung, ...]:
        """Unpopulated rungs, in ladder order."""
        return tuple(r for r in RUNG_ORDER if r not in self.rungs)

    def structure_for_modelling(self) -> RungValue | None:
        """The rung a modelling tool may consume, or ``None``.

        Only :attr:`IdentityRung.MODELLED_STRUCTURE` qualifies. Returning
        ``None`` instead of the nearest structural rung prevents a
        connectivity-only SMILES being docked as though stereochemistry and
        protonation had been decided.
        """
        return self.rungs.get(IdentityRung.MODELLED_STRUCTURE)

    # -- the consistency report -------------------------------------------
    def is_consistent(self) -> LadderConsistency:
        """Report which rungs exist and where information was added.

        Walks the populated rungs in order and compares each with the one below
        it. Four kinds of addition are recognised:

        ``structure_from_text``
            a prose description became a structure;
        ``stereochemistry``
            a stereocentre or an E/Z assignment appeared;
        ``charge_state``
            a net charge appeared or changed;
        ``modelling_edit``
            the modelled string differs from the rung below it.

        An addition between the as-written rung and the normalised rung sets
        :attr:`LadderConsistency.stereocentre_introduced_during_normalisation`,
        because normalisation is supposed to canonicalise what the author wrote,
        not decide which enantiomer they meant. That flag is the specific thing
        this method exists to catch.

        Structure strings are compared textually. Two different writings of one
        molecule are therefore reported as a change needing curation rather than
        quietly accepted, which is the safe direction for the error.
        """
        present = self.present()
        additions: list[InformationAddition] = []
        problems: list[str] = []
        questions: list[str] = []
        stereo_notes: list[str] = []
        normalisation_stereo = False

        if IdentityRung.AS_WRITTEN not in self.rungs:
            questions.append(
                "What did the source actually write for this compound? The "
                "as-written rung is absent, so no rung above it can be checked "
                "against the source."
            )

        for idx, rung in enumerate(present):
            upper = self.rungs[rung]
            lower = self.rungs[present[idx - 1]] if idx else None

            # -- prose -> structure
            if (lower is not None
                    and lower.representation is Representation.FREE_TEXT
                    and upper.representation.is_structural):
                additions.append(InformationAddition(
                    from_rung=lower.rung, to_rung=rung,
                    kind="structure_from_text",
                    detail=(f"a structure ({upper.representation.value}) was "
                            f"derived from the prose description "
                            f"{lower.value!r}"),
                    is_assumption=not upper.is_sourced,
                    needs_curation=not upper.is_sourced,
                    question=("Which registered source maps "
                              f"{lower.value!r} to {upper.value!r}?")
                    if not upper.is_sourced else "",
                ))

            # -- stereochemistry
            up_stereo = upper.has_defined_stereocentre()
            low_stereo = lower.has_defined_stereocentre() if lower else None
            if up_stereo is True and low_stereo is not True:
                basis = ("the rung below states none" if low_stereo is False
                         else ("there is no rung below to compare against"
                               if lower is None
                               else "the rung below cannot be read for "
                                    "stereochemistry"))
                sourced = upper.is_sourced
                detail = (f"a stereocentre is defined at '{rung.value}' while "
                          f"{basis}")
                additions.append(InformationAddition(
                    from_rung=lower.rung if lower else None, to_rung=rung,
                    kind="stereochemistry", detail=detail,
                    is_assumption=not sourced, needs_curation=not sourced,
                    question=("Who determined the configuration, and from which "
                              "source?") if not sourced else "",
                ))
                note = (f"stereocentre present at rung '{rung.value}' "
                        f"({basis}); "
                        + ("sourced to " + str(upper.source) if sourced
                           else "this is an assumption, not a reading of the "
                                "source"))
                stereo_notes.append(note)
                if rung is IdentityRung.NORMALISED_STRUCTURE:
                    normalisation_stereo = True
                    stereo_notes.append(
                        "the stereocentre was introduced during normalisation: "
                        "normalisation canonicalises what the author wrote, so "
                        "a configuration appearing here is an assumption about "
                        "which enantiomer was meant and must be confirmed "
                        "against the source"
                    )
                    questions.append(
                        "Normalisation produced a stereocentre the source did "
                        "not state. Was the starting material a single "
                        "enantiomer, and which one?"
                    )
            elif up_stereo is None and upper.representation.is_structural:
                questions.append(
                    f"Rung '{rung.value}' is a {upper.representation.value} "
                    f"whose stereochemistry cannot be read here; is a "
                    f"configuration defined?"
                )

            up_ez = upper.has_defined_double_bond_stereo()
            low_ez = lower.has_defined_double_bond_stereo() if lower else None
            if up_ez is True and low_ez is False:
                additions.append(InformationAddition(
                    from_rung=lower.rung if lower else None, to_rung=rung,
                    kind="stereochemistry",
                    detail=(f"double-bond (E/Z) geometry appears at "
                            f"'{rung.value}' and is absent below it"),
                    is_assumption=not upper.is_sourced,
                    needs_curation=not upper.is_sourced,
                ))

            # -- stereochemistry silently dropped is a contradiction
            if up_stereo is False and low_stereo is True:
                problems.append(
                    f"rung '{rung.value}' drops the stereochemistry defined at "
                    f"'{lower.rung.value}'; the more interpreted rung must not "
                    f"lose information the rung below it carried"
                )

            # -- charge
            up_q = upper.net_charge()
            low_q = lower.net_charge() if lower else None
            # A neutral reading of a structural string is the default, not a
            # decision, so only a non-zero charge appearing out of nowhere is
            # reported as an addition. Reporting "+0 was assigned" on every
            # ordinary SMILES would bury the real protonation decisions.
            if (up_q is not None and low_q is None and lower is not None
                    and up_q != 0):
                additions.append(InformationAddition(
                    from_rung=lower.rung, to_rung=rung, kind="charge_state",
                    detail=f"a net charge of {up_q:+d} was assigned at "
                           f"'{rung.value}' and is not stated below it",
                    is_assumption=not upper.is_sourced,
                    needs_curation=not upper.is_sourced,
                    question="At which pH, and by which protonation model?"
                    if not upper.is_sourced else "",
                ))
            elif (up_q is not None and low_q is not None and up_q != low_q):
                if rung is IdentityRung.CHARGE_AND_PROTONATION_STATE:
                    additions.append(InformationAddition(
                        from_rung=lower.rung, to_rung=rung, kind="charge_state",
                        detail=f"charge changed from {low_q:+d} to {up_q:+d} "
                               f"when the protonation state was assigned",
                        is_assumption=not upper.is_sourced,
                        needs_curation=not upper.is_sourced,
                    ))
                else:
                    problems.append(
                        f"net charge changes from {low_q:+d} at "
                        f"'{lower.rung.value}' to {up_q:+d} at '{rung.value}' "
                        f"outside the charge rung; a protonation decision taken "
                        f"somewhere else is untraceable"
                    )

            # -- modelling edit
            if rung is IdentityRung.MODELLED_STRUCTURE and lower is not None:
                if _squash(upper.value) != _squash(lower.value):
                    additions.append(InformationAddition(
                        from_rung=lower.rung, to_rung=rung,
                        kind="modelling_edit",
                        detail=(f"the modelled string differs from "
                                f"'{lower.rung.value}'; every geometric number "
                                f"describes the modelled molecule, not the one "
                                f"below"),
                        is_assumption=not upper.is_sourced,
                        needs_curation=True,
                        question=("Which tool edited the structure, and was the "
                                  "edit (hydrogens, tautomer, counter-ion, "
                                  "analogue) intended?"),
                    ))

        # -- ladder-level problems
        modelled = self.rungs.get(IdentityRung.MODELLED_STRUCTURE)
        if modelled is not None:
            if IdentityRung.STEREO_DEFINED_STRUCTURE not in self.rungs \
                    and modelled.has_defined_stereocentre() is True:
                problems.append(
                    "the modelled structure defines a stereocentre but no "
                    "stereo-defined rung records where that configuration came "
                    "from"
                )
            if IdentityRung.CHARGE_AND_PROTONATION_STATE not in self.rungs:
                questions.append(
                    "Which protonation state was modelled? The charge rung is "
                    "absent, so the microspecies behind the geometry is "
                    "unrecorded."
                )
        for rv in self.rungs.values():
            if rv.is_assumption and not rv.is_sourced:
                questions.append(
                    f"Rung '{rv.rung.value}' is marked as an assumption and "
                    f"cites no source; who is responsible for it?"
                )

        needs_curation = bool(problems) or bool(questions) or any(
            a.needs_curation for a in additions)
        return LadderConsistency(
            subject=self.subject,
            rungs_present=present,
            rungs_absent=self.absent(),
            additions=tuple(additions),
            stereocentre_introduced_during_normalisation=normalisation_stereo,
            stereocentre_notes=tuple(stereo_notes),
            problems=tuple(problems),
            questions=tuple(dict.fromkeys(questions)),
            needs_curation=needs_curation,
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form holding every rung separately."""
        return {
            "subject": self.subject,
            "rungs": {r.value: self.rungs[r].as_dict() for r in self.present()},
            "absent_rungs": [r.value for r in self.absent()],
            "notes": list(self.notes),
        }


def _squash(text: str) -> str:
    """Whitespace-insensitive form used for textual structure comparison."""
    return "".join(str(text).split())


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def normalise_query_name(name: str) -> str:
    """Case- and whitespace-normalise a query name, and nothing else.

    Stereo prefixes (``(R)-``, ``(S)-``, ``rac-``, ``(+/-)``), salt suffixes and
    numbering are all left intact. Stripping them is the usual "normalisation"
    that merges enantiomers and salt forms into one entry, which for an
    asymmetric-reduction project destroys the objective itself.
    """
    return " ".join(str(name).split()).casefold()


class ResolutionStatus(str, enum.Enum):
    """Outcome of a name-to-identifier lookup.

    The three unresolved outcomes are kept apart because they need different
    actions: nobody has the name, one source offers several answers, or two
    sources disagree.
    """

    RESOLVED = "resolved"
    UNRESOLVED_NOT_FOUND = "unresolved_not_found"
    UNRESOLVED_AMBIGUOUS = "unresolved_ambiguous"
    UNRESOLVED_CONFLICTING_SOURCES = "unresolved_conflicting_sources"
    REFUSED_GUESS = "refused_guess"

    @property
    def resolved(self) -> bool:
        """Whether an identifier may be used downstream."""
        return self is ResolutionStatus.RESOLVED


@dataclass(frozen=True)
class IdentifierHit:
    """One identifier offered by one registered source, with its provenance.

    The source id and version travel with the hit because a resolution made
    against last year's release is not the same fact as one made today, and the
    manifest has to be able to say which was used.
    """

    query: str
    identifier_type: str
    identifier: str
    source_id: str
    source_version: str | None = None
    source_record_id: str | None = None
    license: str | None = None
    retrieved_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the manifest."""
        return {
            "query": self.query,
            "identifier_type": self.identifier_type,
            "identifier": self.identifier,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_record_id": self.source_record_id,
            "license": self.license,
            "retrieved_at": self.retrieved_at,
        }


@dataclass(frozen=True)
class NameResolution:
    """The result of a lookup: either an identifier, or the unresolved marker.

    The unresolved form is a value, not an exception, so a pipeline can collect
    every unanswered name and put one question list in front of the operator
    instead of dying on the first one. :meth:`require` is the opposite door, for
    callers that cannot continue without an answer.
    """

    query: str
    status: ResolutionStatus
    hits: tuple[IdentifierHit, ...] = ()
    consulted_sources: tuple[str, ...] = ()
    question: str = ""
    reason: str = ""
    needs_curation: bool = True

    def __bool__(self) -> bool:
        return self.status.resolved

    @property
    def resolved(self) -> bool:
        """Whether an identifier may be used downstream."""
        return self.status.resolved

    @property
    def identifier(self) -> str | None:
        """The identifier, or ``None`` for every unresolved status."""
        return self.hits[0].identifier if self.resolved and self.hits else None

    def require(self) -> IdentifierHit:
        """The hit, or :class:`UnresolvedIdentityError` carrying the question."""
        if self.resolved and self.hits:
            return self.hits[0]
        raise UnresolvedIdentityError(
            f"'{self.query}' is unresolved ({self.status.value}): "
            f"{self.reason} Operator question: {self.question}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the manifest."""
        return {
            "query": self.query,
            "status": self.status.value,
            "identifier": self.identifier,
            "hits": [h.as_dict() for h in self.hits],
            "consulted_sources": list(self.consulted_sources),
            "reason": self.reason,
            "question": self.question,
            "needs_curation": self.needs_curation,
        }


@dataclass(frozen=True)
class ResolverSource:
    """A registered lookup table or callable that maps names to identifiers.

    Registration is mandatory: the resolver has no built-in chemical knowledge,
    so a name resolves only if some curated, versioned source says so. That is
    the mechanism preventing a plausible identifier being produced from nothing.
    """

    source_id: str
    identifier_type: str
    version: str | None = None
    license: str | None = None
    retrieved_at: str | None = None
    table: Mapping[str, str] = field(default_factory=dict)
    lookup: Callable[[str], Any] | None = None

    def candidates(self, normalised_query: str) -> tuple[str, ...]:
        """Identifiers this source offers for an already-normalised name."""
        found: list[str] = []
        raw = self.table.get(normalised_query)
        if raw is not None:
            found.append(str(raw))
        if self.lookup is not None:
            got = self.lookup(normalised_query)
            if isinstance(got, str):
                found.append(got)
            elif isinstance(got, Iterable) and not isinstance(got, (bytes, dict)):
                found.extend(str(x) for x in got)
        return tuple(dict.fromkeys(f for f in found if str(f).strip()))


class NameResolver:
    """Maps names to identifiers through registered sources only.

    Three properties make this safe where a generic "name lookup" is not:

    * no source, no answer -- the resolver returns an unresolved marker with a
      question for the operator rather than the closest thing it can find;
    * no fuzzy matching -- lookups are exact on the case- and whitespace-
      normalised string, so ``"1-phenylethanol"`` never answers for
      ``"(S)-1-phenylethanol"``;
    * disagreement is an outcome -- two sources returning different identifiers
      produce ``UNRESOLVED_CONFLICTING_SOURCES``, not a silent first-wins.
    """

    def __init__(self, sources: Iterable[ResolverSource] = ()) -> None:
        self._sources: dict[str, ResolverSource] = {}
        for s in sources:
            self.register(s)

    def register(self, source: ResolverSource) -> ResolverSource:
        """Add a source. Duplicate ids are refused, so provenance stays unique."""
        if source.source_id in self._sources:
            raise IdentityError(
                f"resolver source '{source.source_id}' is already registered; "
                f"two sources sharing an id make a resolution untraceable"
            )
        self._sources[source.source_id] = source
        return source

    def sources(self) -> list[str]:
        """Registered source ids, sorted."""
        return sorted(self._sources)

    def resolve(self, name: str, *, identifier_type: str | None = None) \
            -> NameResolution:
        """Resolve a name, or return the unresolved marker with a question.

        ``identifier_type`` restricts the lookup to sources of that type (for
        example ``"inchikey"``), so a caller needing a structure key does not
        receive a database accession that happens to be registered.
        """
        query = str(name)
        norm = normalise_query_name(query)
        if not norm:
            return NameResolution(
                query=query, status=ResolutionStatus.UNRESOLVED_NOT_FOUND,
                consulted_sources=(), reason="the name is empty",
                question="Which compound was meant? No name was supplied.",
            )

        consulted = tuple(
            s.source_id for s in self._sources.values()
            if identifier_type is None or s.identifier_type == identifier_type
        )
        if not consulted:
            return NameResolution(
                query=query, status=ResolutionStatus.UNRESOLVED_NOT_FOUND,
                consulted_sources=(),
                reason=("no source is registered"
                        + (f" for identifier type '{identifier_type}'"
                           if identifier_type else "")),
                question=(f"Which registered source should map '{query}' to an "
                          f"identifier? None is configured, and the resolver "
                          f"does not guess."),
            )

        hits: list[IdentifierHit] = []
        for sid in consulted:
            src = self._sources[sid]
            for ident in src.candidates(norm):
                hits.append(IdentifierHit(
                    query=query, identifier_type=src.identifier_type,
                    identifier=ident, source_id=src.source_id,
                    source_version=src.version, license=src.license,
                    retrieved_at=src.retrieved_at,
                ))

        if not hits:
            return NameResolution(
                query=query, status=ResolutionStatus.UNRESOLVED_NOT_FOUND,
                consulted_sources=consulted,
                reason=(f"no registered source holds the exact name "
                        f"'{norm}'; lookups are exact, because a near match "
                        f"between compound names is not the same compound"),
                question=(f"What is the structure of '{query}'? Supply an "
                          f"identifier (InChIKey/ChEBI/CID) or an isomeric "
                          f"SMILES, or register a source that contains it."),
            )

        distinct = {h.identifier.strip().upper() for h in hits}
        if len(distinct) > 1:
            by_source = ", ".join(f"{h.source_id}={h.identifier}" for h in hits)
            status = (ResolutionStatus.UNRESOLVED_CONFLICTING_SOURCES
                      if len({h.source_id for h in hits}) > 1
                      else ResolutionStatus.UNRESOLVED_AMBIGUOUS)
            return NameResolution(
                query=query, status=status, hits=tuple(hits),
                consulted_sources=consulted,
                reason=f"registered sources disagree: {by_source}",
                question=(f"Which identifier is correct for '{query}'? "
                          f"{len(distinct)} different answers were returned and "
                          f"the resolver will not pick one."),
            )

        return NameResolution(
            query=query, status=ResolutionStatus.RESOLVED, hits=tuple(hits),
            consulted_sources=consulted,
            reason=(f"exact match in "
                    f"{', '.join(sorted({h.source_id for h in hits}))}"),
            question="", needs_curation=False,
        )

    def guess(self, name: str, detail: str = "") -> NameResolution:
        """Always raise :class:`NameGuessRefusedError`.

        The method exists so the refusal has an address. Code tempted to fall
        back on a fuzzy match calls this and gets a named error naming the
        compound, instead of inventing an identifier three layers down.
        """
        raise NameGuessRefusedError(
            f"refusing to guess an identifier for '{name}': a name is not a "
            f"structure, and a guessed compound propagates into every geometry "
            f"check downstream. Register a source or ask the operator. {detail}"
            .strip()
        )


# ---------------------------------------------------------------------------
# Cofactors
# ---------------------------------------------------------------------------

class RedoxState(str, enum.Enum):
    """Oxidation state of a redox cofactor, with ``UNKNOWN`` as a real value.

    ``UNKNOWN`` exists so that an unstated state cannot be defaulted to the
    convenient one. NAD(P)+ in a template is not NAD(P)H, and a hydride-transfer
    geometry measured against the oxidised form measures nothing.
    """

    REDUCED = "reduced"
    OXIDIZED = "oxidized"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"

    @property
    def is_known(self) -> bool:
        """Whether the state is determined enough to answer redox questions."""
        return self in (RedoxState.REDUCED, RedoxState.OXIDIZED,
                        RedoxState.NOT_APPLICABLE)


class CofactorSpecies(str, enum.Enum):
    """A specific cofactor species, never a family label.

    ``NADH``, ``NADPH``, ``NAD+`` and ``NADP+`` are four different molecules
    with different phosphorylation and different redox state. A single
    "NAD-type cofactor" label loses both distinctions at once: the specificity
    that decides whether an enzyme will work in a whole-cell system, and the
    oxidation state that decides whether the reaction can run at all.

    Each member carries the PDB chemical component ids that the repository's
    own chemistry module documents for it. Species whose component ids are not
    documented here carry an empty tuple rather than a guessed code; a guessed
    code would be read back as a fact by the structure layer.
    """

    def __new__(cls, value: str, backbone: str | None, state: str,
                ligand_codes: tuple[str, ...], doc: str) -> "CofactorSpecies":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.backbone = backbone                 # type: ignore[attr-defined]
        obj.state = RedoxState(state)           # type: ignore[attr-defined]
        obj.ligand_codes = ligand_codes         # type: ignore[attr-defined]
        obj.__doc__ = doc
        return obj

    NADH = ("NADH", "NAD", "reduced", ("NAI",),
            "Reduced, non-phosphorylated nicotinamide cofactor; hydride donor.")
    NAD_PLUS = ("NAD+", "NAD", "oxidized", ("NAD",),
                "Oxidised, non-phosphorylated nicotinamide cofactor; hydride "
                "acceptor, so it supports oxidation, not reduction.")
    NADPH = ("NADPH", "NADP", "reduced", ("NDP",),
             "Reduced, 2'-phosphorylated nicotinamide cofactor; hydride donor "
             "and the usual biosynthetic reductant.")
    NADP_PLUS = ("NADP+", "NADP", "oxidized", ("NAP",),
                 "Oxidised, 2'-phosphorylated nicotinamide cofactor; hydride "
                 "acceptor.")
    FAD = ("FAD", "FAD", "oxidized", (),
           "Oxidised flavin adenine dinucleotide. Component ids are not "
           "recorded here and need curation before use as a structural key.")
    FADH2 = ("FADH2", "FAD", "reduced", (),
             "Reduced flavin adenine dinucleotide. Component ids need curation.")
    FMN = ("FMN", "FMN", "oxidized", (),
           "Oxidised flavin mononucleotide. Component ids need curation.")
    FMNH2 = ("FMNH2", "FMN", "reduced", (),
             "Reduced flavin mononucleotide. Component ids need curation.")
    UNKNOWN = ("unknown", None, "unknown", (),
               "The species has not been determined. Carried explicitly so a "
               "loose label cannot be promoted to a specific molecule.")

    @property
    def doc(self) -> str:
        """The member docstring, exposed for reports."""
        return self.__doc__ or ""

    @property
    def is_nicotinamide(self) -> bool:
        """Whether this species is an NAD- or NADP-type cofactor."""
        return self.backbone in ("NAD", "NADP")


#: Labels that name a family rather than a molecule. Seeing one of these is not
#: a failure of the writer; it is a signal that the specific species was never
#: recorded and has to be asked for before any redox reasoning happens.
LOOSE_COFACTOR_LABELS: tuple[str, ...] = (
    "nad-type cofactor", "nad type cofactor", "nad cofactor",
    "nad(p)", "nad(p)h", "nad(p)+", "nadh/nadph", "nad/nadp",
    "nicotinamide cofactor", "nicotinamide", "pyridine nucleotide",
    "flavin", "flavin cofactor", "cofactor", "nad", "nadp",
)

#: Exact, unambiguous spellings the module is willing to read as a species.
_EXACT_COFACTOR_LABELS: dict[str, CofactorSpecies] = {
    "nadh": CofactorSpecies.NADH,
    "nad+": CofactorSpecies.NAD_PLUS,
    "nad(+)": CofactorSpecies.NAD_PLUS,
    "nadph": CofactorSpecies.NADPH,
    "nadp+": CofactorSpecies.NADP_PLUS,
    "nadp(+)": CofactorSpecies.NADP_PLUS,
    "fad": CofactorSpecies.FAD,
    "fadh2": CofactorSpecies.FADH2,
    "fmn": CofactorSpecies.FMN,
    "fmnh2": CofactorSpecies.FMNH2,
}

_LIGAND_CODE_TO_SPECIES: dict[str, CofactorSpecies] = {
    code: species
    for species in CofactorSpecies
    for code in species.ligand_codes
}


@dataclass(frozen=True)
class HydrideDonorAnswer:
    """A tri-state answer to "can this cofactor donate a hydride?".

    ``answered=False`` is a refusal, not a negative. Encoding the refusal as a
    value lets a gate record "state unknown, cannot evaluate" in its report
    instead of recording a passed or failed check it never actually performed.
    """

    answered: bool
    value: bool | None
    reason: str
    question: str = ""
    needs_curation: bool = False

    def __bool__(self) -> bool:
        return bool(self.answered and self.value)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the gate report."""
        return {
            "answered": self.answered, "value": self.value,
            "reason": self.reason, "question": self.question,
            "needs_curation": self.needs_curation,
        }


@dataclass(frozen=True)
class CofactorRequirementCheck:
    """Whether an observed cofactor satisfies a mechanism's requirement.

    ``satisfied is None`` means the comparison could not be made. It is kept
    distinct from ``False`` so a precondition gate can say "not evaluated" and
    send the candidate back for repair, rather than discarding it as failed.
    """

    satisfied: bool | None
    reason: str
    required: str
    observed: str
    repair_hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the gate report."""
        return {
            "satisfied": self.satisfied, "reason": self.reason,
            "required": self.required, "observed": self.observed,
            "repair_hint": self.repair_hint,
        }


@dataclass(frozen=True)
class CofactorIdentity:
    """A cofactor recorded as a specific species in a specific oxidation state.

    The state is stored even when the species is known, because a structure may
    contain a nicotinamide cofactor whose component id says one thing and whose
    author annotation says another; the conflict must be visible rather than
    resolved by precedence.
    """

    species: CofactorSpecies = CofactorSpecies.UNKNOWN
    state: RedoxState = RedoxState.UNKNOWN
    ligand_code: str | None = None
    label_as_written: str | None = None
    determined_from: str | None = None
    needs_curation: bool = False
    question: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        species = CofactorSpecies(self.species)
        object.__setattr__(self, "species", species)
        state = RedoxState(self.state)
        if state is RedoxState.UNKNOWN and species is not CofactorSpecies.UNKNOWN:
            state = species.state
        object.__setattr__(self, "state", state)
        if species is CofactorSpecies.UNKNOWN and not self.question:
            object.__setattr__(
                self, "question",
                "Which exact cofactor species is this (for example NADH, "
                "NADPH, NAD+ or NADP+), and in which oxidation state?")
            object.__setattr__(self, "needs_curation", True)

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_ligand_code(cls, code: str | None,
                         determined_from: str = "pdb chemical component id") \
            -> "CofactorIdentity":
        """Build from a PDB chemical component id, or return the unknown form.

        Only the component ids documented in this repository's chemistry module
        are recognised. An unrecognised code produces ``UNKNOWN`` with a
        question, never a plausible species, because the component dictionary is
        large and a wrong expansion silently flips an oxidation state.
        """
        if code is None or not str(code).strip():
            return cls(determined_from=determined_from,
                       question="No ligand component id was given; which "
                                "cofactor is bound?")
        token = str(code).strip().upper()
        species = _LIGAND_CODE_TO_SPECIES.get(token)
        if species is None:
            return cls(ligand_code=token, determined_from=determined_from,
                       needs_curation=True,
                       question=(f"Component id '{token}' is not one of the "
                                 f"nicotinamide codes documented here "
                                 f"({', '.join(sorted(_LIGAND_CODE_TO_SPECIES))}); "
                                 f"which species and oxidation state is it?"))
        return cls(species=species, state=species.state, ligand_code=token,
                   determined_from=determined_from)

    @classmethod
    def from_label(cls, label: str | None,
                   determined_from: str = "author label") -> "CofactorIdentity":
        """Build from a written label, refusing family labels.

        ``"NAD-type cofactor"``, ``"NAD(P)H"`` and a bare ``"NAD"`` all describe
        more than one molecule, so they return ``UNKNOWN`` with the question
        attached. This is the specific collapse the class exists to prevent.
        """
        if label is None or not str(label).strip():
            return cls(determined_from=determined_from,
                       question="No cofactor label was given; which cofactor "
                                "does the mechanism use?")
        raw = str(label).strip()
        norm = " ".join(raw.split()).casefold()
        species = _EXACT_COFACTOR_LABELS.get(norm)
        if species is not None:
            return cls(species=species, state=species.state,
                       label_as_written=raw, determined_from=determined_from)
        if norm in LOOSE_COFACTOR_LABELS:
            return cls(label_as_written=raw, determined_from=determined_from,
                       needs_curation=True,
                       question=(f"'{raw}' names a family, not a molecule. Which "
                                 f"species was used -- NADH, NADPH, NAD+ or "
                                 f"NADP+ -- and in which oxidation state?"))
        return cls(label_as_written=raw, determined_from=determined_from,
                   needs_curation=True,
                   question=(f"'{raw}' is not an exact cofactor spelling this "
                             f"module recognises; state the species and "
                             f"oxidation state explicitly."))

    # -- queries -----------------------------------------------------------
    @property
    def is_known(self) -> bool:
        """Whether a specific species has been established."""
        return self.species is not CofactorSpecies.UNKNOWN

    def describe(self) -> str:
        """Compact rendering, e.g. ``NADPH[reduced]`` or ``unknown[unknown]``."""
        return f"{self.species.value}[{self.state.value}]"

    def hydride_donor_answer(self) -> HydrideDonorAnswer:
        """Tri-state answer to the hydride-donor question.

        Returns a refusal when the oxidation state is unknown. Use this in code
        that must keep running; use :meth:`is_hydride_donor` where a wrong
        answer would be unrecoverable.
        """
        if not self.state.is_known:
            return HydrideDonorAnswer(
                answered=False, value=None,
                reason=(f"the oxidation state of "
                        f"{self.species.value if self.is_known else 'this cofactor'} "
                        f"is unknown; a hydride donor and a hydride acceptor "
                        f"differ only in that state"),
                question=(self.question or
                          "Is the bound cofactor the reduced or the oxidised "
                          "form? Check the chemical component id in the source "
                          "structure (NAI/NDP are reduced, NAD/NAP oxidised)."),
                needs_curation=True,
            )
        if self.state is RedoxState.NOT_APPLICABLE:
            return HydrideDonorAnswer(
                answered=True, value=False,
                reason=f"{self.describe()} is not a redox cofactor")
        donor = self.state is RedoxState.REDUCED
        return HydrideDonorAnswer(
            answered=True, value=donor,
            reason=(f"{self.describe()} is "
                    f"{'reduced and can donate' if donor else 'oxidised and cannot donate'}"
                    f" a hydride"))

    def is_hydride_donor(self) -> bool:
        """Whether this cofactor can donate a hydride.

        Raises :class:`CofactorStateUnknownError` when the state is unknown,
        rather than returning ``False``. A silent ``False`` here would reject a
        correct candidate; a silent ``True`` would validate a hydride-transfer
        geometry measured against NAD+.
        """
        ans = self.hydride_donor_answer()
        if not ans.answered:
            raise CofactorStateUnknownError(f"{ans.reason}. {ans.question}")
        return bool(ans.value)

    def satisfies(self, required_species: "CofactorIdentity | CofactorSpecies | str | None",
                  required_state: RedoxState | str | None = None,
                  *, allow_backbone_substitution: bool = False) \
            -> CofactorRequirementCheck:
        """Check this cofactor against a mechanism requirement.

        Used by the cofactor-state precondition gate. Returns
        ``satisfied=None`` whenever either side is unknown, so the gate reports
        "not evaluated" and the candidate goes back for repair instead of being
        failed on missing information.

        ``allow_backbone_substitution`` permits NADH where NADPH is required (a
        specificity question an engineering campaign may legitimately open), but
        never permits an oxidation-state substitution, which is a mechanism
        error rather than a specificity choice.
        """
        req = _as_cofactor_identity(required_species, required_state)
        observed = self.describe()
        required = req.describe()

        if not self.is_known or not self.state.is_known:
            return CofactorRequirementCheck(
                satisfied=None,
                reason=("the modelled cofactor's species or oxidation state is "
                        "unknown, so the mechanism requirement cannot be checked"),
                required=required, observed=observed,
                repair_hint=("resolve the bound cofactor from the chemical "
                             "component id of the source structure, then re-run "
                             "this gate"))
        if not req.is_known or not req.state.is_known:
            return CofactorRequirementCheck(
                satisfied=None,
                reason=("the mechanism template does not state a specific "
                        "cofactor species and oxidation state"),
                required=required, observed=observed,
                repair_hint=("amend the catalytic template to name the exact "
                             "cofactor species and its oxidation state"))

        if self.state is not req.state:
            return CofactorRequirementCheck(
                satisfied=False,
                reason=(f"oxidation state mismatch: the mechanism needs "
                        f"{req.state.value} and the model carries "
                        f"{self.state.value}; this is a mechanism error, not a "
                        f"specificity preference"),
                required=required, observed=observed,
                repair_hint=("rebuild the complex with the correct oxidation "
                             "state, or select a source structure whose "
                             "component id carries it"))
        if self.species is req.species:
            return CofactorRequirementCheck(
                satisfied=True,
                reason=f"{observed} matches the requirement exactly",
                required=required, observed=observed)
        if self.species.backbone == req.species.backbone:
            return CofactorRequirementCheck(
                satisfied=True,
                reason=(f"{observed} and {required} are the same species in the "
                        f"same state"),
                required=required, observed=observed)
        if allow_backbone_substitution:
            return CofactorRequirementCheck(
                satisfied=True,
                reason=(f"{observed} substitutes for {required}: same oxidation "
                        f"state, different phosphorylation, allowed explicitly "
                        f"by the caller"),
                required=required, observed=observed,
                repair_hint=("record the cofactor-specificity substitution as an "
                             "assumption on the candidate"))
        return CofactorRequirementCheck(
            satisfied=False,
            reason=(f"cofactor specificity mismatch: the mechanism needs "
                    f"{required} and the model carries {observed}"),
            required=required, observed=observed,
            repair_hint=("re-model with the required cofactor, or state "
                         "explicitly that a backbone substitution is permitted"))

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the manifest."""
        return {
            "species": self.species.value,
            "state": self.state.value,
            "ligand_code": self.ligand_code,
            "label_as_written": self.label_as_written,
            "determined_from": self.determined_from,
            "needs_curation": self.needs_curation,
            "question": self.question,
        }


def _as_cofactor_identity(
    value: "CofactorIdentity | CofactorSpecies | str | None",
    state: RedoxState | str | None = None,
) -> CofactorIdentity:
    """Coerce a requirement expressed in any accepted form into an identity."""
    if isinstance(value, CofactorIdentity):
        if state is not None:
            return replace(value, state=RedoxState(state))
        return value
    if isinstance(value, CofactorSpecies):
        return CofactorIdentity(species=value,
                                state=RedoxState(state) if state is not None
                                else value.state)
    if value is None:
        return CofactorIdentity(
            state=RedoxState(state) if state is not None else RedoxState.UNKNOWN)
    ident = CofactorIdentity.from_label(str(value))
    if state is not None and ident.is_known:
        return replace(ident, state=RedoxState(state))
    if state is not None:
        return replace(ident, state=RedoxState(state))
    return ident


# ---------------------------------------------------------------------------
# Protein identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AccessionRef:
    """A database accession, which is a *pointer*, not an identity.

    The release or entry version is part of the reference because entries are
    merged, demerged and re-annotated: the same accession string can denote
    different sequences in two releases. An unversioned accession is kept, and
    flagged, rather than rejected -- most literature cites them that way.
    """

    accession: str
    database: str = "uniprot"
    version: str | None = None
    is_primary_in_source: bool | None = None

    def __post_init__(self) -> None:
        if not str(self.accession).strip():
            raise IdentityError("an accession reference needs an accession")
        object.__setattr__(self, "accession", str(self.accession).strip())
        object.__setattr__(self, "database", str(self.database).strip().lower())

    @property
    def is_versioned(self) -> bool:
        """Whether the reference names the release it was read from."""
        return bool(self.version and str(self.version).strip())

    @property
    def token(self) -> str:
        """Compact ``db:accession@version`` rendering used in reports."""
        base = f"{self.database}:{self.accession.upper()}"
        return f"{base}@{self.version}" if self.is_versioned else base

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the manifest."""
        return {"accession": self.accession, "database": self.database,
                "version": self.version, "is_versioned": self.is_versioned,
                "is_primary_in_source": self.is_primary_in_source}


@dataclass(frozen=True)
class ProteinIdentity:
    """A protein identified by sequence hash first and accession second.

    The ordering is not a style preference. An engineered enzyme -- which is
    what this project produces -- usually has no accession at all, and a
    database identifier cannot serve as the identity of a protein that exists
    only in a freezer. The expressed construct is recorded separately from the
    catalytic sequence, because tags, truncations and fusions change what was
    actually measured without changing the catalytic domain.
    """

    label: str | None = None
    sequence: str | None = None
    sequence_sha256: str | None = None
    construct_sequence: str | None = None
    construct_sha256: str | None = None
    construct_description: str | None = None
    accessions: tuple[AccessionRef, ...] = ()
    organism: str | None = None
    is_engineered: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "accessions", tuple(self.accessions))
        if self.sequence and not self.sequence_sha256:
            object.__setattr__(self, "sequence_sha256",
                               _hash_sequence(self.sequence))
        if self.construct_sequence and not self.construct_sha256:
            object.__setattr__(self, "construct_sha256",
                               _hash_sequence(self.construct_sequence))

    # -- queries -----------------------------------------------------------
    @property
    def has_identity(self) -> bool:
        """Whether the primary key (the sequence hash) is available."""
        return bool(self.sequence_sha256)

    def primary_key(self) -> str:
        """The sequence hash, or :class:`IdentityError` naming what is missing.

        Refuses to fall back to an accession. A pipeline keyed on accessions
        merges two different sequences that share one entry, which is exactly
        the error :func:`same_protein` is built to catch.
        """
        if not self.sequence_sha256:
            raise IdentityError(
                f"{self.label or 'protein'} has no sequence hash; an accession "
                f"cannot be used as the primary key, because one accession can "
                f"denote different sequences across releases and an engineered "
                f"variant usually has no accession at all")
        return self.sequence_sha256

    @property
    def construct_differs_from_catalytic_sequence(self) -> bool | None:
        """Tri-state: construct differs, matches, or one of them is absent."""
        if not self.sequence_sha256 or not self.construct_sha256:
            return None
        return self.sequence_sha256 != self.construct_sha256

    def unversioned_accessions(self) -> tuple[AccessionRef, ...]:
        """Accessions cited without a release, which cannot pin a sequence."""
        return tuple(a for a in self.accessions if not a.is_versioned)

    def accession_tokens(self) -> tuple[str, ...]:
        """All accession tokens, for reporting a shared pointer between records."""
        return tuple(a.token for a in self.accessions)

    def shares_accession_with(self, other: "ProteinIdentity") -> tuple[str, ...]:
        """Accession strings (ignoring version) present on both records."""
        mine = {a.accession.upper() for a in self.accessions}
        theirs = {a.accession.upper() for a in other.accessions}
        return tuple(sorted(mine & theirs))

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the manifest."""
        return {
            "label": self.label,
            "sequence_sha256": self.sequence_sha256,
            "construct_sha256": self.construct_sha256,
            "construct_description": self.construct_description,
            "accessions": [a.as_dict() for a in self.accessions],
            "organism": self.organism,
            "is_engineered": self.is_engineered,
            "has_identity": self.has_identity,
            "construct_differs_from_catalytic_sequence":
                self.construct_differs_from_catalytic_sequence,
        }


def _hash_sequence(seq: str) -> str:
    """Hash a sequence through :mod:`eagent.provenance`, or refuse.

    Never re-implements the normalisation. A locally computed hash with
    different whitespace or case handling would fail to match identical
    proteins, producing duplicate entities that look like independent evidence.
    """
    if _sequence_hash is None:
        raise SequenceHashUnavailableError(
            "eagent.provenance.sequence_hash is unavailable, so a sequence hash "
            "cannot be computed here; supply sequence_sha256 explicitly rather "
            "than hashing with different normalisation")
    return _sequence_hash(str(seq))


class ProteinRelation(str, enum.Enum):
    """The only three answers :func:`same_protein` is allowed to give."""

    IDENTICAL_SEQUENCE = "identical_sequence"
    RELATED_NOT_IDENTICAL = "related_not_identical"
    UNDETERMINED = "undetermined"

    @property
    def is_identity(self) -> bool:
        """Whether downstream code may treat the two records as one protein."""
        return self is ProteinRelation.IDENTICAL_SEQUENCE


@dataclass(frozen=True)
class ProteinComparison:
    """The verdict of :func:`same_protein`, with the reason attached.

    ``RELATED_NOT_IDENTICAL`` deliberately covers everything from a point mutant
    to an unrelated protein sharing a name. The function's job is to say "not
    the same", not to quantify how close two proteins are; quantifying closeness
    is a ranking question, and ranking must never feed a merge.
    """

    relation: ProteinRelation
    reason: str
    hash_a: str | None = None
    hash_b: str | None = None
    shared_accessions: tuple[str, ...] = ()
    construct_note: str = ""
    question: str = ""
    needs_curation: bool = False

    def __bool__(self) -> bool:
        return self.relation.is_identity

    @property
    def same(self) -> bool:
        """Whether the two records refer to the same amino-acid sequence."""
        return self.relation.is_identity

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the manifest."""
        return {
            "relation": self.relation.value, "reason": self.reason,
            "hash_a": self.hash_a, "hash_b": self.hash_b,
            "shared_accessions": list(self.shared_accessions),
            "construct_note": self.construct_note,
            "question": self.question, "needs_curation": self.needs_curation,
        }


def same_protein(a: ProteinIdentity, b: ProteinIdentity) -> ProteinComparison:
    """Answer identity on sequence-hash equality, and on nothing else.

    Returns :attr:`ProteinRelation.IDENTICAL_SEQUENCE` only when both records
    carry a sequence hash and the hashes are equal. Everything else is
    ``RELATED_NOT_IDENTICAL`` (the hashes differ) or ``UNDETERMINED`` (a hash is
    missing).

    The case this exists for: two records carrying the same accession and two
    different sequences. That happens whenever an entry has been re-annotated, a
    paper reports an isoform, or a variant was deposited under the wild-type
    accession. Reading the shared accession as an identity merges a wild type
    with a mutant and then attributes the mutant's activity to both.
    """
    shared = a.shares_accession_with(b)
    ha, hb = a.sequence_sha256, b.sequence_sha256

    if ha and hb:
        if ha == hb:
            note = ""
            ca, cb = a.construct_sha256, b.construct_sha256
            if ca and cb and ca != cb:
                note = ("the catalytic sequences are identical but the expressed "
                        "constructs differ (tags, truncations or fusions), so "
                        "measurements may still not be comparable")
            return ProteinComparison(
                relation=ProteinRelation.IDENTICAL_SEQUENCE,
                reason="the sequence hashes are equal",
                hash_a=ha, hash_b=hb, shared_accessions=shared,
                construct_note=note)
        reason = "the sequence hashes differ, so these are not the same protein"
        question = ""
        needs_curation = False
        if shared:
            reason += (f"; they nevertheless share the accession(s) "
                       f"{', '.join(shared)} -- an accession is a pointer into a "
                       f"database release, not an identity, and one accession "
                       f"routinely covers a re-annotated entry, an isoform or a "
                       f"variant deposited under the wild-type id")
            question = (f"Which sequence does {', '.join(shared)} denote in the "
                        f"release each record was read from? One of these two "
                        f"records is mis-keyed.")
            needs_curation = True
        return ProteinComparison(
            relation=ProteinRelation.RELATED_NOT_IDENTICAL, reason=reason,
            hash_a=ha, hash_b=hb, shared_accessions=shared,
            question=question, needs_curation=needs_curation)

    missing = [name for name, h in (("a", ha), ("b", hb)) if not h]
    reason = (f"record(s) {', '.join(missing)} carry no sequence hash; identity "
              f"is decided on the sequence hash alone, so this cannot be "
              f"answered")
    if shared:
        reason += (f". The shared accession(s) {', '.join(shared)} are not an "
                   f"answer: they do not fix a sequence")
    return ProteinComparison(
        relation=ProteinRelation.UNDETERMINED, reason=reason,
        hash_a=ha, hash_b=hb, shared_accessions=shared,
        question="Supply the amino-acid sequence for each record so the hashes "
                 "can be compared.",
        needs_curation=True)


# ---------------------------------------------------------------------------
# Merge policy
# ---------------------------------------------------------------------------

class MergeGround(str, enum.Enum):
    """Reasons someone might give for merging two records, and their verdict.

    Every member carries ``admissible`` and the reason. The inadmissible ones
    are enumerated rather than simply absent, because they are the grounds
    people actually use, and a policy that silently lacks them cannot explain a
    refusal.
    """

    def __new__(cls, value: str, admissible: bool, basis: str,
                reason: str) -> "MergeGround":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.admissible = admissible             # type: ignore[attr-defined]
        obj.basis = basis                       # type: ignore[attr-defined]
        obj.reason = reason                     # type: ignore[attr-defined]
        obj.__doc__ = reason
        return obj

    SEQUENCE_SHA256_EQUAL = (
        "sequence_sha256_equal", True, "identifier",
        "both records carry the same sequence hash, which is the primary key "
        "for a protein entity")
    CONSTRUCT_SHA256_EQUAL = (
        "construct_sha256_equal", True, "identifier",
        "both records carry the same expressed-construct hash, so the measured "
        "material was the same")
    ACCESSION_WITH_VERSION_EQUAL = (
        "accession_with_version_equal", True, "identifier",
        "both records cite the same accession in the same database release, so "
        "they point at one entry as it stood at one time")
    INCHIKEY_EQUAL = (
        "inchikey_equal", True, "identifier",
        "both records carry the same full InChIKey, which fixes connectivity, "
        "stereochemistry and protonation layer")
    CCD_COMPONENT_EQUAL = (
        "ccd_component_equal", True, "identifier",
        "both records name the same chemical component dictionary entry, which "
        "fixes the ligand including its oxidation state")
    NAME_SIMILARITY = (
        "name_similarity", False, "resemblance",
        "names resemble each other. Enzyme names are not identifiers: 'ADH', "
        "'alcohol dehydrogenase' and 'ADH-A' name hundreds of different "
        "proteins, and two records sharing a name routinely describe different "
        "sequences from different organisms")
    STRUCTURAL_RESEMBLANCE = (
        "structural_resemblance", False, "resemblance",
        "the structures look alike (low RMSD, same fold, high TM-score). Fold "
        "is conserved far beyond function; two Rossmann-fold proteins with "
        "different substrate specificity superpose beautifully and are not one "
        "entity")
    SUBSTRATE_NAME_SIMILARITY = (
        "substrate_name_similarity", False, "resemblance",
        "the substrate names nearly match. '4-chloroacetophenone' and "
        "'2-chloroacetophenone' differ by one character and are different "
        "compounds, and '(R)-' versus '(S)-' is the entire objective of an "
        "asymmetric reduction")
    SEQUENCE_IDENTITY_THRESHOLD = (
        "sequence_identity_threshold", False, "resemblance",
        "percent sequence identity exceeds a threshold. Identity ranks "
        "candidates; it does not make two sequences one entity, and a single "
        "active-site substitution between 99%-identical proteins can abolish "
        "activity")
    EMBEDDING_PROXIMITY = (
        "embedding_proximity", False, "resemblance",
        "embeddings or fingerprints are close. A distance in a learned space "
        "is a ranking signal with no identity semantics at all")
    SAME_EC_AND_ORGANISM = (
        "same_ec_and_organism", False, "annotation",
        "the records share an EC number and an organism. One organism commonly "
        "encodes several isoenzymes under one EC class, and EC numbers are "
        "assigned by annotation transfer")
    ACCESSION_WITHOUT_VERSION = (
        "accession_without_version", False, "identifier_incomplete",
        "the accessions match but no database release is recorded. Entries are "
        "merged, demerged and re-annotated, so an accession without its release "
        "does not pin a sequence")
    SAME_PDB_ENTRY = (
        "same_pdb_entry", False, "identifier_incomplete",
        "both records cite the same PDB entry. One entry contains several "
        "chains, often several different proteins, and frequently a construct "
        "that differs from the deposited sequence")
    OPERATOR_DECISION = (
        "operator_decision", True, "human",
        "a named human curator recorded a decision with a reason; the merge is "
        "attributable to a person rather than to an algorithm")

    @property
    def doc(self) -> str:
        """The reason text, exposed for reports."""
        return self.reason


@dataclass(frozen=True)
class RejectedGround:
    """One ground that was considered and refused, with why it was refused."""

    ground: MergeGround
    detail: str

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the decision record."""
        return {"ground": self.ground.value, "reason": self.ground.reason,
                "detail": self.detail}


@dataclass(frozen=True)
class MergeDecision:
    """Whether two records may be merged, and the reason either way.

    The rejected grounds are carried even on a positive decision, so a reviewer
    can see that the merge rests on the hash and not on the fact that the names
    also happened to match.
    """

    may_merge: bool
    ground: MergeGround | None
    reason: str
    rejected: tuple[RejectedGround, ...] = ()
    question: str = ""
    needs_curation: bool = False

    def __bool__(self) -> bool:
        return self.may_merge

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the manifest."""
        return {
            "may_merge": self.may_merge,
            "ground": self.ground.value if self.ground else None,
            "reason": self.reason,
            "rejected": [r.as_dict() for r in self.rejected],
            "question": self.question,
            "needs_curation": self.needs_curation,
        }

    def report_lines(self) -> list[str]:
        """Human-readable rendering of the decision."""
        head = "MERGE ALLOWED" if self.may_merge else "MERGE REFUSED"
        lines = [f"{head}: {self.reason}"]
        for r in self.rejected:
            lines.append(f"  rejected ground '{r.ground.value}': {r.ground.reason}")
            if r.detail:
                lines.append(f"    observed: {r.detail}")
        if self.question:
            lines.append(f"  curator question: {self.question}")
        return lines


class EntityMergePolicy:
    """A documented, testable answer to "may these two records be merged?".

    The policy admits identifier-based grounds only. Name similarity, structural
    resemblance and substrate-name similarity are each enumerated as
    inadmissible, and asking for them returns a refusal carrying the reason
    rather than a bare ``False`` -- a reviewer needs to know *why* the merge was
    refused, and a developer needs the refusal to be greppable.

    Resemblance is not banned from the system; it is banned from *merging*.
    Similarity may rank candidates, choose templates and order a shortlist. It
    may never collapse two rows into one, because once collapsed the error is in
    the primary key and nothing downstream can recover it.
    """

    #: Grounds the policy will accept, in the order it tries them.
    admissible_order: tuple[MergeGround, ...] = (
        MergeGround.SEQUENCE_SHA256_EQUAL,
        MergeGround.CONSTRUCT_SHA256_EQUAL,
        MergeGround.INCHIKEY_EQUAL,
        MergeGround.CCD_COMPONENT_EQUAL,
        MergeGround.ACCESSION_WITH_VERSION_EQUAL,
    )

    def __init__(self, *, allow_operator_override: bool = True) -> None:
        self.allow_operator_override = allow_operator_override

    # -- policy introspection ---------------------------------------------
    def admissible_grounds(self) -> tuple[MergeGround, ...]:
        """Grounds this policy accepts, for documentation and tests."""
        grounds = list(self.admissible_order)
        if self.allow_operator_override:
            grounds.append(MergeGround.OPERATOR_DECISION)
        return tuple(grounds)

    def inadmissible_grounds(self) -> tuple[MergeGround, ...]:
        """Grounds this policy refuses, each with its reason attached."""
        return tuple(g for g in MergeGround if not g.admissible)

    def explain(self, ground: MergeGround | str) -> str:
        """The policy's reason for accepting or refusing a named ground."""
        g = MergeGround(ground)
        verdict = "admissible" if self._is_admissible(g) else "inadmissible"
        return f"{g.value} is {verdict}: {g.reason}"

    # -- decisions ---------------------------------------------------------
    def may_merge(self, a: Any, b: Any, ground: MergeGround | str) \
            -> MergeDecision:
        """Decide a merge proposed on one named ground.

        An inadmissible ground is refused without even looking at the records:
        the ground is wrong regardless of how strongly it holds, and evaluating
        it would invite "but the similarity was 0.99".
        """
        g = MergeGround(ground)
        if not self._is_admissible(g):
            detail = self._observed_detail(a, b, g)
            return MergeDecision(
                may_merge=False, ground=None,
                reason=(f"'{g.value}' is not grounds for merging two records: "
                        f"{g.reason}"),
                rejected=(RejectedGround(g, detail),),
                question=("Which shared identifier, if any, do these two "
                          "records carry? Resemblance may rank them; only an "
                          "identifier may merge them."),
                needs_curation=True)
        if g is MergeGround.OPERATOR_DECISION:
            return MergeDecision(
                may_merge=True, ground=g,
                reason=("a named curator recorded the merge; the decision is "
                        "attributable to a person"),
                needs_curation=True)
        holds, detail = self._ground_holds(a, b, g)
        if holds is True:
            return MergeDecision(may_merge=True, ground=g,
                                 reason=f"{g.reason} ({detail})")
        if holds is False:
            return MergeDecision(
                may_merge=False, ground=None,
                reason=f"'{g.value}' does not hold: {detail}",
                question="Do these records share any admissible identifier?")
        return MergeDecision(
            may_merge=False, ground=None,
            reason=(f"'{g.value}' could not be evaluated: {detail}. A missing "
                    f"identifier is a data gap, not a licence to merge"),
            question=f"Supply the fields needed for {g.value} on both records.",
            needs_curation=True)

    def evaluate(self, a: Any, b: Any,
                 proposed_ground: MergeGround | str | None = None) \
            -> MergeDecision:
        """Decide a merge, trying every admissible ground when none is named.

        When no admissible ground holds, the decision also lists the
        resemblance-based grounds that *would* have fired on these two records.
        That list is the useful part of the refusal: it names the trap the
        caller was about to walk into.
        """
        if proposed_ground is not None:
            return self.may_merge(a, b, proposed_ground)

        unevaluated: list[RejectedGround] = []
        for g in self.admissible_order:
            holds, detail = self._ground_holds(a, b, g)
            if holds is True:
                return MergeDecision(
                    may_merge=True, ground=g, reason=f"{g.reason} ({detail})",
                    rejected=tuple(self._tempting_grounds(a, b)))
            if holds is None:
                unevaluated.append(RejectedGround(g, detail))

        tempting = self._tempting_grounds(a, b)
        reason = ("no admissible identifier links these records, so they stay "
                  "separate")
        if tempting:
            reason += (". They do resemble each other, which is not grounds: "
                       + "; ".join(t.ground.value for t in tempting))
        return MergeDecision(
            may_merge=False, ground=None, reason=reason,
            rejected=tuple(tempting + unevaluated),
            question=("Which shared identifier would justify merging these "
                      "records? If none exists, keep them separate and let the "
                      "lineage layer decide whether they are independent."),
            needs_curation=bool(unevaluated))

    # -- internals ---------------------------------------------------------
    def _is_admissible(self, ground: MergeGround) -> bool:
        if ground is MergeGround.OPERATOR_DECISION:
            return self.allow_operator_override
        return bool(ground.admissible)

    def _ground_holds(self, a: Any, b: Any, ground: MergeGround) \
            -> tuple[bool | None, str]:
        """Whether an admissible ground holds: True / False / None (unknown)."""
        fields = {
            MergeGround.SEQUENCE_SHA256_EQUAL: ("sequence_sha256",
                                                "sequence_hash"),
            MergeGround.CONSTRUCT_SHA256_EQUAL: ("construct_sha256",),
            MergeGround.INCHIKEY_EQUAL: ("inchikey", "inchi_key"),
            MergeGround.CCD_COMPONENT_EQUAL: ("ccd_component_id", "ligand_code",
                                              "comp_id"),
        }.get(ground)
        if ground is MergeGround.ACCESSION_WITH_VERSION_EQUAL:
            acc_a, ver_a = _read(a, "accession"), _read(a, "database_version",
                                                        "accession_version")
            acc_b, ver_b = _read(b, "accession"), _read(b, "database_version",
                                                        "accession_version")
            if not acc_a or not acc_b:
                return None, "one record carries no accession"
            if not ver_a or not ver_b:
                return None, ("the accessions are present but at least one "
                              "lacks a database release")
            same = (str(acc_a).strip().upper() == str(acc_b).strip().upper()
                    and str(ver_a).strip() == str(ver_b).strip())
            return (True, f"{acc_a}@{ver_a}") if same else (
                False, f"{acc_a}@{ver_a} vs {acc_b}@{ver_b}")
        if fields is None:  # pragma: no cover - defensive
            return None, f"no reader is defined for {ground.value}"
        va, vb = _read(a, *fields), _read(b, *fields)
        if not va or not vb:
            return None, f"{fields[0]} is absent on at least one record"
        same = str(va).strip().upper() == str(vb).strip().upper()
        return (True, f"{fields[0]}={va}") if same else (
            False, f"{va} vs {vb}")

    def _tempting_grounds(self, a: Any, b: Any) -> list[RejectedGround]:
        """Resemblance-based grounds that would have fired on these records."""
        out: list[RejectedGround] = []
        na, nb = _read(a, "name", "label", "protein_name"), \
            _read(b, "name", "label", "protein_name")
        if na and nb and _names_resemble(str(na), str(nb)):
            out.append(RejectedGround(
                MergeGround.NAME_SIMILARITY, f"{na!r} vs {nb!r}"))
        sa, sb = _read(a, "substrate_name"), _read(b, "substrate_name")
        if sa and sb and _names_resemble(str(sa), str(sb)):
            out.append(RejectedGround(
                MergeGround.SUBSTRATE_NAME_SIMILARITY, f"{sa!r} vs {sb!r}"))
        acc_a, acc_b = _read(a, "accession"), _read(b, "accession")
        if acc_a and acc_b and str(acc_a).strip().upper() == str(acc_b).strip().upper():
            if not (_read(a, "database_version", "accession_version")
                    and _read(b, "database_version", "accession_version")):
                out.append(RejectedGround(
                    MergeGround.ACCESSION_WITHOUT_VERSION,
                    f"both cite {acc_a} with no release recorded"))
        pid_a, pid_b = _read(a, "percent_identity"), _read(b, "percent_identity")
        if pid_a is not None and pid_b is not None:
            out.append(RejectedGround(
                MergeGround.SEQUENCE_IDENTITY_THRESHOLD,
                f"percent identity recorded on both records "
                f"({pid_a}, {pid_b})"))
        pdb_a, pdb_b = _read(a, "pdb_id"), _read(b, "pdb_id")
        if pdb_a and pdb_b and str(pdb_a).strip().upper() == str(pdb_b).strip().upper():
            out.append(RejectedGround(MergeGround.SAME_PDB_ENTRY,
                                      f"both cite PDB {pdb_a}"))
        return out

    def _observed_detail(self, a: Any, b: Any, ground: MergeGround) -> str:
        """A short note on what the records actually carry, for the refusal."""
        if ground is MergeGround.NAME_SIMILARITY:
            return f"{_read(a, 'name', 'label')!r} vs {_read(b, 'name', 'label')!r}"
        if ground is MergeGround.SUBSTRATE_NAME_SIMILARITY:
            return (f"{_read(a, 'substrate_name')!r} vs "
                    f"{_read(b, 'substrate_name')!r}")
        if ground is MergeGround.STRUCTURAL_RESEMBLANCE:
            return "a structural similarity score is not read by this policy"
        return ""


def _read(record: Any, *names: str) -> Any:
    """Read the first present field from a mapping or an object."""
    for name in names:
        if isinstance(record, Mapping):
            if name in record and record[name] not in (None, ""):
                return record[name]
            continue
        value = getattr(record, name, None)
        if value not in (None, ""):
            return value
    return None


def _names_resemble(a: str, b: str) -> bool:
    """Advisory-only check that two names look alike. Never a merge ground.

    Deliberately crude -- case-folded equality or a shared token. It exists only
    so a refusal can say "yes, the names match, and that is still not grounds".
    Making it cleverer would invite someone to promote it to a decision rule.
    """
    na, nb = normalise_query_name(a), normalise_query_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta = {t for t in re.split(r"[^a-z0-9]+", na) if len(t) > 2}
    tb = {t for t in re.split(r"[^a-z0-9]+", nb) if len(t) > 2}
    return bool(ta & tb)
