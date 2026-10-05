"""The six-layer data architecture, and the only joins allowed between layers.

Why this module exists
----------------------
Enzyme projects fail in a characteristic way: a name-level match is treated as
an identity. "ADH from *Lactobacillus*" in a kinetics table, "alcohol
dehydrogenase" in a structure paper, and a 92%-identical hit from a metagenome
are written into one row, and the resulting row describes no real catalytic
system. Nothing downstream can recover from that, because the error is in the
primary key.

So this module makes two things explicit and checkable:

1. **The layers answer different questions.** A sequence-family record cannot
   stand in for an activity measurement, and a predicted complex cannot stand in
   for either. :class:`DataLayer` names the six questions and the agent stage
   each one serves, and :class:`LayerCoverage` counts what was actually found per
   layer so the system can say "we are thin here" instead of implying uniform
   support.

2. **Records are joined by identifiers, never by resemblance.**
   :class:`JoinKey` enumerates the permitted links; :func:`validate_join`
   performs one; :func:`refuse_similarity_join` is the named, explicit refusal
   path for every similarity-, embedding-, identity- or name-based "join". Two
   enzymes with similar names, two structures that look alike and two substrate
   strings that nearly match are not the same validated catalytic system, and no
   amount of cosine distance makes them one.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:  # the harness error base; guarded so this module imports standalone
    from ..errors import EAgentError
except Exception:  # pragma: no cover - only when eagent.errors is unavailable
    class EAgentError(Exception):  # type: ignore[no-redef]
        """Fallback base error used when :mod:`eagent.errors` cannot be imported."""

try:  # canonical sequence hash; never re-implemented locally (see below)
    from ..provenance import sequence_hash as _sequence_hash
except Exception:  # pragma: no cover - only when eagent.provenance is unavailable
    _sequence_hash = None  # type: ignore[assignment]


__all__ = [
    "DataLayer",
    "LAYER_ORDER",
    "JoinKey",
    "JoinKeySpec",
    "JoinResult",
    "JoinError",
    "JoinFieldMissingError",
    "UnknownJoinKeyError",
    "SimilarityJoinRefusedError",
    "SIMILARITY_PSEUDO_KEY_STEMS",
    "is_similarity_pseudo_key",
    "refuse_similarity_join",
    "validate_join",
    "permitted_keys",
    "join_key_spec",
    "LayerCoverage",
]


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class DataLayer(str, enum.Enum):
    """The six questions the data has to answer, kept apart on purpose.

    Each member carries the question it answers, the agent stage it serves, and
    a statement of what it cannot substitute for. The split exists because a
    single "enzyme database" table silently mixes a nomenclature assignment, a
    kinetic measurement and a predicted structure, and a downstream ranker then
    treats all three as the same kind of support.
    """

    def __new__(cls, value: str, question: str, agent_stage: str,
                doc: str, cannot_substitute: str) -> "DataLayer":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.question = question                     # type: ignore[attr-defined]
        obj.agent_stage = agent_stage               # type: ignore[attr-defined]
        obj.cannot_substitute = cannot_substitute   # type: ignore[attr-defined]
        obj.__doc__ = doc
        return obj

    REACTION_AND_CHEMISTRY = (
        "reaction_and_chemistry",
        "What is the substrate, which bond changes, and what is the product "
        "configuration?",
        "task normalisation, reaction search",
        "Defines the transformation itself: substrate and product structures, the "
        "bond that changes, reaction direction and stereochemical outcome. Serves "
        "task normalisation and reaction search. Prevents the most expensive error "
        "in the pipeline, which is searching for the wrong reaction very "
        "efficiently.",
        "A reaction record says nothing about whether any particular protein "
        "catalyses it.",
    )
    ENZYMOLOGY_EVIDENCE = (
        "enzymology_evidence",
        "Which enzymes actually catalysed this or a near substrate, and under "
        "what conditions?",
        "seed selection, activity evidence",
        "Holds measured activity: enzyme, substrate, conditions, outcome and "
        "detection. Serves seed selection and the activity-evidence trail behind "
        "every candidate. Prevents a family annotation being reported as a "
        "measurement.",
        "An activity record for one substrate is not evidence for another "
        "substrate, another cofactor or another direction.",
    )
    SEQUENCE_FAMILY_EVOLUTION = (
        "sequence_family_evolution",
        "Which sequences are worth expanding, and which belong to the same "
        "mechanism family?",
        "mining, clustering, diversity selection",
        "Holds sequences, clusters, families, domains and orthology. Serves "
        "mining, clustering and diversity selection. Prevents a candidate set "
        "that is one protein sampled ninety times.",
        "Family membership is not activity on the target substrate, and cluster "
        "co-membership is not equivalence.",
    )
    STRUCTURE_AND_MECHANISM = (
        "structure_and_mechanism",
        "Can the substrate, the cofactor and the catalytic residues form a "
        "sensible arrangement?",
        "complex modelling, geometry",
        "Holds experimental and predicted structures, residue mappings, ligand "
        "chemistry and catalytic-residue assignments. Serves complex modelling "
        "and geometric screening. Prevents a candidate being advanced on sequence "
        "similarity while its active site cannot accommodate the substrate.",
        "A plausible geometry is a hypothesis, not a turnover number; a predicted "
        "complex is not an observed one.",
    )
    MUTATION_AND_PERFORMANCE = (
        "mutation_and_performance",
        "Which positions should change, and what might that change cost?",
        "local design, directed evolution",
        "Holds variant-effect, stability and engineering-campaign data. Serves "
        "local design and directed-evolution planning. Prevents a redesign that "
        "buys selectivity by destroying expression.",
        "A stability or binding measurement is not a measurement of activity on "
        "the target substrate.",
    )
    LITERATURE_AND_FEEDBACK = (
        "literature_and_feedback",
        "How do we obtain new evidence, and how does this project's own data "
        "re-enter the next decision?",
        "literature agent, active learning",
        "Holds literature retrieval, extraction corpora, author archives, strain "
        "and thermodynamic context, and the project's own experimental output. "
        "Serves the literature agent and the active-learning loop. Prevents a "
        "closed system that never learns from its own failed batch.",
        "A machine-extracted relation is not a curated record, and a favourable "
        "thermodynamic result is not catalysis.",
    )

    @property
    def doc(self) -> str:
        """The member docstring, exposed for CLI and report rendering."""
        return self.__doc__ or ""

    def describe(self) -> str:
        """One-paragraph rendering used in coverage reports and the CLI."""
        return (f"{self.value}: {self.question} "
                f"[stage: {self.agent_stage}] "
                f"Not a substitute: {self.cannot_substitute}")


#: Display order for reports. Reports always list all six, including empties.
LAYER_ORDER: tuple[DataLayer, ...] = (
    DataLayer.REACTION_AND_CHEMISTRY,
    DataLayer.ENZYMOLOGY_EVIDENCE,
    DataLayer.SEQUENCE_FAMILY_EVOLUTION,
    DataLayer.STRUCTURE_AND_MECHANISM,
    DataLayer.MUTATION_AND_PERFORMANCE,
    DataLayer.LITERATURE_AND_FEEDBACK,
)


# ---------------------------------------------------------------------------
# Join errors
# ---------------------------------------------------------------------------

class JoinError(EAgentError):
    """Base class for every refusal to link two records.

    A join that cannot be made must fail loudly here rather than degrade into a
    fuzzy match somewhere downstream.
    """


class JoinFieldMissingError(JoinError):
    """A record does not carry the identifier the requested join needs.

    Raised instead of returning "not joined", because a missing identifier and a
    genuine mismatch call for different actions: the first needs the field
    resolved, the second is an answer.
    """

    def __init__(self, key: "JoinKey | str", side: str, fields: Sequence[str]):
        self.key = key
        self.side = side
        self.fields = list(fields)
        super().__init__(
            f"join on '{getattr(key, 'value', key)}' needs "
            f"{', '.join(self.fields)} on the {side} record; it is absent or empty"
        )


class UnknownJoinKeyError(JoinError):
    """A join key that is not in :class:`JoinKey` was requested.

    The permitted joins are a closed set on purpose; an ad-hoc key is how an
    unvalidated link enters the graph.
    """


class SimilarityJoinRefusedError(JoinError):
    """A similarity-, identity-, embedding- or name-based "join" was requested.

    This is the refusal that gives the layer architecture its value. Vector
    similarity, sequence identity, name string distance, structural RMSD and
    nearest-neighbour lookup all answer "these resemble each other". None of them
    answers "these are the same validated catalytic system", which is the only
    question a join may answer. Resemblance may *rank* candidates; it may never
    *merge* records.
    """

    def __init__(self, requested: str, detail: str = ""):
        self.requested = requested
        msg = (f"refusing to join records on '{requested}': resemblance is not "
               f"identity. Records are linked only by the identifiers in "
               f"JoinKey; two enzymes with similar names, two structures that "
               f"look alike and two nearly matching substrate strings are not "
               f"the same validated catalytic system.")
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


#: Substrings that mark a proposed key as resemblance-based rather than
#: identity-based. Matched case-insensitively against the requested key name.
SIMILARITY_PSEUDO_KEY_STEMS: tuple[str, ...] = (
    "similar", "similarity", "identity", "embed", "cosine", "fuzzy", "tanimoto",
    "rmsd", "tmscore", "tm_score", "blast", "diamond", "mmseqs", "levenshtein",
    "distance", "nearest", "knn", "vector", "approximate", "looks_like",
    "name_match", "fuzzy_name", "alignment_score", "homolog", "cluster_member",
    "same_name", "substring",
)


def is_similarity_pseudo_key(name: str) -> bool:
    """Whether a proposed key name describes resemblance rather than identity.

    Exists so that the refusal is triggered by the *shape* of the request rather
    than by an exhaustive list of tool names; a new similarity tool invented
    tomorrow still trips it as long as it is honest about what it computes.
    """
    low = str(name).strip().lower()
    return any(stem in low for stem in SIMILARITY_PSEUDO_KEY_STEMS)


def refuse_similarity_join(requested: str, detail: str = "") -> None:
    """Always raise :class:`SimilarityJoinRefusedError`.

    The explicit refusal path. Call it from any connector that is tempted to
    merge records on a score, so the refusal appears in one place with one
    message rather than as scattered silent skips.
    """
    raise SimilarityJoinRefusedError(requested, detail)


# ---------------------------------------------------------------------------
# Join keys
# ---------------------------------------------------------------------------

class JoinKey(str, enum.Enum):
    """The closed set of permitted links between records, across or within layers.

    Each key is an *identifier* that two records either carry or do not. A key is
    deliberately composite where a bare identifier is ambiguous: an accession
    without its database version, a PDB entry without a chain and a SIFTS residue
    mapping, a ligand component without an atom name, or a DOI without the
    identifier of the measurement campaign inside the paper all name more than
    one thing.
    """

    SEQUENCE_SHA256 = "sequence_sha256"
    ACCESSION_WITH_DB_VERSION = "accession_with_database_version"
    RHEA_REACTION_ID = "rhea_reaction_id"
    CHEBI_ID = "chebi_id"
    PUBCHEM_CID = "pubchem_cid"
    INCHIKEY = "inchikey"
    PDB_CHAIN_SIFTS_RESIDUE = "pdb_chain_sifts_residue"
    CCD_COMPONENT_ATOM = "ccd_component_atom"
    DOI_WITH_EXPERIMENT_ACTIVITY_ID = "doi_with_experiment_activity_id"

    @property
    def spec(self) -> "JoinKeySpec":
        """The field list, semantics and caveats attached to this key."""
        return _JOIN_SPECS[self]


@dataclass(frozen=True)
class JoinKeySpec:
    """What a join key reads, what it establishes, and what it does not.

    The ``does_not_establish`` text exists because every one of these keys is
    routinely over-read: an InChIKey match is taken as "same assay substrate", a
    PDB match as "same construct", a DOI match as "same experiment". The caveat
    travels with the join result so a reviewer sees it at the point of use.
    """

    key: JoinKey
    fields: tuple[str, ...]
    aliases: Mapping[str, tuple[str, ...]]
    case_sensitive_fields: frozenset[str]
    connects: frozenset[DataLayer]
    establishes: str
    does_not_establish: str


_JOIN_SPECS: dict[JoinKey, JoinKeySpec] = {
    JoinKey.SEQUENCE_SHA256: JoinKeySpec(
        key=JoinKey.SEQUENCE_SHA256,
        fields=("sequence_sha256",),
        aliases={"sequence_sha256": ("seq_sha256", "sequence_hash")},
        case_sensitive_fields=frozenset(),
        connects=frozenset({
            DataLayer.ENZYMOLOGY_EVIDENCE, DataLayer.SEQUENCE_FAMILY_EVOLUTION,
            DataLayer.STRUCTURE_AND_MECHANISM, DataLayer.MUTATION_AND_PERFORMANCE,
            DataLayer.LITERATURE_AND_FEEDBACK,
        }),
        establishes="the two records refer to the same amino-acid sequence",
        does_not_establish=(
            "the same construct: tags, truncations and fusions change the "
            "expressed protein without changing the catalytic domain sequence"),
    ),
    JoinKey.ACCESSION_WITH_DB_VERSION: JoinKeySpec(
        key=JoinKey.ACCESSION_WITH_DB_VERSION,
        fields=("accession", "database_version"),
        aliases={"accession": ("uniprot_accession", "primary_accession"),
                 "database_version": ("db_version", "release")},
        case_sensitive_fields=frozenset({"database_version"}),
        connects=frozenset({
            DataLayer.ENZYMOLOGY_EVIDENCE, DataLayer.SEQUENCE_FAMILY_EVOLUTION,
            DataLayer.STRUCTURE_AND_MECHANISM, DataLayer.MUTATION_AND_PERFORMANCE,
            DataLayer.LITERATURE_AND_FEEDBACK,
        }),
        establishes="the two records cite the same database entry in the same release",
        does_not_establish=(
            "the same sequence across releases: entries are merged, demerged and "
            "re-annotated, so an accession without its release is not an identity"),
    ),
    JoinKey.RHEA_REACTION_ID: JoinKeySpec(
        key=JoinKey.RHEA_REACTION_ID,
        fields=("rhea_id",),
        aliases={"rhea_id": ("reaction_id", "rhea")},
        case_sensitive_fields=frozenset(),
        connects=frozenset({
            DataLayer.REACTION_AND_CHEMISTRY, DataLayer.ENZYMOLOGY_EVIDENCE,
            DataLayer.SEQUENCE_FAMILY_EVOLUTION, DataLayer.LITERATURE_AND_FEEDBACK,
        }),
        establishes="the two records cite the same Rhea reaction entry",
        does_not_establish=(
            "the same measured direction: a Rhea identifier has a reference "
            "direction and a bidirectional parent, and an oxidation-direction "
            "record is not evidence of reduction activity"),
    ),
    JoinKey.CHEBI_ID: JoinKeySpec(
        key=JoinKey.CHEBI_ID,
        fields=("chebi_id",),
        aliases={"chebi_id": ("chebi",)},
        case_sensitive_fields=frozenset(),
        connects=frozenset({
            DataLayer.REACTION_AND_CHEMISTRY, DataLayer.ENZYMOLOGY_EVIDENCE,
            DataLayer.MUTATION_AND_PERFORMANCE, DataLayer.LITERATURE_AND_FEEDBACK,
        }),
        establishes="the two records cite the same ChEBI entry",
        does_not_establish=(
            "the same compound in the assay: a ChEBI class entry covers many "
            "specific compounds, and protonation state and stereochemistry differ "
            "between an entry and the material in the flask"),
    ),
    JoinKey.PUBCHEM_CID: JoinKeySpec(
        key=JoinKey.PUBCHEM_CID,
        fields=("pubchem_cid",),
        aliases={"pubchem_cid": ("cid",)},
        case_sensitive_fields=frozenset(),
        connects=frozenset({
            DataLayer.REACTION_AND_CHEMISTRY, DataLayer.ENZYMOLOGY_EVIDENCE,
            DataLayer.MUTATION_AND_PERFORMANCE,
        }),
        establishes="the two records cite the same PubChem compound identifier",
        does_not_establish=(
            "the same salt form, charge state or enantiomer: a CID resolved from "
            "a trivial name frequently differs from the assayed material in "
            "exactly those respects"),
    ),
    JoinKey.INCHIKEY: JoinKeySpec(
        key=JoinKey.INCHIKEY,
        fields=("inchikey",),
        aliases={"inchikey": ("inchi_key", "standard_inchikey")},
        case_sensitive_fields=frozenset(),
        connects=frozenset({
            DataLayer.REACTION_AND_CHEMISTRY, DataLayer.ENZYMOLOGY_EVIDENCE,
            DataLayer.STRUCTURE_AND_MECHANISM, DataLayer.MUTATION_AND_PERFORMANCE,
            DataLayer.LITERATURE_AND_FEEDBACK,
        }),
        establishes="the two records describe the same connectivity, stereochemistry "
                    "and protonation layer as encoded by the full InChIKey",
        does_not_establish=(
            "equivalence under a skeleton-only comparison: comparing only the "
            "first block of an InChIKey discards stereochemistry, which is the "
            "whole objective of an asymmetric reduction"),
    ),
    JoinKey.PDB_CHAIN_SIFTS_RESIDUE: JoinKeySpec(
        key=JoinKey.PDB_CHAIN_SIFTS_RESIDUE,
        fields=("pdb_id", "chain_id", "sifts_residue"),
        aliases={"pdb_id": ("pdb", "entry_id"),
                 "chain_id": ("chain", "auth_asym_id"),
                 "sifts_residue": ("sifts_uniprot_residue", "uniprot_residue")},
        case_sensitive_fields=frozenset({"chain_id"}),
        connects=frozenset({
            DataLayer.STRUCTURE_AND_MECHANISM, DataLayer.SEQUENCE_FAMILY_EVOLUTION,
            DataLayer.MUTATION_AND_PERFORMANCE,
        }),
        establishes="the two records point at the same residue of the same chain "
                    "of the same entry, through an explicit SIFTS mapping",
        does_not_establish=(
            "agreement of residue numbering: author numbering, label numbering and "
            "UniProt numbering diverge, so a position compared without the SIFTS "
            "mapping is a different position"),
    ),
    JoinKey.CCD_COMPONENT_ATOM: JoinKeySpec(
        key=JoinKey.CCD_COMPONENT_ATOM,
        fields=("ccd_component_id", "atom_name"),
        aliases={"ccd_component_id": ("ligand_code", "comp_id", "chem_comp_id"),
                 "atom_name": ("atom", "atom_id")},
        case_sensitive_fields=frozenset({"atom_name"}),
        connects=frozenset({
            DataLayer.STRUCTURE_AND_MECHANISM, DataLayer.REACTION_AND_CHEMISTRY,
        }),
        establishes="the two records name the same atom of the same chemical "
                    "component dictionary entry",
        does_not_establish=(
            "the same cofactor oxidation state: NAD and NAI, or NAP and NDP, are "
            "different component ids for what a prose label calls the same cofactor"),
    ),
    JoinKey.DOI_WITH_EXPERIMENT_ACTIVITY_ID: JoinKeySpec(
        key=JoinKey.DOI_WITH_EXPERIMENT_ACTIVITY_ID,
        fields=("doi", "experiment_activity_id"),
        aliases={"doi": ("source_doi",),
                 "experiment_activity_id": ("activity_id", "campaign_id")},
        case_sensitive_fields=frozenset({"experiment_activity_id"}),
        connects=frozenset({
            DataLayer.LITERATURE_AND_FEEDBACK, DataLayer.ENZYMOLOGY_EVIDENCE,
            DataLayer.MUTATION_AND_PERFORMANCE,
        }),
        establishes="the two records come from the same measurement campaign in "
                    "the same publication, and are therefore not independent",
        does_not_establish=(
            "independent corroboration: this key exists mainly to *collapse* "
            "duplicates, because the same campaign re-published through four "
            "databases is one piece of evidence, not four"),
    ),
}


def join_key_spec(key: "JoinKey | str") -> JoinKeySpec:
    """Resolve a key name to its spec, refusing similarity pseudo-keys by name.

    Prevents an unvalidated link entering the graph through a string that merely
    looks like a key.
    """
    return _JOIN_SPECS[_coerce_key(key)]


def permitted_keys(a: DataLayer, b: DataLayer) -> tuple[JoinKey, ...]:
    """Keys that may legitimately link a record in layer ``a`` to one in ``b``.

    Exists so a planner can discover that two layers have no identifier in common
    and must stay unjoined, rather than reaching for a resemblance score to
    bridge the gap.
    """
    return tuple(k for k, s in _JOIN_SPECS.items()
                 if a in s.connects and b in s.connects)


@dataclass(frozen=True)
class JoinResult:
    """The outcome of one join attempt, with its caveat attached.

    ``partial`` marks the dangerous case: the primary identifier matched but a
    qualifier (database release, chain, residue, atom) did not. That is the exact
    situation in which a human reads "same accession" and merges anyway.
    """

    key: JoinKey
    joined: bool
    values_a: tuple[str, ...]
    values_b: tuple[str, ...]
    reason: str
    partial: bool = False
    establishes: str = ""
    does_not_establish: str = ""

    def __bool__(self) -> bool:
        return self.joined


def validate_join(a: Any, b: Any, key: "JoinKey | str") -> JoinResult:
    """Decide whether two records may be linked, using one permitted key.

    ``a`` and ``b`` may be mappings or objects; fields are read by key or by
    attribute, including the aliases declared on the key spec.

    Three distinct outcomes, kept distinct on purpose:

    * returns ``joined=True`` when every component of the key is present on both
      sides and equal after normalisation;
    * returns ``joined=False`` (``partial=True`` when only a qualifier differs)
      when the records genuinely disagree;
    * raises :class:`JoinFieldMissingError` when a record does not carry the
      identifier at all, because an absent key is a data gap to resolve, not a
      negative answer;
    * raises :class:`SimilarityJoinRefusedError` when the requested key is a
      resemblance measure.

    Collapsing the second and third outcomes into one boolean is how "we could
    not check" becomes "we checked and they differ".
    """
    jk = _coerce_key(key)
    spec = _JOIN_SPECS[jk]

    vals_a = _read_key(a, spec, "left")
    vals_b = _read_key(b, spec, "right")

    mismatched = [f for f, x, y in zip(spec.fields, vals_a, vals_b) if x != y]
    if not mismatched:
        return JoinResult(
            key=jk, joined=True, values_a=vals_a, values_b=vals_b,
            reason=f"all components of {jk.value} match: "
                   + ", ".join(f"{f}={v}" for f, v in zip(spec.fields, vals_a)),
            establishes=spec.establishes,
            does_not_establish=spec.does_not_establish,
        )

    primary = spec.fields[0]
    partial = primary not in mismatched and len(spec.fields) > 1
    if partial and jk is JoinKey.ACCESSION_WITH_DB_VERSION:
        reason = (f"accession {vals_a[0]} is the same but the database release "
                  f"differs ({vals_a[1]} vs {vals_b[1]}); entries are "
                  f"re-annotated, merged and demerged between releases, so this "
                  f"is not an identity without re-fetching both")
    elif partial:
        reason = (f"{primary} matches but "
                  f"{', '.join(mismatched)} differ; the qualifier is part of the "
                  f"identifier, so the records point at different things")
    else:
        reason = (f"{primary} differs ({vals_a[0]} vs {vals_b[0]}); "
                  f"no join on {jk.value}")

    return JoinResult(
        key=jk, joined=False, values_a=vals_a, values_b=vals_b, reason=reason,
        partial=partial, establishes="",
        does_not_establish=spec.does_not_establish,
    )


# -- internals --------------------------------------------------------------

def _coerce_key(key: "JoinKey | str") -> JoinKey:
    if isinstance(key, JoinKey):
        return key
    name = str(key).strip()
    if is_similarity_pseudo_key(name):
        refuse_similarity_join(name)
    try:
        return JoinKey(name.lower())
    except ValueError:
        pass
    try:
        return JoinKey[name.upper()]
    except KeyError as exc:
        raise UnknownJoinKeyError(
            f"'{name}' is not a permitted join key; permitted keys are "
            f"{', '.join(k.value for k in JoinKey)}"
        ) from exc


def _read_key(record: Any, spec: JoinKeySpec, side: str) -> tuple[str, ...]:
    out: list[str] = []
    for fname in spec.fields:
        raw = _get_field(record, fname)
        if _is_empty(raw):
            for alias in spec.aliases.get(fname, ()):
                raw = _get_field(record, alias)
                if not _is_empty(raw):
                    break
        if _is_empty(raw) and fname == "sequence_sha256":
            raw = _derive_sequence_hash(record, side)
        if _is_empty(raw):
            raise JoinFieldMissingError(spec.key, side, (fname,) + tuple(
                spec.aliases.get(fname, ())))
        out.append(_normalise(fname, str(raw), spec))
    return tuple(out)


def _derive_sequence_hash(record: Any, side: str) -> Any:
    """Use the canonical hash when a raw sequence is present; never re-implement it.

    If :mod:`eagent.provenance` is unavailable we refuse rather than compute a
    hash with possibly different normalisation, because a near-miss hash is worse
    than a missing one: it silently fails to join records that are identical.
    """
    seq = _get_field(record, "sequence")
    if _is_empty(seq):
        return None
    if _sequence_hash is None:
        raise JoinFieldMissingError(
            JoinKey.SEQUENCE_SHA256, side,
            ("sequence_sha256 (eagent.provenance.sequence_hash is unavailable, "
             "so it cannot be derived from 'sequence' here)",))
    return _sequence_hash(str(seq))


def _get_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
        return None
    return getattr(record, name, None)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


_ID_PREFIXES: dict[str, tuple[str, ...]] = {
    "chebi_id": ("CHEBI:",),
    "rhea_id": ("RHEA:",),
    "pubchem_cid": ("CID:", "CID"),
    "doi": ("HTTPS://DOI.ORG/", "HTTP://DOI.ORG/", "DOI:"),
}


def _normalise(field_name: str, value: str, spec: JoinKeySpec) -> str:
    v = value.strip()
    if field_name in spec.case_sensitive_fields:
        return v
    upper = v.upper()
    for prefix in _ID_PREFIXES.get(field_name, ()):
        if upper.startswith(prefix):
            upper = upper[len(prefix):].strip()
            break
    if field_name == "doi":
        return upper.lower()
    if field_name == "sequence_sha256":
        return upper.lower()
    return upper


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

@dataclass
class LayerCoverage:
    """How many records each layer actually supplied for one task.

    Exists so the system can say "the enzymology layer gave us four records and
    the mutation layer gave us none" instead of presenting a candidate table that
    implies all six layers contributed equally. A ranked list built on one thin
    layer looks exactly like a ranked list built on six rich ones, which is why
    thinness has to be reported as a first-class number rather than inferred.

    ``experimental`` is tracked separately because twelve annotation-only records
    and twelve measurements are not the same coverage.
    """

    task_id: str
    thin_threshold: int = 3
    counts: dict[DataLayer, int] = field(default_factory=dict)
    experimental_counts: dict[DataLayer, int] = field(default_factory=dict)
    by_source: dict[DataLayer, dict[str, int]] = field(default_factory=dict)
    gaps: dict[DataLayer, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for layer in LAYER_ORDER:
            self.counts.setdefault(layer, 0)
            self.experimental_counts.setdefault(layer, 0)
            self.by_source.setdefault(layer, {})
            self.gaps.setdefault(layer, [])

    # -- mutation ----------------------------------------------------------
    def add(self, layer: DataLayer, n: int = 1, source_id: str | None = None,
            experimental: bool = False) -> "LayerCoverage":
        """Record ``n`` records obtained for ``layer``, optionally per source."""
        if n < 0:
            raise ValueError("record counts cannot be negative")
        layer = DataLayer(layer)
        self.counts[layer] = self.counts.get(layer, 0) + n
        if experimental:
            self.experimental_counts[layer] = \
                self.experimental_counts.get(layer, 0) + n
        if source_id:
            bucket = self.by_source.setdefault(layer, {})
            bucket[source_id] = bucket.get(source_id, 0) + n
        return self

    def record_gap(self, layer: DataLayer, reason: str) -> "LayerCoverage":
        """Note why a layer is empty or thin, so the report explains itself."""
        self.gaps.setdefault(DataLayer(layer), []).append(reason)
        return self

    # -- queries -----------------------------------------------------------
    def count(self, layer: DataLayer) -> int:
        return self.counts.get(DataLayer(layer), 0)

    def experimental_count(self, layer: DataLayer) -> int:
        return self.experimental_counts.get(DataLayer(layer), 0)

    def sources(self, layer: DataLayer) -> list[str]:
        return sorted(self.by_source.get(DataLayer(layer), {}))

    def empty_layers(self) -> list[DataLayer]:
        """Layers that supplied nothing at all."""
        return [l for l in LAYER_ORDER if self.counts.get(l, 0) == 0]

    def thin_layers(self) -> list[DataLayer]:
        """Layers below ``thin_threshold``, empty ones included."""
        return [l for l in LAYER_ORDER
                if self.counts.get(l, 0) < self.thin_threshold]

    def total(self) -> int:
        return sum(self.counts.get(l, 0) for l in LAYER_ORDER)

    def is_single_source(self, layer: DataLayer) -> bool:
        """Whether a layer rests on exactly one registered source."""
        return len(self.by_source.get(DataLayer(layer), {})) == 1

    # -- reporting ---------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "thin_threshold": self.thin_threshold,
            "layers": {
                l.value: {
                    "records": self.counts.get(l, 0),
                    "experimental_records": self.experimental_counts.get(l, 0),
                    "sources": dict(sorted(self.by_source.get(l, {}).items())),
                    "gaps": list(self.gaps.get(l, [])),
                    "thin": self.counts.get(l, 0) < self.thin_threshold,
                }
                for l in LAYER_ORDER
            },
            "total_records": self.total(),
            "empty_layers": [l.value for l in self.empty_layers()],
            "thin_layers": [l.value for l in self.thin_layers()],
        }

    def report_lines(self) -> list[str]:
        """Human-readable coverage, always listing all six layers."""
        lines = [f"layer coverage for task {self.task_id} "
                 f"(thin below {self.thin_threshold} records)"]
        for l in LAYER_ORDER:
            n = self.counts.get(l, 0)
            e = self.experimental_counts.get(l, 0)
            srcs = ", ".join(self.sources(l)) or "-"
            mark = "EMPTY " if n == 0 else ("THIN  " if n < self.thin_threshold
                                            else "ok    ")
            lines.append(f"  {mark} {l.value:<26} {n:>4} records "
                         f"({e} experimental) from [{srcs}]")
            for g in self.gaps.get(l, []):
                lines.append(f"           gap: {g}")
        return lines

    def describe(self) -> str:
        return "\n".join(self.report_lines())
