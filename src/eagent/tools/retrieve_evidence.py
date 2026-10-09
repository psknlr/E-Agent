"""Interface ``retrieve_evidence``: assemble the evidence, and show its holes.

Why this step exists
--------------------
The question "which enzymes are known to do this?" is answered badly in two
opposite ways. A search that is too narrow returns three papers and the project
concludes the chemistry is unexplored. A search that is too broad returns four
database records that are four re-publications of one 1998 measurement, and the
project concludes it is well established. Both failures are invisible in the
output, because a list of records looks the same either way.

So this step produces three artifacts, in this order of importance:

1. **The query plan**, written before anything is retrieved. Recall cannot be
   judged from results; it can only be judged from the queries. A human reads
   ``evidence_query_plan.yaml`` and says "you never searched the cyclic
   ketones" -- which is a conversation that cannot happen if the plan exists
   only inside the code.
2. **The evidence matrix**, binned on family x substrate chemotype, with the
   outcome classes kept apart. ``confirmed``, ``not detected``, ``expression
   failure`` and ``untested`` collapse to the same empty cell in a normal
   screening table, and that collapse is what makes public enzyme data unusable
   for substrate-level prediction. Here they stay distinct, and the cell also
   separates wild-type from engineered success, so a chemotype that only works
   after protein engineering cannot be read as a chemotype with mature natural
   enzymes.
3. **The records themselves**, as typed
   :class:`~eagent.schemas.record.ExperimentRecord` objects whose evidence
   strength has been capped at what the source is entitled to claim.

Offline by default. Every connector consults a local cache; a miss is reported
as a gap naming the exact file a curator must provide, and never as an absence
of activity. That distinction is the single most important thing this module
does: a cache miss and a measured negative are opposite facts, and only one of
them is evidence.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Sequence

import yaml

from ..connectors.base import (
    AccessPolicy, CachedResponse, Connector, FileCache, connector_spec,
    records_in, resolve_strength,
)
from ..context import RunContext
from ..datalayer.intake import direction_check
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..provenance import canonical_json, sha256_file, sha256_obj, sha256_text, utc_now
from ..schemas import (
    CofactorSpec, Conditions, Detection, EvidenceRef, EvidenceStrength,
    ExperimentRecord, OutcomeClass, ProductSpec, ReactionClass,
    ReactionDirection, SubstrateSpec, TaskSpec,
)
from .base import ScientificInterface
from .normalize_reaction import SUBSTRATE_CLASS_SCAFFOLD, SubstrateChemotype

__all__ = [
    "DEFAULT_ENGINEERING_KEYWORDS",
    "MIN_INDEPENDENT_SOURCES_FOR_MATURE",
    "PlannedQuery",
    "SkippedConnector",
    "QueryPlan",
    "build_query_plan",
    "ChemotypeAssignment",
    "assign_chemotype",
    "EvidenceRow",
    "ExtractionFailure",
    "extract_records",
    "Maturity",
    "MatrixCell",
    "EvidenceMatrix",
    "build_evidence_matrix",
    "RetrieveEvidence",
]


#: Search terms that find engineering campaigns. These are methodological
#: keywords, not scientific claims: they widen literature recall toward variants
#: that never reach a structured database. Overridable per run, and written into
#: the plan artifact so a reviewer can see exactly what was searched for.
DEFAULT_ENGINEERING_KEYWORDS: tuple[str, ...] = (
    "directed evolution", "variant", "mutant", "engineered",
    "saturation mutagenesis", "substrate scope", "enantioselectivity",
    "stereoselectivity", "activity improvement",
)

#: How many independent sources a cell needs before it is reported as having
#: mature natural enzymes. A REPORTING convention for the evidence matrix, not a
#: catalytic threshold: it decides what a summary cell is allowed to say, never
#: whether a candidate passes. Two is the smallest number at which one
#: re-published measurement cannot by itself create the impression of
#: consensus. Calibrate per family and per project, and record the change in
#: the run config.
MIN_INDEPENDENT_SOURCES_FOR_MATURE: int = 2

#: Column order of the evidence matrix. Fixed so two runs' matrices diff.
CHEMOTYPE_COLUMNS: tuple[SubstrateChemotype, ...] = (
    SubstrateChemotype.AROMATIC_KETONE,
    SubstrateChemotype.ALIPHATIC_KETONE,
    SubstrateChemotype.CYCLIC_KETONE,
    SubstrateChemotype.FUNCTIONALISED_KETONE,
    SubstrateChemotype.UNCLASSIFIED,
)

#: Row label for records whose family nobody recorded. A visible row, not a
#: silent drop: unassigned evidence is a curation gap, and hiding it would make
#: the matrix look more complete than the data is.
UNASSIGNED_FAMILY = "unassigned_family"


# ---------------------------------------------------------------------------
# query plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedQuery:
    """One query, with the reason it is being asked and the limits of its answer.

    ``must_not_be_used_for`` is copied from the connector registry into the plan
    so that the artifact is self-contained: a reviewer reading the plan six
    months later sees that the BRENDA query cannot settle the atom mapping,
    without having to go and look the resource up.
    """

    query_id: str
    connector: str
    data_layer: str
    purpose: str
    query: Mapping[str, Any]
    evidence_strength_ceiling: str
    must_not_be_used_for: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "connector": self.connector,
            "data_layer": self.data_layer,
            "purpose": self.purpose,
            "query": dict(self.query),
            "evidence_strength_ceiling": self.evidence_strength_ceiling,
            "must_not_be_used_for": list(self.must_not_be_used_for),
        }


@dataclass(frozen=True)
class SkippedConnector:
    """A resource that was not queried, and precisely why.

    Recorded because "we did not search there" and "we searched and found
    nothing" are different facts about the world, and a plan that lists only the
    queries it ran lets the first silently masquerade as the second.
    """

    connector: str
    data_layer: str
    reason: str
    unblocked_by: str

    def to_dict(self) -> dict[str, Any]:
        return {"connector": self.connector, "data_layer": self.data_layer,
                "reason": self.reason, "unblocked_by": self.unblocked_by}


@dataclass
class QueryPlan:
    """The whole search, written down before it runs."""

    task_id: str
    target_reaction: dict[str, Any]
    substrate_synonyms: tuple[str, ...]
    cofactors: tuple[str, ...]
    candidate_families: tuple[str, ...]
    engineering_keywords: tuple[str, ...]
    queries: tuple[PlannedQuery, ...]
    skipped: tuple[SkippedConnector, ...]
    recall_caveats: tuple[str, ...]

    @property
    def plan_sha256(self) -> str:
        """Hash of the plan, so a later run can prove it searched the same way."""
        return sha256_text(canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_reaction": self.target_reaction,
            "substrate_synonyms": list(self.substrate_synonyms),
            "cofactors": list(self.cofactors),
            "candidate_families": list(self.candidate_families),
            "engineering_keywords": list(self.engineering_keywords),
            "queries": [q.to_dict() for q in self.queries],
            "skipped_connectors": [s.to_dict() for s in self.skipped],
            "recall_caveats": list(self.recall_caveats),
        }


def _qid(connector: str, query: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json({"c": connector, "q": dict(query)}))[:16]


def _families_from_context(ctx: RunContext,
                           families: Sequence[str] | None) -> tuple[str, ...]:
    """Candidate family names from the operator, the config, or the templates.

    Never from this module. Naming the families that matter for a reaction is a
    scientific claim; a hard-coded list here would quietly become the search's
    only axis and would silently exclude whatever the author had not thought of.
    """
    if families:
        return tuple(dict.fromkeys(str(f) for f in families))
    cfg = ctx.config.get("candidate_families")
    if isinstance(cfg, Sequence) and not isinstance(cfg, str):
        return tuple(dict.fromkeys(str(f) for f in cfg))
    lib = getattr(ctx, "templates", None)
    for attr in ("family_templates", "families"):
        coll = getattr(lib, attr, None) if lib is not None else None
        items: Iterable[Any]
        if isinstance(coll, Mapping):
            items = coll.values()
        elif isinstance(coll, Sequence) and not isinstance(coll, str):
            items = coll
        else:
            continue
        names = [str(getattr(t, "family_name")) for t in items
                 if getattr(t, "family_name", None)]
        if names:
            return tuple(dict.fromkeys(names))
    return ()


def build_query_plan(
    task: TaskSpec,
    *,
    families: Sequence[str] = (),
    substrate_synonyms: Sequence[str] = (),
    engineering_keywords: Sequence[str] = DEFAULT_ENGINEERING_KEYWORDS,
    connector_keys: Sequence[str] = (
        "rhea", "enzymemap", "brenda", "sabio_rk", "mcsa", "uniprot",
        "interpro", "pubmed", "pdb", "alphafold",
    ),
) -> QueryPlan:
    """Turn the reaction spec into an auditable list of queries.

    A query is only planned when every term it needs is actually known. A
    connector whose fields cannot be filled is recorded as *skipped* with the
    input that would unblock it, rather than being sent a query built from a
    placeholder -- a search for the string ``"None"`` returns nothing and looks
    exactly like a search that found nothing.
    """
    rxn = task.reaction
    synonyms: list[str] = []
    for candidate in (rxn.substrate.name, *substrate_synonyms):
        if candidate and candidate not in synonyms:
            synonyms.append(str(candidate))

    cofactors = tuple(c.describe() for c in task.conditions.cofactor_options)
    fams = tuple(dict.fromkeys(str(f) for f in families))
    kws = tuple(dict.fromkeys(str(k) for k in engineering_keywords))

    target = {
        "reaction_class": rxn.reaction_class.value,
        "substrate_smiles": rxn.substrate.isomeric_smiles,
        "substrate_inchikey": rxn.substrate.inchikey,
        "product_smiles": rxn.product.isomeric_smiles,
        "target_stereochemistry": rxn.product.target_stereochemistry.value,
        "rhea_id": rxn.rhea_id,
        "ec_hint": rxn.ec_hint,
        "atom_mapped_reaction_smiles": rxn.atom_mapped_reaction_smiles,
        "direction_requirement":
            "records measured in the reverse direction do not support the "
            "target direction and are counted separately",
    }

    queries: list[PlannedQuery] = []
    skipped: list[SkippedConnector] = []

    def plan(key: str, purpose: str, query: Mapping[str, Any]) -> None:
        spec = connector_spec(key)
        queries.append(PlannedQuery(
            query_id=_qid(key, query), connector=key,
            data_layer=spec.data_layer.value, purpose=purpose,
            query=dict(query),
            evidence_strength_ceiling=spec.evidence_strength_ceiling.value,
            must_not_be_used_for=spec.not_good_for,
        ))

    def skip(key: str, reason: str, unblocked_by: str) -> None:
        spec = connector_spec(key)
        skipped.append(SkippedConnector(key, spec.data_layer.value, reason,
                                        unblocked_by))

    wanted = list(connector_keys)

    # -- reaction layer: define the transformation -------------------------
    if "rhea" in wanted:
        if rxn.rhea_id:
            plan("rhea", "pin the target transformation to a defined reaction",
                 {"rhea_id": rxn.rhea_id})
        elif rxn.ec_hint:
            plan("rhea", "find the balanced reactions behind the EC hint",
                 {"ec": rxn.ec_hint})
        else:
            skip("rhea", "no Rhea id and no EC hint, so there is nothing to "
                         "resolve the transformation against",
                 "an operator-supplied EC number or Rhea id")
    if "enzymemap" in wanted:
        if rxn.ec_hint:
            plan("enzymemap", "obtain an atom-mapped form of the reaction",
                 {"ec": rxn.ec_hint})
        elif rxn.substrate.isomeric_smiles and rxn.product.isomeric_smiles:
            plan("enzymemap", "obtain an atom-mapped form of the reaction",
                 {"reaction_smiles": f"{rxn.substrate.isomeric_smiles}>>"
                                     f"{rxn.product.isomeric_smiles}"})
        else:
            skip("enzymemap", "no EC hint and no substrate/product structures",
                 "an EC number, or both structures as SMILES")

    # -- kinetics layer: who was measured on what --------------------------
    for key in ("brenda", "sabio_rk"):
        if key not in wanted:
            continue
        planned_any = False
        for syn in synonyms:
            q: dict[str, Any] = {"substrate": syn}
            if rxn.ec_hint:
                q["ec"] = rxn.ec_hint
            plan(key, f"reported activity on substrate synonym '{syn}'", q)
            planned_any = True
        if rxn.ec_hint and not planned_any:
            plan(key, "reported activity for the EC class", {"ec": rxn.ec_hint})
            planned_any = True
        if not planned_any:
            skip(key, "no substrate name or synonym and no EC hint to search on",
                 "a substrate name or an EC number; note that the structure "
                 "alone is not searchable here")

    # -- mechanism layer ---------------------------------------------------
    if "mcsa" in wanted:
        if rxn.ec_hint:
            plan("mcsa", "catalytic residues and mechanism for the EC class",
                 {"ec": rxn.ec_hint})
        else:
            skip("mcsa", "M-CSA is searched by EC, accession or PDB id, none of "
                         "which is known yet",
                 "an EC number, or a seed accession from the sequence layer")

    # -- sequence and family layers ---------------------------------------
    if "uniprot" in wanted:
        if fams:
            for fam in fams:
                q = {"family": fam}
                if rxn.ec_hint:
                    q["ec"] = rxn.ec_hint
                plan("uniprot", f"seed sequences annotated to family '{fam}'", q)
        elif rxn.ec_hint:
            plan("uniprot", "seed sequences annotated with the EC hint",
                 {"ec": rxn.ec_hint})
        else:
            skip("uniprot", "no candidate family names and no EC hint",
                 "candidate families from a sourced FamilyTemplate, or an EC "
                 "number")
    if "interpro" in wanted:
        if fams:
            for fam in fams:
                plan("interpro", f"signature definition for family '{fam}'",
                     {"family_name": fam})
        else:
            skip("interpro", "no candidate family names to resolve to signatures",
                 "candidate families from a sourced FamilyTemplate")

    # -- structure layer: nothing to ask until there are accessions --------
    for key in ("pdb", "alphafold"):
        if key in wanted:
            skip(key, "structure retrieval is keyed on accessions, which do not "
                      "exist until the sequence-mining step has run",
                 "the candidate accession list from mine_sequences")

    # -- literature layer --------------------------------------------------
    if "pubmed" in wanted:
        terms = synonyms or ([rxn.ec_hint] if rxn.ec_hint else [])
        if terms:
            for term in terms:
                parts = [str(term)]
                if fams:
                    parts.append("(" + " OR ".join(f'"{f}"' for f in fams) + ")")
                if kws:
                    parts.append("(" + " OR ".join(f'"{k}"' for k in kws) + ")")
                plan("pubmed",
                     f"primary reports and engineering campaigns for '{term}'",
                     {"query": " AND ".join(parts)})
        else:
            skip("pubmed", "no substrate synonym and no EC hint to anchor a "
                           "literature query",
                 "a substrate name, or an EC number")

    caveats = [
        "A cache miss is a fact about this machine, not about the literature: "
        "it is never evidence that no enzyme performs this reaction.",
        "Substrate matching in the kinetics layer is by name or EC, so a "
        "substrate whose records use a different synonym will be missed; the "
        "synonym list is the recall limit of this whole step.",
        "Records keyed to an EC number and an organism are capped at "
        "ec_species_mapped and must not be read as sequence-level evidence.",
        "Several of these resources re-publish one another's records; counting "
        "hits overstates corroboration, so the matrix counts independent "
        "sources instead.",
    ]
    if not fams:
        caveats.append(
            "No candidate family names were supplied, so the family axis of the "
            "evidence matrix is populated only from whatever the records "
            "themselves declare.")
    if not synonyms:
        caveats.append(
            "No substrate name or synonym was supplied, so the kinetics layer "
            "could only be searched by EC class, if at all.")

    return QueryPlan(
        task_id=task.task_id, target_reaction=target,
        substrate_synonyms=tuple(synonyms), cofactors=cofactors,
        candidate_families=fams, engineering_keywords=kws,
        queries=tuple(queries), skipped=tuple(skipped),
        recall_caveats=tuple(caveats),
    )


# ---------------------------------------------------------------------------
# chemotype binning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChemotypeAssignment:
    """Which sub-space a record's substrate belongs to, and on whose authority.

    ``basis`` exists because the chemotype is the x-axis of the evidence matrix:
    if an assignment were guessed from a SMILES substring, the matrix would
    present a guess as a finding. Only a declaration in the curated payload or
    an operator-supplied assignment table counts; anything else is
    ``UNCLASSIFIED`` with ``basis='unassigned'``, which shows up as its own
    column rather than being quietly spread across the others.
    """

    chemotype: SubstrateChemotype
    basis: str                      # "payload_declared" | "operator_table" | "unassigned"
    key: str | None = None          # the identifier the assignment was keyed on

    def to_dict(self) -> dict[str, Any]:
        return {"chemotype": self.chemotype.value, "basis": self.basis,
                "key": self.key}


def assign_chemotype(raw: Mapping[str, Any],
                     assignments: Mapping[str, str] | None = None
                     ) -> ChemotypeAssignment:
    """Bin a record's substrate, or admit that nobody has binned it.

    Deliberately has no structure-perception branch. Deciding from a SMILES
    whether a ketone is "aromatic" needs a toolkit and a definition of which
    ring counts; a substring check would call every substrate with a phenyl
    group an aromatic ketone, including an aliphatic ketone three carbons away
    from a benzene ring. Being wrong on that axis mislabels the entire matrix.
    """
    declared = raw.get("chemotype")
    if isinstance(declared, str):
        try:
            return ChemotypeAssignment(SubstrateChemotype(declared),
                                       "payload_declared")
        except ValueError:
            pass  # an unknown label is not silently coerced into a known one

    table = {str(k): str(v) for k, v in (assignments or {}).items()}
    sub = raw.get("substrate") if isinstance(raw.get("substrate"), Mapping) else {}
    for field_name in ("inchikey", "isomeric_smiles", "name"):
        key = sub.get(field_name) if isinstance(sub, Mapping) else None
        if key and str(key) in table:
            try:
                return ChemotypeAssignment(SubstrateChemotype(table[str(key)]),
                                           "operator_table", str(key))
            except ValueError:
                continue
    return ChemotypeAssignment(SubstrateChemotype.UNCLASSIFIED, "unassigned")


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionFailure:
    """A payload record that could not become an ExperimentRecord, and why.

    Kept and reported rather than dropped. A record rejected because a negative
    carries no detection limit is a curation task, and a run that silently
    discarded it would show a cleaner evidence base than it has.
    """

    connector: str
    raw_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"connector": self.connector, "raw_id": self.raw_id,
                "reason": self.reason}


@dataclass(frozen=True)
class EvidenceRow:
    """One ExperimentRecord plus the annotations the matrix is binned on.

    The annotations live outside the record because
    :class:`~eagent.schemas.record.ExperimentRecord` forbids extra fields, and
    rightly so: a family label and a chemotype are interpretations layered on a
    measurement, and keeping them separate keeps their provenance separate too.
    """

    record: ExperimentRecord
    connector: str
    family: str
    family_basis: str
    chemotype: ChemotypeAssignment
    strength_decisions: tuple[dict[str, Any], ...] = ()
    database_version: str | None = None
    retrieved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.model_dump(mode="json"),
            "annotation": {
                "connector": self.connector,
                "family": self.family,
                "family_basis": self.family_basis,
                "chemotype": self.chemotype.to_dict(),
                "database_version": self.database_version,
                "retrieved_at": self.retrieved_at,
            },
            "evidence_strength_decisions": [dict(d) for d in
                                            self.strength_decisions],
        }


def _tsv_safe(value: Any) -> str:
    """Flatten a value into one TSV field.

    A pydantic validation message spans several lines, and writing it raw would
    turn one rejected record into several malformed rows -- quietly corrupting
    the very report that is supposed to make the rejections auditable.
    """
    return " ".join(str(value).split())


def _coerce(enum_cls: Any, value: Any, default: Any) -> Any:
    """Parse an enum value, falling back only to an explicit 'unknown' member."""
    if value is None:
        return default
    try:
        return enum_cls(value)
    except ValueError:
        return default


def extract_records(
    connector_key: str,
    response: CachedResponse,
    *,
    chemotype_assignments: Mapping[str, str] | None = None,
) -> tuple[list[EvidenceRow], list[ExtractionFailure]]:
    """Turn a cached payload into typed records, capping what each may claim.

    Three things happen here that must not be skipped.

    *The evidence ceiling is applied.* A BRENDA record claiming sequence-level
    experimental evidence is capped at ``ec_species_mapped`` unless the payload
    names a human verifier, because a record keyed to an EC number and an
    organism does not identify a sequence. This is the step that stops "some
    enzyme of this class was reported active" turning into "this protein is
    active".

    *Shared upstreams travel with the record.* ``upstream_sources`` is seeded
    from the connector registry's lineage, so the downstream independence
    accounting can see that EnzymeMap and BRENDA are not two witnesses.

    *Schema rejections become reported failures.* A positive without a
    product-identifying detection method, or a negative without a detection
    limit, is refused by :class:`ExperimentRecord` itself. That refusal is
    caught and reported as an :class:`ExtractionFailure` -- never repaired by
    inventing the missing field.
    """
    spec = connector_spec(connector_key)
    rows: list[EvidenceRow] = []
    failures: list[ExtractionFailure] = []

    for n, raw in enumerate(records_in(response.payload)):
        raw_id = str(raw.get("record_id") or raw.get("id") or f"<index {n}>")
        try:
            row = _one_record(spec, connector_key, response, raw, raw_id,
                              chemotype_assignments)
        except _RecordRejected as exc:
            failures.append(ExtractionFailure(connector_key, raw_id, str(exc)))
        except Exception as exc:          # pydantic validation and friends
            failures.append(ExtractionFailure(
                connector_key, raw_id,
                f"{type(exc).__name__}: {exc}"))
        else:
            rows.append(row)
    return rows, failures


class _RecordRejected(Exception):
    """A payload record is missing something that cannot be supplied by code."""


def _one_record(spec: Any, connector_key: str, response: CachedResponse,
                raw: Mapping[str, Any], raw_id: str,
                chemotype_assignments: Mapping[str, str] | None) -> EvidenceRow:
    if not raw.get("record_id"):
        raise _RecordRejected(
            "payload record has no record_id; without a stable id the record "
            "cannot be de-duplicated or traced back to its source")

    outcome_raw = raw.get("outcome")
    if outcome_raw is None:
        raise _RecordRejected(
            "payload record states no outcome; 'not tested', 'not detected' and "
            "'expression failure' are different facts and none may be assumed")
    try:
        outcome = OutcomeClass(outcome_raw)
    except ValueError as exc:
        raise _RecordRejected(f"unknown outcome '{outcome_raw}'") from exc

    sub_raw = raw.get("substrate")
    if not isinstance(sub_raw, Mapping) or not any(
            sub_raw.get(k) for k in ("isomeric_smiles", "inchikey", "name")):
        raise _RecordRejected(
            "payload record identifies no substrate; a record that cannot be "
            "placed on the substrate axis cannot enter the evidence matrix")
    substrate = SubstrateSpec.model_validate(dict(sub_raw))

    product = None
    if isinstance(raw.get("product_observed"), Mapping):
        product = ProductSpec.model_validate(dict(raw["product_observed"]))

    cofactor = None
    if isinstance(raw.get("cofactor"), Mapping):
        cofactor = CofactorSpec.model_validate(dict(raw["cofactor"]))

    conditions = Conditions()
    if isinstance(raw.get("conditions"), Mapping):
        conditions = Conditions.model_validate(dict(raw["conditions"]))

    detection = Detection()
    if isinstance(raw.get("detection"), Mapping):
        detection = Detection.model_validate(dict(raw["detection"]))

    evidence, decisions = _evidence_refs(spec, connector_key, response, raw)
    if not evidence:
        raise _RecordRejected(
            "payload record carries no evidence reference; an unciteable record "
            "cannot be audited and must not enter the evidence base")

    record = ExperimentRecord(
        record_id=f"{connector_key}:{raw['record_id']}",
        sequence=raw.get("sequence"),
        accession=raw.get("accession"),
        database_version=response.database_version,
        is_variant=bool(raw.get("is_variant", False)),
        parent_sequence_sha256=raw.get("parent_sequence_sha256"),
        mutations=list(raw.get("mutations") or []),
        construct_description=raw.get("construct_description"),
        construct_sequence=raw.get("construct_sequence"),
        substrate=substrate,
        product_observed=product,
        reaction_class=_coerce(ReactionClass, raw.get("reaction_class"),
                               ReactionClass.OTHER),
        reaction_direction=_coerce(ReactionDirection,
                                   raw.get("reaction_direction"),
                                   ReactionDirection.UNSPECIFIED),
        reaction_id=raw.get("reaction_id"),
        cofactor=cofactor,
        conditions=conditions,
        outcome=outcome,
        detection=detection,
        conversion_pct=raw.get("conversion_pct"),
        ee_target_pct=raw.get("ee_target_pct"),
        specific_activity=raw.get("specific_activity"),
        specific_activity_unit=raw.get("specific_activity_unit"),
        measurement_type=raw.get("measurement_type"),
        measurement_value=raw.get("measurement_value"),
        measurement_unit=raw.get("measurement_unit"),
        kcat_s=raw.get("kcat_s"),
        km_mM=raw.get("km_mM"),
        soluble_expression=raw.get("soluble_expression"),
        evidence=evidence,
        notes=str(raw.get("notes") or ""),
    )

    family = str(raw.get("family") or "").strip()
    return EvidenceRow(
        record=record,
        connector=connector_key,
        family=family or UNASSIGNED_FAMILY,
        family_basis="payload_declared" if family else "unassigned",
        chemotype=assign_chemotype(raw, chemotype_assignments),
        strength_decisions=tuple(decisions),
        database_version=response.database_version,
        retrieved_at=response.retrieved_at,
    )


def _evidence_refs(spec: Any, connector_key: str, response: CachedResponse,
                   raw: Mapping[str, Any]) -> tuple[list[EvidenceRef],
                                                    list[dict[str, Any]]]:
    """Build the citations, capping each at what the source may claim."""
    refs: list[EvidenceRef] = []
    decisions: list[dict[str, Any]] = []
    items = raw.get("evidence")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return refs, decisions

    for item in items:
        if not isinstance(item, Mapping):
            continue
        identifier = item.get("identifier")
        if not identifier:
            continue      # an evidence entry with no identifier cites nothing
        claimed = _coerce(EvidenceStrength, item.get("strength"),
                          EvidenceStrength.ANNOTATION_ONLY)
        verifier = item.get("verified_by")
        decision = resolve_strength(connector_key, claimed,
                                    human_reviewer=verifier)
        decisions.append(decision.to_dict())
        upstream = list(dict.fromkeys(
            [*(item.get("upstream_sources") or []), *spec.derived_from]))
        refs.append(EvidenceRef(
            source_type=str(item.get("source_type") or "database"),
            identifier=str(identifier),
            locator=item.get("locator"),
            strength=decision.granted,
            extracted_by="automated_ingest:eagent.retrieve_evidence",
            verified_by=verifier,
            quote=item.get("quote"),
            retrieved_at=response.retrieved_at,
            database_version=response.database_version,
            source_doi=item.get("source_doi"),
            source_record_id=item.get("source_record_id"),
            license=item.get("license"),
            upstream_sources=upstream,
            experiment_activity_id=item.get("experiment_activity_id"),
        ))
    return refs, decisions


# ---------------------------------------------------------------------------
# the evidence matrix
# ---------------------------------------------------------------------------

class Maturity(str, enum.Enum):
    """What a matrix cell is entitled to say about a family x chemotype pair.

    A reporting vocabulary, not a score. Its whole purpose is to keep three
    situations apart that a count of positives merges: a chemotype with
    several independent wild-type successes, one that only works after protein
    engineering, and one that has simply never been tested.
    """

    MATURE_NATURAL = "mature_natural"              # independent wild-type successes
    NATURAL_SINGLE_REPORT = "natural_single_report"  # one wild-type success only
    ENGINEERED_ONLY = "engineered_only"            # succeeds only as a variant
    TESTED_NEGATIVE = "tested_negative"            # tried, product not detected
    EXPRESSION_LIMITED = "expression_limited"      # never expressed; ability unknown
    REVERSE_DIRECTION_ONLY = "reverse_direction_only"
    NO_RELIABLE_DATA = "no_reliable_data"          # only annotation or in-silico
    UNTESTED = "untested"                          # no record at all

    def claim(self) -> str:
        return _MATURITY_CLAIM[self]


_MATURITY_CLAIM: dict[Maturity, str] = {
    Maturity.MATURE_NATURAL:
        "independent wild-type successes exist for this chemotype",
    Maturity.NATURAL_SINGLE_REPORT:
        "one wild-type success; not corroborated independently",
    Maturity.ENGINEERED_ONLY:
        "success reported only for engineered variants; a wild-type starting "
        "point for this chemotype is not evidenced here",
    Maturity.TESTED_NEGATIVE:
        "tested, target product not detected at the stated detection limit",
    Maturity.EXPRESSION_LIMITED:
        "constructs failed expression; catalytic ability is undetermined, which "
        "is not the same as inactive",
    Maturity.REVERSE_DIRECTION_ONLY:
        "activity reported only in the reverse direction; it does not support "
        "the target direction",
    Maturity.NO_RELIABLE_DATA:
        "only annotation-level or computational entries; essentially no "
        "reliable experimental data",
    Maturity.UNTESTED:
        "no record at all for this combination",
}


@dataclass
class MatrixCell:
    """Counts for one family x chemotype pair, with the outcomes kept apart."""

    confirmed: int = 0
    confirmed_wild_type: int = 0
    confirmed_variant: int = 0
    confirmed_reverse_direction: int = 0
    #: Confirmed, but the direction was never recorded. Kept apart from the
    #: reverse bucket: not knowing which way a measurement ran is a different
    #: state from knowing it ran the other way, and the remedies differ.
    confirmed_direction_unknown: int = 0
    #: Confirmed, but the record contradicts itself -- its declared direction
    #: and its own chemistry disagree. Counted nowhere else.
    confirmed_direction_conflict: int = 0
    not_detected: int = 0
    #: Not detected, but measured in the reverse direction. A failure to see
    #: the oxidation is not a failure to see the reduction.
    not_detected_reverse_direction: int = 0
    expression_failure: int = 0
    other_product: int = 0
    not_tested: int = 0
    computational_only: int = 0
    independent_sources: int = 0
    #: Independent sources among the **wild-type** confirmations only.
    #:
    #: Counted apart from :attr:`independent_sources` because the maturity
    #: question is about wild-type enzymes. One wild-type success plus one
    #: engineered-variant success from another source made the pooled count
    #: reach two and the cell read "independent wild-type successes exist for
    #: this chemotype" -- while there was exactly one wild-type success and
    #: the second witness was about a protein somebody had already had to
    #: engineer. That is the difference between "pick one off the shelf" and
    #: "budget an engineering campaign".
    independent_wild_type_sources: int = 0
    max_strength: EvidenceStrength = EvidenceStrength.COMPUTATIONAL_CONSTRUCT
    record_ids: list[str] = field(default_factory=list)

    @property
    def n_records(self) -> int:
        return len(self.record_ids)

    def maturity(self) -> Maturity:
        """Classify the cell. Order matters and is argued in the comments."""
        if self.n_records == 0:
            return Maturity.UNTESTED
        if self.confirmed_wild_type > 0:
            # Corroboration is counted in independent sources, not in records:
            # four databases re-publishing one measurement is one witness.
            # And in the sources of the WILD-TYPE confirmations: a variant's
            # success corroborates the variant, not the natural enzyme.
            if (self.independent_wild_type_sources
                    >= MIN_INDEPENDENT_SOURCES_FOR_MATURE):
                return Maturity.MATURE_NATURAL
            return Maturity.NATURAL_SINGLE_REPORT
        if self.confirmed_variant > 0:
            return Maturity.ENGINEERED_ONLY
        if self.confirmed_reverse_direction > 0:
            return Maturity.REVERSE_DIRECTION_ONLY
        if self.not_detected > 0 or self.other_product > 0:
            # A real negative outranks an expression failure: it tested the
            # chemistry, whereas a failed construct tested the construct.
            return Maturity.TESTED_NEGATIVE
        if self.expression_failure > 0:
            return Maturity.EXPRESSION_LIMITED
        return Maturity.NO_RELIABLE_DATA

    def render(self) -> str:
        """Compact TSV cell: the classification plus every count behind it.

        An empty cell renders ``str=none`` rather than the enum's zero-rank
        default, because printing ``computational_construct`` where no record
        exists would suggest a computational entry was found and dismissed.
        """
        strength = self.max_strength.value if self.n_records else "none"
        return (f"{self.maturity().value} c={self.confirmed} "
                f"wt={self.confirmed_wild_type} var={self.confirmed_variant} "
                f"rev={self.confirmed_reverse_direction} "
                f"nd={self.not_detected} ef={self.expression_failure} "
                f"op={self.other_product} ut={self.not_tested} "
                f"comp={self.computational_only} "
                f"ind={self.independent_sources} "
                f"indwt={self.independent_wild_type_sources} "
                f"str={strength}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maturity": self.maturity().value,
            "confirmed": self.confirmed,
            "confirmed_wild_type": self.confirmed_wild_type,
            "confirmed_variant": self.confirmed_variant,
            "confirmed_reverse_direction": self.confirmed_reverse_direction,
            "not_detected": self.not_detected,
            "expression_failure": self.expression_failure,
            "other_product": self.other_product,
            "not_tested": self.not_tested,
            "computational_only": self.computational_only,
            "independent_sources": self.independent_sources,
            "independent_wild_type_sources": self.independent_wild_type_sources,
            "max_strength": self.max_strength.value,
            "n_records": self.n_records,
        }


@dataclass
class EvidenceMatrix:
    """family x chemotype x outcome, with the method used to count corroboration."""

    cells: dict[tuple[str, SubstrateChemotype], MatrixCell]
    families: tuple[str, ...]
    chemotypes: tuple[SubstrateChemotype, ...]
    independence_method: str

    def cell(self, family: str, chemotype: SubstrateChemotype) -> MatrixCell:
        return self.cells.get((family, chemotype), MatrixCell())

    def to_tsv(self, *, extra_header: Sequence[str] = ()) -> str:
        """Render with a legend, because an undocumented cell code is noise."""
        lines: list[str] = [
            "# Evidence matrix: family (rows) x substrate chemotype (columns).",
            "# Each cell: <maturity> then the counts it was derived from.",
            "#   c   confirmed target product (target direction only)",
            "#   wt  confirmed by a wild-type sequence",
            "#   var confirmed only by an engineered variant",
            "#   rev confirmed, but measured in the reverse direction; this does "
            "not support the target direction",
            "#   nd  tested, target product not detected at the stated limit",
            "#   ef  expression or solubility failure; catalytic ability unknown",
            "#   op  turnover to another product or the wrong configuration",
            "#   ut  present but not tested",
            "#   comp computational entries only; never an experimental negative",
            "#   ind number of independent sources behind the confirmations",
            "#   str strongest evidence any record in the cell is entitled to claim",
            "#",
            "# Maturity vocabulary:",
            *[f"#   {m.value}: {m.claim()}" for m in Maturity],
            "#",
            f"# Corroboration counted by: {self.independence_method}",
            f"# A cell reading '{Maturity.UNTESTED.value}' means no record was "
            f"retrieved; it is not a negative result.",
            *[f"# {_tsv_safe(line)}" for line in extra_header],
        ]
        header = ["family"] + [c.value for c in self.chemotypes]
        lines.append("\t".join(header))
        for fam in self.families:
            row = [_tsv_safe(fam)] + [self.cell(fam, c).render()
                                      for c in self.chemotypes]
            lines.append("\t".join(row))
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "families": list(self.families),
            "chemotypes": [c.value for c in self.chemotypes],
            "independence_method": self.independence_method,
            "cells": {f"{fam}|{ct.value}": self.cell(fam, ct).to_dict()
                      for fam in self.families for ct in self.chemotypes},
        }


def _count_independent(records: Sequence[ExperimentRecord]) -> tuple[int, str]:
    """Count witnesses, not hits, preferring the project's lineage machinery.

    Four databases carrying one re-curated measurement are one piece of
    evidence. :mod:`eagent.datalayer.lineage` knows how to collapse them; when
    it cannot be imported, the fallback counts distinct citation identifiers,
    which is weaker (it cannot see that two PMIDs report the same campaign) and
    says so in the method string rather than pretending otherwise.
    """
    if not records:
        return 0, "no records"
    try:
        from ..datalayer.lineage import count_independent
    except Exception:
        pass
    else:
        try:
            return int(count_independent(list(records))), \
                "eagent.datalayer.lineage.count_independent"
        except Exception:
            pass
    ids = {ev.identifier for r in records for ev in r.evidence if ev.identifier}
    return len(ids), ("distinct citation identifiers (lineage grouping "
                      "unavailable; shared campaigns are not collapsed)")


def build_evidence_matrix(rows: Sequence[EvidenceRow],
                          target_reaction: Any = None) -> EvidenceMatrix:
    """Bin rows into the family x chemotype matrix, keeping outcomes apart.

    A confirmed record measured in the reverse direction is counted in its own
    bucket and excluded from ``confirmed``. An alcohol-oxidation measurement is
    not evidence that the same enzyme performs the reduction under the target
    conditions, and every published dataset contains some.
    """
    cells: dict[tuple[str, SubstrateChemotype], MatrixCell] = {}
    confirmed_records: dict[tuple[str, SubstrateChemotype],
                            list[ExperimentRecord]] = {}
    families: list[str] = []

    for row in rows:
        key = (row.family, row.chemotype.chemotype)
        cell = cells.setdefault(key, MatrixCell())
        if row.family not in families:
            families.append(row.family)
        rec = row.record
        cell.record_ids.append(rec.record_id)
        if rec.max_strength.rank > cell.max_strength.rank:
            cell.max_strength = rec.max_strength

        outcome = rec.outcome
        # Both signals, not just the label. A record can carry
        # forward_as_target and describe the oxidation in its own reaction
        # class, substrate and product; trusting the label alone counts that
        # as evidence for the reduction. direction_check already reads both
        # and is reused here rather than re-implemented, so the matrix cannot
        # drift away from the intake rule.
        verdict = (direction_check(rec, target_reaction)
                   if target_reaction is not None else None)
        if verdict is not None:
            supports = verdict.supports
            is_reverse = verdict.is_reverse
            unknown = verdict.is_unspecified
            conflict = not supports and not is_reverse and not unknown
        else:
            supports = rec.reaction_direction.supports_target_direction
            is_reverse = (rec.reaction_direction
                          is ReactionDirection.REVERSE_OF_TARGET)
            unknown = not supports and not is_reverse
            conflict = False

        if outcome is OutcomeClass.CONFIRMED_TARGET_PRODUCT:
            if supports:
                cell.confirmed += 1
                if rec.is_variant:
                    cell.confirmed_variant += 1
                else:
                    cell.confirmed_wild_type += 1
                confirmed_records.setdefault(key, []).append(rec)
            elif is_reverse:
                cell.confirmed_reverse_direction += 1
            elif conflict:
                cell.confirmed_direction_conflict += 1
            else:
                cell.confirmed_direction_unknown += 1
        elif outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED:
            if supports:
                cell.not_detected += 1
            else:
                # Not seeing the oxidation is not not seeing the reduction.
                cell.not_detected_reverse_direction += 1
        elif outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE:
            cell.expression_failure += 1
        elif outcome is OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION:
            cell.other_product += 1
        elif outcome is OutcomeClass.NOT_TESTED:
            cell.not_tested += 1
        else:
            cell.computational_only += 1

    method = "no confirmed records to count"
    for key, recs in confirmed_records.items():
        n, method = _count_independent(recs)
        cells[key].independent_sources = n
        wild_type = [r for r in recs if not r.is_variant]
        if wild_type:
            cells[key].independent_wild_type_sources = (
                _count_independent(wild_type)[0])

    ordered_families = tuple(sorted(f for f in families if f != UNASSIGNED_FAMILY))
    if UNASSIGNED_FAMILY in families:
        ordered_families = ordered_families + (UNASSIGNED_FAMILY,)
    return EvidenceMatrix(cells=cells, families=ordered_families,
                          chemotypes=CHEMOTYPE_COLUMNS,
                          independence_method=method)


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------

class RetrieveEvidence(ScientificInterface):
    """Plan the search, run it against the cache, and publish the gaps with it.

    Returns ``PARTIAL`` whenever a planned query missed, a record was rejected,
    or nothing was retrieved. ``SUCCESS`` means every planned query resolved and
    every record in those payloads was typed without loss -- which, offline with
    an unseeded cache, it correctly never is.
    """

    name: ClassVar[str] = "retrieve_evidence"
    description: ClassVar[str] = (
        "Build an auditable query plan, extract typed experiment records from "
        "the cached sources, and publish the family x chemotype evidence matrix."
    )
    #: Empty for the same reason as in ``normalize_reaction``: in mode B the
    #: substrate is legitimately unknown, and this step still has to produce the
    #: plan and the matrix skeleton that make the gap visible.
    required_fields: ClassVar[tuple[str, ...]] = ()
    required_approvals: ClassVar[tuple[str, ...]] = ()
    depends_on: ClassVar[tuple[str, ...]] = ("normalize_reaction",)
    version: ClassVar[str] = "0.1.0"

    def execute(
        self, ctx: RunContext, *,
        connectors: Mapping[str, Connector] | None = None,
        families: Sequence[str] = (),
        substrate_synonyms: Sequence[str] = (),
        engineering_keywords: Sequence[str] = DEFAULT_ENGINEERING_KEYWORDS,
        chemotype_assignments: Mapping[str, str] | None = None,
        cache_root: str | Path | None = None,
        snapshots: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        task = ctx.task
        result = ToolResult(status=Status.PARTIAL,
                            provenance=Provenance(tool=self.name,
                                                  tool_version=self.version))

        fams = tuple(families) or _families_from_context(ctx, None)
        assignments = dict(chemotype_assignments
                           or ctx.config.get("chemotype_assignments") or {})

        plan = build_query_plan(
            task, families=fams, substrate_synonyms=substrate_synonyms,
            engineering_keywords=engineering_keywords)
        plan_artifact = self._write_plan(ctx, plan)
        result.artifacts.append(plan_artifact)

        if not plan.queries:
            return self._nothing_to_search(result, plan, ctx)

        conns = self._connectors(ctx, connectors, cache_root, snapshots, plan)

        rows: list[EvidenceRow] = []
        failures: list[ExtractionFailure] = []
        responses: list[CachedResponse] = []
        gaps: list[dict[str, Any]] = []

        for pq in plan.queries:
            conn = conns.get(pq.connector)
            if conn is None:
                gaps.append({"query_id": pq.query_id, "connector": pq.connector,
                             "status": "no_connector",
                             "needed": [f"a connector instance for "
                                        f"'{pq.connector}'"]})
                continue
            response = conn.search(pq.query)
            responses.append(response)
            if not response.ok:
                gaps.append({"query_id": pq.query_id, "connector": pq.connector,
                             "status": response.status.value,
                             "reason": response.miss_reason,
                             "cache_path": response.cache_path,
                             "needed": list(response.needed)})
                continue
            got, bad = extract_records(pq.connector, response,
                                       chemotype_assignments=assignments)
            rows.extend(got)
            failures.extend(bad)
            if not got and not bad:
                gaps.append({
                    "query_id": pq.query_id, "connector": pq.connector,
                    "status": "payload_without_records",
                    "reason": "the cached payload carries no 'records' list, so "
                              "it is reference data rather than evidence",
                    "cache_path": response.cache_path, "needed": [],
                })

        # The target reaction is passed so the matrix can check each
        # record's declared direction against its own chemistry, rather
        # than trusting the label a source happened to carry.
        matrix = build_evidence_matrix(rows, ctx.task.reaction)
        records_artifact = self._write_records(ctx, rows)
        matrix_artifact = self._write_matrix(ctx, matrix, plan, rows, gaps)
        gaps_artifact = self._write_gaps(ctx, gaps, failures)
        result.artifacts.extend([records_artifact, matrix_artifact,
                                 gaps_artifact])

        self._flag_findings(result, plan, rows, failures, gaps, responses)

        databases = {r.connector: r.version_for_provenance for r in responses}
        result.provenance = Provenance(
            tool=self.name, tool_version=self.version,
            inputs_sha256={
                "task_spec": sha256_obj(task.model_dump(mode="json")),
                "query_plan": plan.plan_sha256,
            },
            databases=databases,
            models={},
            parameters={
                "n_queries": len(plan.queries),
                "n_skipped_connectors": len(plan.skipped),
                "candidate_families": list(plan.candidate_families),
                "substrate_synonyms": list(plan.substrate_synonyms),
                "engineering_keywords": list(plan.engineering_keywords),
                "min_independent_sources_for_mature":
                    MIN_INDEPENDENT_SOURCES_FOR_MATURE,
                "independence_method": matrix.independence_method,
                "chemotype_assignment_entries": len(assignments),
            },
            random_seed=ctx.seed_for(self.name),
            finished_at=utc_now(),
        )
        result.data.update({
            "n_records": len(rows),
            "n_extraction_failures": len(failures),
            "gaps": gaps,
            "matrix": matrix.to_dict(),
            "plan_sha256": plan.plan_sha256,
            "unpinned_sources": sorted(
                {r.connector for r in responses
                 if r.ok and not r.database_version}),
        })

        if rows and not gaps and not failures:
            result.status = Status.SUCCESS
            result.message = (f"{len(rows)} record(s) retrieved from "
                              f"{len(databases)} source(s); every planned query "
                              f"resolved")
        else:
            result.status = Status.PARTIAL
            result.message = (
                f"{len(rows)} record(s) retrieved; {len(gaps)} query gap(s) and "
                f"{len(failures)} rejected payload record(s). Gaps are missing "
                f"data, not negative results.")
        return result

    # -- helpers -----------------------------------------------------------
    def _nothing_to_search(self, result: ToolResult, plan: QueryPlan,
                           ctx: RunContext) -> ToolResult:
        """No query could be built: fail, rather than search for a placeholder."""
        result.status = Status.FAILED
        result.message = (
            "no query could be planned: the reaction spec supplies no substrate "
            "name or synonym, no EC hint, no Rhea id and no candidate families")
        result.add_flag("no_searchable_terms", Severity.BLOCKER,
                        result.message, subject="reaction")
        result.add_uncertainty(
            "no_searchable_terms",
            "Which substrate name, EC number, Rhea id or candidate family "
            "should the evidence search be anchored on?",
            affects=[self.name], resolvable_by="operator input")
        result.add_next(
            "normalize_reaction",
            "Resolve the reaction spec far enough to anchor a search; a search "
            "built from placeholders returns nothing and looks like a search "
            "that found nothing",
            {"skipped_connectors": [s.to_dict() for s in plan.skipped]},
            requires_human=True)
        result.provenance = Provenance(
            tool=self.name, tool_version=self.version,
            inputs_sha256={"task_spec": sha256_obj(
                ctx.task.model_dump(mode="json"))},
            parameters={"n_queries": 0}, random_seed=ctx.seed_for(self.name))
        return result

    def _connectors(self, ctx: RunContext,
                    connectors: Mapping[str, Connector] | None,
                    cache_root: str | Path | None,
                    snapshots: Mapping[str, str] | None,
                    plan: QueryPlan) -> dict[str, Connector]:
        """Use the caller's connectors, or build offline ones from the registry.

        The default is deliberately the offline wiring: an interface must not be
        able to acquire a live network client by importing a different module,
        and ``ctx.policy.allow_network`` is the only thing that can change that.
        """
        if connectors is not None:
            return dict(connectors)
        from ..connectors import offline_connectors

        root = (cache_root if cache_root is not None
                else ctx.config.get("connector_cache_root"))
        cache = FileCache(root) if root is not None else FileCache()
        access = AccessPolicy.from_execution_policy(ctx.policy)
        keys = sorted({q.connector for q in plan.queries})
        return dict(offline_connectors(
            cache=cache, access=access,
            snapshots=dict(snapshots or ctx.config.get("database_snapshots") or {}),
            keys=keys))

    def _flag_findings(self, result: ToolResult, plan: QueryPlan,
                       rows: Sequence[EvidenceRow],
                       failures: Sequence[ExtractionFailure],
                       gaps: Sequence[Mapping[str, Any]],
                       responses: Sequence[CachedResponse]) -> None:
        """Turn what happened into flags a controller can route on."""
        if not rows:
            result.add_flag(
                "no_evidence_retrieved", Severity.BLOCKER,
                "no experiment record was retrieved. This is an empty evidence "
                "base, not a negative result: nothing downstream may treat it "
                "as evidence that no enzyme performs this reaction.",
                subject="evidence")
            result.add_uncertainty(
                "evidence_base_empty",
                "Which curated imports must be placed in the connector cache "
                "before this task has an evidence base?",
                affects=[self.name, "mine_sequences", "rank_candidates"],
                resolvable_by="a curator populating the cache paths listed in "
                              "evidence_gaps.tsv")

        if gaps:
            result.add_flag(
                "evidence_gaps", Severity.WARN,
                f"{len(gaps)} of {len(plan.queries)} planned queries returned no "
                f"payload; see evidence_gaps.tsv for the exact cache files "
                f"needed. Absence here is absence of data, never of activity.",
                subject="evidence")
            result.add_next(
                "populate_connector_cache",
                "A curator imports the named records so the run can be replayed "
                "offline with the gaps closed",
                {"gaps": [dict(g) for g in gaps]}, requires_human=True)

        if plan.skipped:
            result.add_flag(
                "layers_not_searched", Severity.INFO,
                "not searched: " + ", ".join(
                    f"{s.connector} ({s.reason})" for s in plan.skipped),
                subject="query_plan")

        if failures:
            result.add_flag(
                "records_rejected", Severity.WARN,
                f"{len(failures)} payload record(s) were rejected rather than "
                f"repaired; the commonest causes are a positive without a "
                f"product-identifying method and a negative without a detection "
                f"limit", subject="evidence")

        capped = [d for row in rows for d in row.strength_decisions
                  if d.get("capped")]
        if capped:
            result.add_flag(
                "evidence_strength_capped", Severity.INFO,
                f"{len(capped)} evidence reference(s) claimed more than their "
                f"source may assert and were capped at the source's ceiling; a "
                f"named human reviewer is required to promote them",
                subject="evidence")

        unpinned = sorted({r.connector for r in responses
                           if r.ok and not r.database_version})
        if unpinned:
            result.add_flag(
                "database_version_unknown", Severity.WARN,
                "no release string is recorded for: " + ", ".join(unpinned)
                + "; the run is not reproducible against a pinned snapshot",
                subject="provenance")

        mismatched = [row.record.record_id for row in rows
                      if row.record.cofactor is not None
                      and row.record.cofactor.state.value == "unknown"]
        if mismatched:
            result.add_flag(
                "record_cofactor_state_unknown", Severity.WARN,
                f"{len(mismatched)} record(s) carry a cofactor whose oxidation "
                f"state is unknown; NAD(P)+ and NAD(P)H records must not be "
                f"pooled", subject="evidence")

        unclassified = sum(1 for r in rows
                           if not r.chemotype.chemotype.is_decided)
        if unclassified:
            result.add_flag(
                "chemotype_unassigned", Severity.WARN,
                f"{unclassified} record(s) could not be placed on the chemotype "
                f"axis and sit in the 'unclassified' column; no structure "
                f"perception was attempted, because a substring guess would "
                f"mislabel the matrix", subject="evidence_matrix")
            result.add_uncertainty(
                "chemotype_assignment",
                "Which chemotype does each unclassified substrate belong to?",
                affects=["evidence_matrix"],
                resolvable_by="an operator-supplied chemotype assignment table "
                              "keyed on InChIKey")

    # -- artifacts ---------------------------------------------------------
    def _write_plan(self, ctx: RunContext, plan: QueryPlan) -> Artifact:
        path = ctx.path("evidence_query_plan.yaml")
        header = (
            "# evidence_query_plan.yaml -- written BEFORE retrieval.\n"
            "#\n"
            "# Recall cannot be judged from results, only from queries. Read\n"
            "# 'queries' for what was asked, 'skipped_connectors' for the layers\n"
            "# that were not searched and why, and 'recall_caveats' for the known\n"
            "# limits of this search before trusting anything downstream of it.\n"
        )
        document = dict(plan.to_dict())
        document["plan_sha256"] = plan.plan_sha256
        document["sub_space_reference"] = [s.to_dict()
                                           for s in SUBSTRATE_CLASS_SCAFFOLD]
        path.write_text(header + yaml.safe_dump(document, sort_keys=False,
                                                allow_unicode=True),
                        encoding="utf-8")
        return Artifact(key="evidence_query_plan", path=str(path), kind="file",
                        sha256=sha256_file(path), n_records=len(plan.queries),
                        summary=f"{len(plan.queries)} planned quer(ies), "
                                f"{len(plan.skipped)} connector(s) not searched")

    def _write_records(self, ctx: RunContext,
                       rows: Sequence[EvidenceRow]) -> Artifact:
        path = ctx.path("evidence_records.jsonl")
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False,
                                    default=str) + "\n")
        return Artifact(key="evidence_records", path=str(path), kind="table",
                        sha256=sha256_file(path), n_records=len(rows),
                        summary=f"{len(rows)} typed experiment record(s) with "
                                f"capped evidence strength")

    def _write_matrix(self, ctx: RunContext, matrix: EvidenceMatrix,
                      plan: QueryPlan, rows: Sequence[EvidenceRow],
                      gaps: Sequence[Mapping[str, Any]]) -> Artifact:
        path = ctx.path("evidence_matrix.tsv")
        extra = [
            f"Task: {plan.task_id}; reaction class "
            f"{plan.target_reaction.get('reaction_class')}",
            f"Built from {len(rows)} record(s) over {len(plan.queries)} planned "
            f"quer(ies), of which {len(gaps)} returned no payload.",
            f"Family rows come from the records themselves; "
            f"'{UNASSIGNED_FAMILY}' means no family was recorded.",
            "Chemotype columns are assigned only from a curated declaration or "
            "an operator table; nothing is perceived from structure.",
        ]
        path.write_text(matrix.to_tsv(extra_header=extra), encoding="utf-8")
        return Artifact(key="evidence_matrix", path=str(path), kind="table",
                        sha256=sha256_file(path),
                        n_records=len(matrix.families) * len(matrix.chemotypes),
                        summary=f"{len(matrix.families)} family row(s) x "
                                f"{len(matrix.chemotypes)} chemotype column(s)")

    def _write_gaps(self, ctx: RunContext, gaps: Sequence[Mapping[str, Any]],
                    failures: Sequence[ExtractionFailure]) -> Artifact:
        """List exactly what a curator must supply, with the cache path for each.

        A gap report that says "BRENDA returned nothing" is not actionable. One
        that names the cache file and the query it must answer can be worked
        through in an afternoon, which is the difference between a run that gets
        repaired and one that gets abandoned.
        """
        path = ctx.path("evidence_gaps.tsv")
        lines = [
            "# What is missing from the evidence base, and what would close it.",
            "# kind=query_gap: a planned query returned no payload. This is "
            "missing data, not a negative result.",
            "# kind=record_rejected: a cached record could not be typed; it was "
            "not repaired, because the missing field is a measurement.",
            "\t".join(["kind", "connector", "subject", "reason", "needed"]),
        ]
        for g in gaps:
            lines.append("\t".join(_tsv_safe(v) for v in [
                "query_gap", g.get("connector", ""),
                g.get("query_id", "") or g.get("cache_path", ""),
                g.get("reason") or g.get("status", ""),
                "; ".join(str(n) for n in (g.get("needed") or [])),
            ]))
        for f in failures:
            lines.append("\t".join(_tsv_safe(v) for v in [
                "record_rejected", f.connector, f.raw_id, f.reason, "",
            ]))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return Artifact(key="evidence_gaps", path=str(path), kind="table",
                        sha256=sha256_file(path),
                        n_records=len(gaps) + len(failures),
                        summary=f"{len(gaps)} query gap(s), "
                                f"{len(failures)} rejected record(s)")
