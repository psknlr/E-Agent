"""Enzymology-evidence connectors: BRENDA, OED, SABIO-RK, EnzEngDB,
RetroBioCat-DB and STRENDA DB.

Why this module exists
----------------------
This is the layer that answers "somebody measured something". It is also the
layer where the measurements are re-published, so the three guards here are
about counting, not about retrieval.

*One measurement counted four times.* OED re-integrates BRENDA and SABIO-RK;
other collections re-integrate OED. Four hits on one re-curated row look like
four-fold support to anything that counts rows. Every record this module emits
carries :attr:`~eagent.schemas.record.EvidenceRef.upstream_sources`, populated
from the registry's transitive lineage, so
:mod:`eagent.datalayer.lineage` can collapse them back into one independent
measurement.

*An oxidation counted as a reduction.* Curated kinetic rows arrive with the
same EC number and the same substrate name whichever way the assay ran.
Direction is read from the record and defaults to
:attr:`~eagent.schemas.record.ReactionDirection.UNSPECIFIED`, never to forward,
and :func:`~eagent.datalayer.intake.direction_check` decides whether a row may
count.

*A web database dressed up as an API.* OED, RetroBioCat-DB, STRENDA DB and
ProtaBank are registered with no unattended route. They get
:class:`CuratedFileImporter` subclasses whose ``fetch`` exists only to refuse,
so no author can "fix" a missing client by inventing one. ProtaBank is a
performance resource rather than a kinetics one, but it is the fourth source
under the same access policy and has no other module in this package, so its
importer lives here beside the others rather than nowhere.

Nothing here promotes evidence. Every imported row enters through
:func:`eagent.datalayer.intake.ingest` with no claimed strength, which stamps
the weakest rung and leaves the registry ceiling as a cap that only a named
human reviewer can approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..datalayer.intake import (
    EvidenceTier,
    ExtractionMethod,
    IntakeRecord,
    direction_check,
    ingest,
    normalise_outcome,
)
from ..datalayer.registry import CapabilityState, SourceRegistry
from ..schemas.chem import SubstrateSpec
from ..schemas.reaction import Conditions, ReactionClass
from ..schemas.record import (
    EvidenceRef,
    EvidenceStrength,
    ExperimentRecord,
    OutcomeClass,
    ReactionDirection,
)
from .base import CachedResponse, ConnectorLayer
from .chemistry import (
    CuratedFileImporter,
    CuratedImportError,
    LayerSemanticsError,
    RegistryBackedConnector,
    evidence_ref_for,
    upstream_sources_for,
)

__all__ = [
    "BRENDAConnector",
    "EnzEngDBConnector",
    "EngineeringCampaignRecord",
    "KineticMeasurement",
    "MeasurementNotSequenceLevelError",
    "OEDImporter",
    "ProtaBankImporter",
    "RetroBioCatDBImporter",
    "SABIORKConnector",
    "STRENDADBImporter",
    "parse_reaction_direction",
]


class MeasurementNotSequenceLevelError(LayerSemanticsError):
    """A row keyed to an EC number and an organism was read as one sequence.

    This is the single step that converts "some enzyme of this class was
    reported active" into "this protein is active". It is refused here rather
    than capped, because a caller that believes it is holding sequence-level
    data has a bug that a silent downgrade would hide.
    """


def parse_reaction_direction(raw: Any) -> ReactionDirection:
    """Read a declared direction, defaulting to ``UNSPECIFIED``.

    Never defaults to forward. An unrecorded direction is not a forward one,
    and a curated kinetic table that omits it is the usual way an alcohol
    oxidation enters a reduction seed set as a positive.
    """
    if isinstance(raw, ReactionDirection):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return ReactionDirection.UNSPECIFIED
    token = raw.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return ReactionDirection(token)
    except ValueError:
        pass
    aliases = {
        "forward": ReactionDirection.FORWARD_AS_TARGET,
        "target": ReactionDirection.FORWARD_AS_TARGET,
        "reverse": ReactionDirection.REVERSE_OF_TARGET,
        "backward": ReactionDirection.REVERSE_OF_TARGET,
        "reversible": ReactionDirection.REVERSIBLE_BOTH_SHOWN,
        "both": ReactionDirection.REVERSIBLE_BOTH_SHOWN,
    }
    return aliases.get(token, ReactionDirection.UNSPECIFIED)


# ---------------------------------------------------------------------------
# the shape every kinetics resource returns
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KineticMeasurement:
    """One reported measurement, with everything needed to refuse to pool it.

    ``measurement_type`` and ``unit`` are kept because a kcat, a specific
    activity and a percentage conversion are three different quantities;
    averaging them produces a number with no units and no meaning.

    ``sequence_resolved`` is the flag that keeps this layer honest. Most curated
    rows are keyed to an EC number and an organism, not to a sequence, and
    :meth:`require_sequence_level` refuses to pretend otherwise.
    """

    source_id: str
    record_id: str
    ec_number: str | None
    organism: str | None
    uniprot_accession: str | None
    substrate_name: str | None
    substrate_smiles: str | None
    measurement_type: str | None
    value: float | None
    unit: str | None
    reaction_direction: ReactionDirection
    reaction_class: ReactionClass | None
    conditions: Conditions
    publication_id: str | None
    upstream_sources: tuple[str, ...]
    evidence: EvidenceRef
    caveats: tuple[str, ...] = ()

    @property
    def sequence_resolved(self) -> bool:
        """Whether this row names the exact protein that was measured."""
        return bool(self.uniprot_accession)

    @property
    def publication_evidence(self) -> EvidenceRef | None:
        """A pointer to the primary report, or ``None`` when the row has none.

        Emitted as a second, publication-typed
        :class:`~eagent.schemas.record.EvidenceRef` so that
        :mod:`eagent.datalayer.lineage` can see that a BRENDA row and a
        SABIO-RK row citing one paper are one measurement. Without it both
        rows look unlinkable and are counted separately, which is the
        four-databases-one-measurement error this layer exists to prevent.
        """
        if not self.publication_id:
            return None
        return EvidenceRef(
            source_type="publication",
            identifier=self.publication_id,
            strength=EvidenceStrength.ANNOTATION_ONLY,
            extracted_by="connector",
            upstream_sources=list(self.upstream_sources),
        )

    def require_sequence_level(self) -> str:
        """The accession, or a refusal. See :class:`MeasurementNotSequenceLevelError`."""
        if not self.uniprot_accession:
            raise MeasurementNotSequenceLevelError(
                f"{self.source_id}:{self.record_id} is keyed to "
                f"ec={self.ec_number or 'unstated'} / "
                f"organism={self.organism or 'unstated'} and names no sequence. "
                f"It may not be ingested as sequence-level evidence without a "
                f"per-record resolution step performed by a person.")
        return self.uniprot_accession

    def supports_direction(self, target: Any) -> bool:
        """Whether this row may count as support for the target direction."""
        return bool(direction_check(self.to_experiment_record(), target).supports)

    def to_experiment_record(self) -> ExperimentRecord:
        """A record carrying only what the row states.

        The outcome stays ``NOT_TESTED`` unless the row reports one. A curated
        kinetic constant is not an outcome class: inferring
        ``confirmed_target_product`` from the presence of a kcat would create a
        positive nobody measured for this substrate under these conditions.
        """
        return ExperimentRecord(
            record_id=f"{self.source_id}:{self.record_id}",
            accession=self.uniprot_accession,
            substrate=SubstrateSpec(name=self.substrate_name,
                                    isomeric_smiles=self.substrate_smiles),
            reaction_class=self.reaction_class or ReactionClass.OTHER,
            reaction_direction=self.reaction_direction,
            conditions=self.conditions,
            outcome=OutcomeClass.NOT_TESTED,
            measurement_type=self.measurement_type,
            measurement_value=self.value,
            measurement_unit=self.unit,
            evidence=[e for e in (self.evidence, self.publication_evidence)
                      if e is not None],
            notes="; ".join(self.caveats),
        )

    def to_intake_record(self, registry: SourceRegistry, *,
                         at: str | None = None) -> IntakeRecord:
        """Admit the row through intake at the weakest defensible strength.

        ``claimed_strength`` is left unset on purpose: intake then stamps the
        floor and records that the connector asserted nothing, while the
        registry ceiling stays a cap that only
        :func:`~eagent.datalayer.intake.promote` can approach.
        """
        return ingest(
            self.to_experiment_record(),
            tier=EvidenceTier.CURATED_DATABASE,
            source_id=self.source_id,
            registry=registry,
            extraction_method=ExtractionMethod.DATABASE_EXPORT,
            claimed_strength=None,
            uncertainties=self.caveats,
            at=at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id, "record_id": self.record_id,
            "ec_number": self.ec_number, "organism": self.organism,
            "uniprot_accession": self.uniprot_accession,
            "sequence_resolved": self.sequence_resolved,
            "substrate_name": self.substrate_name,
            "substrate_smiles": self.substrate_smiles,
            "measurement_type": self.measurement_type,
            "value": self.value, "unit": self.unit,
            "reaction_direction": self.reaction_direction.value,
            "publication_id": self.publication_id,
            "upstream_sources": list(self.upstream_sources),
            "caveats": list(self.caveats),
        }


class _KineticsConnector(RegistryBackedConnector):
    """Shared reader for the kinetics resources with a programmatic route.

    Both BRENDA and SABIO-RK publish rows of the same logical shape, and both
    need the same three caveats attached. Writing the caveats once means a
    change to the de-duplication rule cannot leave one of them behind.

    Reads the payload shape::

        {"records": [{"record_id": "...", "ec": "1.1.1.1",
                      "organism": "...", "uniprot_accession": "...",
                      "substrate": "...", "substrate_smiles": "...",
                      "measurement_type": "kcat", "value": 1.0, "unit": "s^-1",
                      "direction": "...", "reaction_class": "...",
                      "pH": 7.0, "temperature_C": 30.0, "buffer": "...",
                      "pubmed_id": "..."}]}
    """

    data_layer = ConnectorLayer.KINETICS

    def measurements(self, query: Mapping[str, Any]) -> tuple[KineticMeasurement, ...]:
        """Cached measurements for a structured query, with caveats attached."""
        response = self.guarded("search", query, "keyword_query")
        return self._measurements_from(response)

    def measurements_for_ec(self, ec_number: str, *, substrate: str | None = None
                            ) -> tuple[KineticMeasurement, ...]:
        """Cached measurements for one EC number, optionally one substrate."""
        query: dict[str, Any] = {"ec": ec_number}
        if substrate:
            query["substrate"] = substrate
        return self.measurements(query)

    def _measurements_from(self, response: CachedResponse
                           ) -> tuple[KineticMeasurement, ...]:
        upstream = upstream_sources_for(self.source_id, self.registry)
        out: list[KineticMeasurement] = []
        for index, row in enumerate(self.records(response), start=1):
            record_id = _text(row.get("record_id")) or f"row{index}"
            accession = _text(row.get("uniprot_accession"))
            substrate_name = _text(row.get("substrate"))
            substrate_smiles = _text(row.get("substrate_smiles"))
            direction = parse_reaction_direction(row.get("direction"))
            caveats = self._caveats(row, accession, substrate_name,
                                    substrate_smiles, direction, upstream)
            out.append(KineticMeasurement(
                source_id=self.source_id,
                record_id=record_id,
                ec_number=_text(row.get("ec")),
                organism=_text(row.get("organism")),
                uniprot_accession=accession,
                substrate_name=substrate_name,
                substrate_smiles=substrate_smiles,
                measurement_type=_text(row.get("measurement_type")),
                value=_float(row.get("value")),
                unit=_text(row.get("unit")),
                reaction_direction=direction,
                reaction_class=_reaction_class(row.get("reaction_class")),
                conditions=Conditions(
                    pH=_float(row.get("pH")),
                    temperature_C=_float(row.get("temperature_C")),
                    buffer=_text(row.get("buffer")),
                    solvent_system=_text(row.get("solvent_system")),
                ),
                publication_id=_text(row.get("pubmed_id")) or _text(row.get("doi")),
                upstream_sources=upstream,
                evidence=evidence_ref_for(
                    self.source, record_id, registry=self.registry,
                    locator=_text(row.get("locator")),
                    database_version=response.database_version,
                    retrieved_at=response.retrieved_at),
                caveats=caveats))
        return tuple(out)

    def _caveats(self, row: Mapping[str, Any], accession: str | None,
                 substrate_name: str | None, substrate_smiles: str | None,
                 direction: ReactionDirection,
                 upstream: Sequence[str]) -> tuple[str, ...]:
        """The caveats every kinetics row carries, each naming a real failure."""
        notes: list[str] = [
            f"re-curated lineage: {', '.join(upstream)}; rows sharing an upstream "
            f"are one piece of evidence, not several",
        ]
        if not accession:
            notes.append(
                "keyed to an EC number and an organism rather than a sequence; "
                "it may not be used as sequence-level evidence without a "
                "per-record resolution step")
        if substrate_name and not substrate_smiles:
            notes.append(
                f"the substrate is given as the prose name '{substrate_name}' "
                f"with no structure; for an asymmetric reduction a name without "
                f"stereochemistry does not identify the material assayed")
        if direction is ReactionDirection.UNSPECIFIED:
            notes.append(
                "no reaction direction is recorded; an oxidation measurement and "
                "a reduction measurement are indistinguishable here, so this row "
                "cannot be counted as support until a curator reads the assay")
        if row.get("measurement_type") and not row.get("unit"):
            notes.append(
                "a measured value is present with no unit; values without units "
                "must not be placed on a common scale")
        if not self.source.derived_from_complete:
            notes.append(
                f"'{self.source_id}' declares an incomplete lineage, so it may "
                f"not be counted as independent corroboration of anything")
        return tuple(notes)


class BRENDAConnector(_KineticsConnector):
    """BRENDA rows, treated as pointers into the literature.

    The registry caps this source at ``ec_species_mapped``, which is the whole
    point: a BRENDA row is usually a statement about an EC number in an
    organism, and reading it as a statement about one protein is how a family
    annotation becomes a measurement.

    ``exact_record_fetch`` is registered ``unknown`` for this source, so a
    ``fetch`` here returns a result marked
    :data:`~eagent.connectors.chemistry.UNVERIFIED_CAPABILITY` rather than one
    that looks like a verified retrieval.
    """

    source_id = "brenda"
    description = "BRENDA: curated enzyme functional data"


class SABIORKConnector(_KineticsConnector):
    """SABIO-RK rows, preferred when the question is comparability.

    Keeps pH, temperature and buffer with the value, because two kinetic
    constants measured under different conditions are not two measurements of
    one quantity. A row arriving without them is flagged, not completed: a
    guessed pH would make two incomparable numbers look comparable.
    """

    source_id = "sabio_rk"
    description = "SABIO-RK: kinetic parameters with experimental context"

    def _caveats(self, row: Mapping[str, Any], accession: str | None,
                 substrate_name: str | None, substrate_smiles: str | None,
                 direction: ReactionDirection,
                 upstream: Sequence[str]) -> tuple[str, ...]:
        notes = list(super()._caveats(row, accession, substrate_name,
                                      substrate_smiles, direction, upstream))
        missing = [name for name, key in
                   (("pH", "pH"), ("temperature", "temperature_C"),
                    ("buffer", "buffer"))
                   if row.get(key) in (None, "")]
        if missing:
            notes.append(
                f"the row states no {', '.join(missing)}; the condition set is "
                f"incomplete, so this value may not be compared with another "
                f"value from a different assay")
        notes.append(
            "coverage of this resource is narrow: the absence of an entry is not "
            "the absence of activity")
        return tuple(notes)


# ---------------------------------------------------------------------------
# EnzEngDB: engineering campaigns, no programmatic route established
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineeringCampaignRecord:
    """A reported mutation and what it was reported to change.

    ``parent_accession`` and ``mutations`` are kept apart from any performance
    number, because a campaign reports an effect relative to its own parent
    under its own assay. Transferring the number to another parent is an
    inference, and one that looks exactly like a measurement once written into
    a table.
    """

    source_id: str
    record_id: str
    parent_accession: str | None
    mutations: tuple[str, ...]
    reported_effect: str | None
    measurement_type: str | None
    value: float | None
    unit: str | None
    publication_id: str | None
    upstream_sources: tuple[str, ...]
    evidence: EvidenceRef
    caveats: tuple[str, ...] = ()

    def to_experiment_record(self) -> ExperimentRecord:
        """A variant record, refusing to exist without its parent.

        :class:`~eagent.schemas.record.ExperimentRecord` requires a variant to
        name its parent sequence hash. A campaign row that does not identify its
        parent therefore cannot become a variant record at all, which is correct:
        a mutation without a parent is not a construct.
        """
        if self.mutations and not self.parent_accession:
            raise LayerSemanticsError(
                f"{self.source_id}:{self.record_id} lists mutations "
                f"{list(self.mutations)} but names no parent; a mutation "
                f"relative to an unnamed parent identifies no construct")
        return ExperimentRecord(
            record_id=f"{self.source_id}:{self.record_id}",
            accession=self.parent_accession,
            mutations=list(self.mutations),
            outcome=OutcomeClass.NOT_TESTED,
            measurement_type=self.measurement_type,
            measurement_value=self.value,
            measurement_unit=self.unit,
            evidence=[self.evidence],
            notes="; ".join(self.caveats),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "record_id": self.record_id,
                "parent_accession": self.parent_accession,
                "mutations": list(self.mutations),
                "reported_effect": self.reported_effect,
                "measurement_type": self.measurement_type,
                "value": self.value, "unit": self.unit,
                "publication_id": self.publication_id,
                "upstream_sources": list(self.upstream_sources),
                "caveats": list(self.caveats)}


class EnzEngDBConnector(RegistryBackedConnector):
    """Reported enzyme-engineering campaigns.

    Registered with ``bulk_download`` and ``offline_import`` only, so there is
    no per-record service to call: every capability flag is ``unknown`` and
    every operation therefore comes back marked
    :data:`~eagent.connectors.chemistry.UNVERIFIED_CAPABILITY`. It stays a
    connector rather than an importer because a bulk archive can be pinned and
    replayed from the cache without a person in the loop for each record, which
    is not true of the browse-only resources in this module.

    Reads the payload shape::

        {"records": [{"record_id": "...", "parent_accession": "...",
                      "mutations": ["A123V"], "effect": "...",
                      "measurement_type": "...", "value": 1.0, "unit": "...",
                      "pubmed_id": "..."}]}
    """

    source_id = "enzengdb"
    data_layer = ConnectorLayer.KINETICS
    description = "EnzEngDB: reported enzyme engineering campaigns"

    def campaigns_for(self, query: Mapping[str, Any]
                      ) -> tuple[EngineeringCampaignRecord, ...]:
        """Cached campaign rows for a structured query."""
        response = self.guarded("search", query, "keyword_query")
        upstream = upstream_sources_for(self.source_id, self.registry)
        out: list[EngineeringCampaignRecord] = []
        for index, row in enumerate(self.records(response), start=1):
            record_id = _text(row.get("record_id")) or f"row{index}"
            mutations = tuple(str(m).strip() for m in (row.get("mutations") or [])
                              if str(m).strip())
            caveats = [
                "a reported effect is relative to this campaign's own parent "
                "and assay; it does not transfer to another parent",
                f"re-curated lineage: {', '.join(upstream)}",
            ]
            if self.source.capabilities.version_information \
                    is not CapabilityState.SUPPORTED:
                caveats.append(
                    "no release identifier is recordable for this source, so "
                    "the row cannot be pinned to a version")
            out.append(EngineeringCampaignRecord(
                source_id=self.source_id, record_id=record_id,
                parent_accession=_text(row.get("parent_accession")),
                mutations=mutations,
                reported_effect=_text(row.get("effect")),
                measurement_type=_text(row.get("measurement_type")),
                value=_float(row.get("value")),
                unit=_text(row.get("unit")),
                publication_id=_text(row.get("pubmed_id")),
                upstream_sources=upstream,
                evidence=evidence_ref_for(
                    self.source, record_id, registry=self.registry,
                    database_version=response.database_version,
                    retrieved_at=response.retrieved_at),
                caveats=tuple(caveats)))
        return tuple(out)


# ---------------------------------------------------------------------------
# importers: resources with no unattended route at all
# ---------------------------------------------------------------------------

class OEDImporter(CuratedFileImporter):
    """Open Enzyme Database export, imported by a person.

    The registry records this source's access mode as ``unknown`` on purpose:
    it is described as offering programmatic integration, but whether that is a
    hosted API, an installable package or a download has not been established
    here. Guessing it as a REST endpoint is the failure this importer exists to
    make impossible -- there is no client to guess with.

    It also re-integrates BRENDA and SABIO-RK with an admittedly incomplete
    upstream list, so every imported row carries the lineage and a warning that
    it may not be counted as independent corroboration.
    """

    source_id = "oed"
    required_fields = ("record_id", "ec")
    description = "OED: re-integrated enzyme kinetics, imported by hand"

    def _build_record(self, row: Mapping[str, Any], row_number: int
                      ) -> ExperimentRecord:
        accession = _text(row.get("uniprot_accession"))
        return ExperimentRecord(
            record_id=f"oed:{_text(row.get('record_id'))}",
            accession=accession,
            substrate=SubstrateSpec(name=_text(row.get("substrate")),
                                    isomeric_smiles=_text(row.get("substrate_smiles"))),
            reaction_direction=parse_reaction_direction(row.get("direction")),
            reaction_class=_reaction_class(row.get("reaction_class"))
            or ReactionClass.OTHER,
            conditions=Conditions(pH=_float(row.get("pH")),
                                  temperature_C=_float(row.get("temperature_C")),
                                  buffer=_text(row.get("buffer"))),
            outcome=OutcomeClass.NOT_TESTED,
            measurement_type=_text(row.get("measurement_type")),
            measurement_value=_float(row.get("value")),
            measurement_unit=_text(row.get("unit")),
            evidence=[evidence_ref_for(
                self.source, str(_text(row.get("record_id"))),
                registry=self.registry,
                locator=_text(row.get("locator")))],
        )

    def _row_uncertainties(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        notes = [
            f"re-integrated from "
            f"{', '.join(upstream_sources_for(self.source_id, self.registry))}; "
            f"an agreeing row in an upstream resource is the same measurement, "
            f"not a second one",
        ]
        if not _text(row.get("upstream_record_id")):
            notes.append(
                "the row carries no upstream record identifier, so it cannot be "
                "de-duplicated against BRENDA or SABIO-RK; until it can, this "
                "row must be treated as possibly already counted")
        if not _text(row.get("uniprot_accession")):
            notes.append(
                "no accession: the row is EC-and-organism level and is not "
                "evidence about any one sequence")
        return tuple(notes)


class RetroBioCatDBImporter(CuratedFileImporter):
    """RetroBioCat specificity data, imported by a person.

    The registry note is explicit that installing the published package yields
    only the bundled example data, and that registering it as a local package
    "would mark this source programmatically reachable and suppress the warning
    that a person must obtain and check the real records first". That warning is
    this class.
    """

    source_id = "retrobiocat_db"
    required_fields = ("record_id", "enzyme_name", "substrate")
    description = "RetroBioCat database: biocatalysis specificity data, imported"

    def _build_record(self, row: Mapping[str, Any], row_number: int
                      ) -> ExperimentRecord:
        outcome, detection, note = _outcome_and_detection(row)
        return ExperimentRecord(
            record_id=f"retrobiocat:{_text(row.get('record_id'))}",
            accession=_text(row.get("uniprot_accession")),
            substrate=SubstrateSpec(
                name=_text(row.get("substrate")),
                isomeric_smiles=_text(row.get("substrate_smiles"))),
            reaction_class=_reaction_class(row.get("reaction_class"))
            or ReactionClass.OTHER,
            reaction_direction=parse_reaction_direction(row.get("direction")),
            outcome=outcome,
            detection=detection,
            evidence=[evidence_ref_for(
                self.source, str(_text(row.get("record_id"))),
                registry=self.registry)],
            notes=note,
        )

    def _row_uncertainties(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        notes = [
            "obtained through the web interface or an author archive; a person "
            "checked this row, and which person is recorded on the import",
        ]
        if not _text(row.get("substrate_smiles")):
            notes.append(
                "the substrate is a prose name with no structure, so the "
                "stereochemistry of the material assayed is not determined by "
                "this row")
        return tuple(notes)


class ProtaBankImporter(CuratedFileImporter):
    """ProtaBank variant-performance records, imported by a person.

    The registry records this source's access mode as ``unknown`` deliberately:
    whether the resource is currently reachable, and by what route, has not
    been established here. Writing a client against a remembered URL is the
    failure this class makes impossible -- there is no client.

    Its ceiling is ``homolog_experimental``, which is a cap and not a default:
    rows still enter at the floor, because an imported spreadsheet row is not a
    measurement somebody here has read.
    """

    source_id = "protabank"
    required_fields = ("record_id",)
    description = "ProtaBank: protein engineering datasets, imported by hand"

    def _build_record(self, row: Mapping[str, Any], row_number: int
                      ) -> ExperimentRecord:
        mutations = [m.strip() for m in
                     str(row.get("mutations", "")).replace(";", ",").split(",")
                     if m.strip()]
        parent = _text(row.get("parent_accession")) or _text(
            row.get("uniprot_accession"))
        if mutations and not parent:
            raise CuratedImportError(
                "the row lists mutations but names no parent accession; a "
                "mutation relative to an unnamed parent identifies no construct")
        outcome, detection, note = _outcome_and_detection(row)
        return ExperimentRecord(
            record_id=f"protabank:{_text(row.get('record_id'))}",
            accession=parent,
            mutations=mutations,
            outcome=outcome,
            detection=detection,
            measurement_type=_text(row.get("measurement_type")),
            measurement_value=_float(row.get("value")),
            measurement_unit=_text(row.get("unit")),
            evidence=[evidence_ref_for(
                self.source, str(_text(row.get("record_id"))),
                registry=self.registry)],
            notes=note,
        )

    def _row_uncertainties(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            "a reported effect is relative to this campaign's own parent and "
            "assay and does not transfer to another parent",
            "overlap with the other engineering collections registered here is "
            "unconfirmed, so the same campaign may already have been counted",
        )


class STRENDADBImporter(CuratedFileImporter):
    """STRENDA DB records, imported by a person, capped per row.

    The registry gives this source a ``sequence_level_experimental`` ceiling,
    with a curation note saying that ceiling holds "only for records that
    actually carry the measured construct's sequence" and that a record lacking
    it "must be downgraded to ec_species_mapped on ingest". That per-row rule is
    implemented in :meth:`_row_ceiling`, so the ceiling cannot be applied to a
    row that does not earn it.

    The rows themselves still enter at the floor: a reporting standard says a
    submission was complete, not that anybody here read it.
    """

    source_id = "strenda_db"
    required_fields = ("record_id",)
    description = "STRENDA DB: standards-compliant enzymology records, imported"

    def _build_record(self, row: Mapping[str, Any], row_number: int
                      ) -> ExperimentRecord:
        sequence = _text(row.get("sequence"))
        return ExperimentRecord(
            record_id=f"strenda:{_text(row.get('record_id'))}",
            sequence=sequence,
            accession=_text(row.get("uniprot_accession")),
            construct_sequence=sequence,
            construct_description=_text(row.get("construct_description")),
            substrate=SubstrateSpec(
                name=_text(row.get("substrate")),
                isomeric_smiles=_text(row.get("substrate_smiles"))),
            reaction_direction=parse_reaction_direction(row.get("direction")),
            conditions=Conditions(pH=_float(row.get("pH")),
                                  temperature_C=_float(row.get("temperature_C")),
                                  buffer=_text(row.get("buffer"))),
            outcome=OutcomeClass.NOT_TESTED,
            measurement_type=_text(row.get("measurement_type")),
            measurement_value=_float(row.get("value")),
            measurement_unit=_text(row.get("unit")),
            evidence=[evidence_ref_for(
                self.source, str(_text(row.get("record_id"))),
                registry=self.registry)],
        )

    def _row_ceiling(self, row: Mapping[str, Any]) -> EvidenceStrength:
        """Sequence-level only for a row that carries the measured sequence."""
        if _text(row.get("sequence")):
            return EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL
        return EvidenceStrength.EC_SPECIES_MAPPED

    def _row_ceiling_reason(self, row: Mapping[str, Any]) -> str:
        return ("the row carries no construct sequence, so the registry's "
                "sequence_level_experimental ceiling does not apply to it; the "
                "measurement cannot be bound to a protein nobody recorded")

    def _row_uncertainties(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        if _text(row.get("sequence")):
            return (
                "the row carries the measured construct's sequence, so a named "
                "reviewer could later promote it; nothing automatic may",
            )
        return (
            "no construct sequence on the row: the reporting standard was met "
            "for the submission, not for binding this measurement to a protein",
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _outcome_and_detection(row: Mapping[str, Any]) -> tuple[Any, Any, str]:
    """Read an outcome through intake, with the detection block it needs.

    The detection block is built first and handed to
    :func:`~eagent.datalayer.intake.normalise_outcome`, because that function
    already encodes the two rules this module must not restate: a positive
    needs a method that identifies the product, and a negative needs the limit
    it is negative at. Passing the detection in means an unsupportable outcome
    degrades to ``not_tested`` with a recorded reason, rather than being written
    and then rejected by the record model as if the export were malformed.

    ``n.d.`` therefore resolves to ``not_tested``, never to a negative: it means
    "not detected" in one paper and "not determined" in the next, and a column
    of guessed negatives is indistinguishable from a column of measured ones.
    """
    from ..schemas.record import Detection

    detection = Detection(
        method=_text(row.get("detection_method")),
        limit_of_detection=_float(row.get("limit_of_detection")),
        limit_unit=_text(row.get("limit_unit")),
        confirms_product_identity=str(
            row.get("confirms_product_identity", "")).strip().lower()
        in ("1", "true", "yes", "y"),
    )
    raw = _text(row.get("outcome")) or _text(row.get("activity"))
    if not raw:
        return (OutcomeClass.NOT_TESTED, detection,
                "no outcome stated on the imported row")
    normalisation = normalise_outcome(raw, detection=detection)
    note = f"outcome normalised from {raw!r} to {normalisation.outcome.value}"
    if normalisation.blocked_reason:
        note += f"; the stated outcome was not storable: {normalisation.blocked_reason}"
    return normalisation.outcome, detection, note


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reaction_class(value: Any) -> ReactionClass | None:
    if value is None:
        return None
    if isinstance(value, ReactionClass):
        return value
    try:
        return ReactionClass(str(value).strip())
    except ValueError:
        return None
