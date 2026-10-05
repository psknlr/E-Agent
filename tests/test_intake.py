"""Tests for staged evidence intake.

Every case here is a way a dataset quietly stops meaning anything:

* a pipeline promotes its own output, so "verified" becomes a label a script
  applied;
* an EC-number-plus-species mapping acquires a sequence-level experimental
  label, so a family annotation becomes a measurement;
* ``n.d.`` is read as "no activity", so a column of negatives appears that
  nobody measured;
* a negative is stored without the limit it is negative at, so it excludes
  nothing and is still counted;
* an alcohol oxidation record enters a reduction seed set as a positive.

In each case the broken row looks exactly like a good one once it is written,
which is why the refusal has to happen at intake.
"""

from __future__ import annotations

import unittest

from eagent.datalayer.intake import (
    MODEL_INFERENCE_IS_NOT_EVIDENCE,
    NO_AUTOMATIC_PROMOTION,
    TIER_ORDER,
    AuditEntry,
    EvidenceTier,
    ExtractionMethod,
    IntakeError,
    IntakeRecord,
    IntakeStore,
    ModelInferenceNotEvidenceError,
    ReviewState,
    ReviewerRequiredError,
    SourceCeilingError,
    TierMergeRefusedError,
    TierPromotionError,
    coverage,
    direction_check,
    ingest,
    normalise_outcome,
    promote,
    reverse_class_of,
)
from eagent.datalayer.layers import DataLayer
from eagent.datalayer.registry import (
    AccessMode,
    DataSource,
    SourceRegistry,
)
from eagent.schemas.chem import SubstrateSpec
from eagent.schemas.reaction import Conditions, ReactionClass, ReactionSpec
from eagent.schemas.record import (
    Detection,
    EvidenceRef,
    EvidenceStrength,
    ExperimentRecord,
    OutcomeClass,
    ReactionDirection,
)

REVIEWER = "Jordan Keeler"
SEQ = "MKVQAWYTGSDLNPFRTTVSHQ"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _source(sid: str, ceiling: EvidenceStrength) -> DataSource:
    return DataSource(
        id=sid,
        display_name=f"test source {sid}",
        layers=[DataLayer.ENZYMOLOGY_EVIDENCE],
        good_for=["test fixture"],
        not_good_for=["anything real"],
        access_modes=[AccessMode.UNKNOWN],
        evidence_strength_ceiling=ceiling,
        needs_curation=False,
    )


def _registry() -> SourceRegistry:
    """A tiny registry: one EC-level source, one sequence-level source."""
    return SourceRegistry([
        _source("ec_level_db", EvidenceStrength.EC_SPECIES_MAPPED),
        _source("primary_db", EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL),
        _source("annotation_db", EvidenceStrength.ANNOTATION_ONLY),
    ])


def _record(
    record_id: str = "rec1",
    *,
    outcome: OutcomeClass = OutcomeClass.NOT_TESTED,
    detection: Detection | None = None,
    direction: ReactionDirection = ReactionDirection.UNSPECIFIED,
    reaction_class: ReactionClass = ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
    evidence: list[EvidenceRef] | None = None,
    sequence: str | None = SEQ,
    substrate: SubstrateSpec | None = None,
    conditions: Conditions | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        record_id=record_id,
        sequence=sequence,
        substrate=substrate or SubstrateSpec(name="acetophenone"),
        reaction_class=reaction_class,
        reaction_direction=direction,
        outcome=outcome,
        detection=detection or Detection(),
        conditions=conditions or Conditions(),
        evidence=evidence if evidence is not None else [
            EvidenceRef(source_type="database", identifier="ec_level_db:1.1.1.1",
                        strength=EvidenceStrength.EC_SPECIES_MAPPED)],
    )


def _ec_species_intake(registry: SourceRegistry) -> IntakeRecord:
    """The dangerous row: an EC-number-plus-species mapping from a database."""
    return ingest(
        _record("ec_row"),
        tier=EvidenceTier.CURATED_DATABASE,
        source_id="ec_level_db",
        registry=registry,
        extraction_method=ExtractionMethod.DATABASE_EXPORT,
    )


# ---------------------------------------------------------------------------
# no automatic promotion
# ---------------------------------------------------------------------------

class TestNoAutomaticPromotion(unittest.TestCase):
    """Nothing raises a record's tier except a person saying so."""

    def setUp(self) -> None:
        self.reg = _registry()

    def test_module_exposes_no_automatic_promotion_path(self) -> None:
        import eagent.datalayer.intake as intake

        for name in dir(intake):
            lowered = name.lower()
            if "promote" in lowered:
                self.assertEqual(
                    name, "promote",
                    f"'{name}' looks like a second promotion path; promote() "
                    f"must be the only one")
        self.assertFalse(hasattr(IntakeRecord, "promote"))
        self.assertIn("named human reviewer", NO_AUTOMATIC_PROMOTION)

    def test_ingest_defaults_to_the_floor_not_the_ceiling(self) -> None:
        """A row that states no strength gets the weakest, not the strongest allowed.

        Defaulting to the source's ceiling stamps every bulk row at the best
        claim the source could ever support, which is a guess upward from a row
        that said nothing at all.
        """
        rec = ingest(_record(), tier=EvidenceTier.MACHINE_EXTRACTED_PENDING,
                     source_id="primary_db", registry=self.reg,
                     extraction_method=ExtractionMethod.MACHINE_EXTRACTION_LLM)
        self.assertIs(rec.tier, EvidenceTier.MACHINE_EXTRACTED_PENDING)
        self.assertIs(rec.review_state, ReviewState.NOT_REVIEWED)
        self.assertIs(rec.claimed_strength, EvidenceStrength.ANNOTATION_ONLY)
        self.assertFalse(rec.is_usable_as_label)

    def test_an_explicit_claim_above_the_ceiling_is_refused(self) -> None:
        """Claiming past the ceiling raises rather than being quietly capped.

        A silent cap would let a caller ask for sequence-level experimental
        evidence and receive something weaker without noticing; the refusal
        makes the disagreement visible at the call site.
        """
        from eagent.datalayer.intake import SourceCeilingError
        with self.assertRaises(SourceCeilingError):
            ingest(_record(), tier=EvidenceTier.MACHINE_EXTRACTED_PENDING,
                   source_id="primary_db", registry=self.reg,
                   extraction_method=ExtractionMethod.MACHINE_EXTRACTION_LLM,
                   claimed_strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL)

    def test_machine_extraction_cannot_be_ingested_above_its_tier(self) -> None:
        with self.assertRaises(TierPromotionError) as ctx:
            ingest(_record(), tier=EvidenceTier.CURATED_DATABASE,
                   source_id="primary_db", registry=self.reg,
                   extraction_method=ExtractionMethod.MACHINE_EXTRACTION_LLM)
        self.assertIn("promote it", str(ctx.exception))

    def test_promotion_without_a_reviewer_is_refused(self) -> None:
        rec = _ec_species_intake(self.reg)
        for bad in ("", "   ", None):
            with self.assertRaises(ReviewerRequiredError):
                promote(rec, bad, "read the original paper carefully",
                        registry=self.reg,
                        to_tier=EvidenceTier.EXPERT_VERIFIED_PRIMARY)

    def test_promotion_signed_by_software_is_refused(self) -> None:
        rec = _ec_species_intake(self.reg)
        for actor in ("auto-ingest", "model", "claude", "curation pipeline",
                      "system", "llm-extractor"):
            with self.assertRaises(ReviewerRequiredError) as ctx:
                promote(rec, actor, "high confidence score from the extractor",
                        registry=self.reg,
                        to_tier=EvidenceTier.EXPERT_VERIFIED_PRIMARY)
            self.assertIn("automated", str(ctx.exception).lower())

    def test_promotion_needs_a_substantive_justification(self) -> None:
        rec = _ec_species_intake(self.reg)
        with self.assertRaises(TierPromotionError):
            promote(rec, REVIEWER, "ok", registry=self.reg,
                    to_tier=EvidenceTier.EXPERT_VERIFIED_PRIMARY)

    def test_promotion_needs_a_target(self) -> None:
        rec = _ec_species_intake(self.reg)
        with self.assertRaises(TierPromotionError) as ctx:
            promote(rec, REVIEWER, "checked against the supplementary table",
                    registry=self.reg)
        self.assertIn("target", str(ctx.exception))

    def test_promotion_only_moves_upward(self) -> None:
        rec = _ec_species_intake(self.reg)
        with self.assertRaises(TierPromotionError) as ctx:
            promote(rec, REVIEWER, "downgrading this row after a second look",
                    registry=self.reg,
                    to_tier=EvidenceTier.MACHINE_EXTRACTED_PENDING)
        self.assertIn("not a promotion", str(ctx.exception))

    def test_a_model_inference_is_never_promoted(self) -> None:
        rec = ingest(
            _record("pred1", outcome=OutcomeClass.COMPUTATIONAL_NEGATIVE),
            tier=EvidenceTier.MODEL_INFERRED, source_id="annotation_db",
            registry=self.reg, extraction_method=ExtractionMethod.MODEL_PREDICTION)
        with self.assertRaises(ModelInferenceNotEvidenceError) as ctx:
            promote(rec, REVIEWER,
                    "a senior scientist reviewed the prediction and agrees",
                    registry=self.reg, to_tier=EvidenceTier.CURATED_DATABASE)
        self.assertIn("not a measurement", str(ctx.exception))
        self.assertIn("hypothesis", MODEL_INFERENCE_IS_NOT_EVIDENCE)

    def test_a_successful_promotion_writes_an_audit_entry(self) -> None:
        rec = _ec_species_intake(self.reg)
        out = promote(rec, REVIEWER,
                      "read the original paper and matched the construct",
                      registry=self.reg,
                      to_tier=EvidenceTier.EXPERT_VERIFIED_PRIMARY)
        self.assertIs(out.tier, EvidenceTier.EXPERT_VERIFIED_PRIMARY)
        self.assertIs(out.review_state, ReviewState.REVIEWED_ACCEPTED)
        self.assertEqual(out.reviewer, REVIEWER)
        self.assertEqual(len(out.audit), 2)
        last: AuditEntry = out.audit[-1]
        self.assertEqual(last.action, "promote")
        self.assertIs(last.from_tier, EvidenceTier.CURATED_DATABASE)
        self.assertIs(last.to_tier, EvidenceTier.EXPERT_VERIFIED_PRIMARY)
        self.assertEqual(last.reviewer, REVIEWER)
        self.assertFalse(last.ceiling_exceeded)
        # The original is untouched: a refusal cannot half-promote a record.
        self.assertIs(rec.tier, EvidenceTier.CURATED_DATABASE)
        self.assertEqual(len(rec.audit), 1)


# ---------------------------------------------------------------------------
# the EC-plus-species to sequence-level jump
# ---------------------------------------------------------------------------

class TestSourceCeiling(unittest.TestCase):
    """An EC-number-plus-species mapping is not a sequence-level label."""

    def setUp(self) -> None:
        self.reg = _registry()
        self.rec = _ec_species_intake(self.reg)

    def test_ingest_refuses_a_claim_above_the_source_ceiling(self) -> None:
        with self.assertRaises(SourceCeilingError) as ctx:
            ingest(_record(), tier=EvidenceTier.CURATED_DATABASE,
                   source_id="ec_level_db", registry=self.reg,
                   claimed_strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL)
        self.assertIn("ec_species_mapped", str(ctx.exception))

    def test_sequence_level_promotion_without_a_reviewer_is_refused(self) -> None:
        with self.assertRaises(ReviewerRequiredError):
            promote(self.rec, "", "the accession matched",
                    registry=self.reg,
                    to_strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL)

    def test_sequence_level_promotion_needs_the_primary_evidence(self) -> None:
        """A reviewer re-reading the same database row is not new evidence."""
        with self.assertRaises(SourceCeilingError) as ctx:
            promote(self.rec, REVIEWER,
                    "the EC number and species look right to me",
                    registry=self.reg,
                    to_strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL)
        msg = str(ctx.exception)
        self.assertIn("evidence_strength_ceiling=ec_species_mapped", msg)
        self.assertIn("supporting_evidence", msg)

    def test_supporting_evidence_must_itself_be_strong_enough(self) -> None:
        weak = EvidenceRef(source_type="publication", identifier="PMID:0000001",
                           strength=EvidenceStrength.HOMOLOG_EXPERIMENTAL,
                           extracted_by="human", verified_by=REVIEWER)
        with self.assertRaises(SourceCeilingError) as ctx:
            promote(self.rec, REVIEWER, "attached the homolog paper as support",
                    registry=self.reg,
                    to_strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
                    supporting_evidence=weak)
        self.assertIn("itself only", str(ctx.exception))

    def test_supporting_evidence_must_name_its_verifier(self) -> None:
        unverified = EvidenceRef(
            source_type="publication", identifier="PMID:0000002",
            strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
            extracted_by="human")
        with self.assertRaises(SourceCeilingError) as ctx:
            promote(self.rec, REVIEWER, "attached the primary paper as support",
                    registry=self.reg,
                    to_strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
                    supporting_evidence=unverified)
        self.assertIn("no verifier", str(ctx.exception))

    def test_ceiling_may_be_exceeded_only_loudly(self) -> None:
        primary = EvidenceRef(
            source_type="publication", identifier="PMID:0000003",
            locator="Table S3, entry 7",
            strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
            extracted_by="human", verified_by=REVIEWER)
        out = promote(
            self.rec, REVIEWER,
            "read Table S3 and confirmed the construct sequence matches",
            registry=self.reg,
            to_tier=EvidenceTier.EXPERT_VERIFIED_PRIMARY,
            to_strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
            supporting_evidence=primary)
        self.assertIs(out.claimed_strength,
                      EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL)
        self.assertTrue(out.audit[-1].ceiling_exceeded)
        self.assertEqual(out.audit[-1].supporting_evidence, "PMID:0000003")
        self.assertIn("PMID:0000003",
                      [e.identifier for e in out.record.evidence])
        self.assertTrue(any("above the registered ceiling" in u
                            for u in out.uncertainties))

    def test_promotion_refuses_an_unregistered_source(self) -> None:
        rec = IntakeRecord(
            intake_id="orphan", record=_record("orphan"),
            tier=EvidenceTier.CURATED_DATABASE, source_id="not_registered")
        with self.assertRaises(SourceCeilingError) as ctx:
            promote(rec, REVIEWER, "checked the record against the paper",
                    registry=self.reg,
                    to_tier=EvidenceTier.EXPERT_VERIFIED_PRIMARY)
        self.assertIn("not registered", str(ctx.exception))


# ---------------------------------------------------------------------------
# tier invariants and storage
# ---------------------------------------------------------------------------

class TestTierInvariants(unittest.TestCase):

    def setUp(self) -> None:
        self.reg = _registry()

    def test_expert_tier_requires_a_reviewer(self) -> None:
        with self.assertRaises(IntakeError):
            IntakeRecord(intake_id="x", record=_record(),
                         tier=EvidenceTier.EXPERT_VERIFIED_PRIMARY,
                         source_id="primary_db",
                         review_state=ReviewState.REVIEWED_ACCEPTED)

    def test_model_tier_may_not_carry_an_experimental_outcome(self) -> None:
        det = Detection(method="chiral HPLC", confirms_product_identity=True)
        with self.assertRaises(IntakeError) as ctx:
            IntakeRecord(
                intake_id="x",
                record=_record(outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                               detection=det),
                tier=EvidenceTier.MODEL_INFERRED, source_id="annotation_db")
        self.assertIn("model_inferred", str(ctx.exception))

    def test_sequence_level_claim_always_names_a_reviewer(self) -> None:
        with self.assertRaises(IntakeError):
            IntakeRecord(
                intake_id="x", record=_record(),
                tier=EvidenceTier.CURATED_DATABASE, source_id="primary_db",
                claimed_strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL)

    def test_store_keeps_tiers_in_separate_partitions(self) -> None:
        store = IntakeStore()
        store.add(ingest(_record("a"), tier=EvidenceTier.CURATED_DATABASE,
                         source_id="ec_level_db", registry=self.reg,
                         extraction_method=ExtractionMethod.DATABASE_EXPORT))
        store.add(ingest(_record("b", outcome=OutcomeClass.COMPUTATIONAL_NEGATIVE),
                         tier=EvidenceTier.MODEL_INFERRED,
                         source_id="annotation_db", registry=self.reg,
                         extraction_method=ExtractionMethod.MODEL_PREDICTION))
        parts = store.partitions()
        self.assertEqual(len(parts), len(TIER_ORDER))
        self.assertEqual(len(parts["tier/curated_database"]), 1)
        self.assertEqual(len(parts["tier/model_inferred"]), 1)
        self.assertEqual(len(parts["tier/expert_verified_primary"]), 0)

    def test_pooling_model_inferences_with_evidence_is_refused(self) -> None:
        store = IntakeStore([
            ingest(_record("a"), tier=EvidenceTier.CURATED_DATABASE,
                   source_id="ec_level_db", registry=self.reg,
                   extraction_method=ExtractionMethod.DATABASE_EXPORT)])
        with self.assertRaises(TierMergeRefusedError):
            store.pool([EvidenceTier.CURATED_DATABASE,
                        EvidenceTier.MODEL_INFERRED],
                       justification="one table for the ranker")
        with self.assertRaises(TierMergeRefusedError):
            store.pool([], justification="everything")
        with self.assertRaises(TierMergeRefusedError):
            store.pool([EvidenceTier.CURATED_DATABASE], justification="  ")
        pooled = store.pool([EvidenceTier.CURATED_DATABASE],
                            justification="seed triage only")
        self.assertEqual(len(pooled), 1)

    def test_store_refiles_a_promoted_record(self) -> None:
        store = IntakeStore()
        rec = store.add(_ec_species_intake(self.reg))
        out = promote(rec, REVIEWER, "read the paper and matched the construct",
                      registry=self.reg,
                      to_tier=EvidenceTier.EXPERT_VERIFIED_PRIMARY)
        store.replace(out)
        self.assertEqual(len(store.of_tier(EvidenceTier.CURATED_DATABASE)), 0)
        self.assertEqual(
            len(store.of_tier(EvidenceTier.EXPERT_VERIFIED_PRIMARY)), 1)
        self.assertEqual(len(store), 1)


# ---------------------------------------------------------------------------
# outcome normalisation
# ---------------------------------------------------------------------------

class TestNormaliseOutcome(unittest.TestCase):
    """An ambiguous statement becomes not_tested, never a negative."""

    def test_nd_is_ambiguous_and_never_a_negative(self) -> None:
        n = normalise_outcome("n.d.")
        self.assertIs(n.outcome, OutcomeClass.NOT_TESTED)
        self.assertIsNone(n.proposed_outcome)
        self.assertFalse(n.confident)
        self.assertFalse(n.is_negative)
        self.assertTrue(any("not determined" in u for u in n.uncertainties))

    def test_other_ambiguous_tokens_do_not_become_negatives(self) -> None:
        for raw in ("trace", "n/a", "no data", "racemic", "weak", "low",
                    "not reported", "negative", "yes"):
            n = normalise_outcome(raw)
            self.assertIs(n.outcome, OutcomeClass.NOT_TESTED, raw)
            self.assertFalse(n.confident, raw)
            self.assertTrue(n.uncertainties, raw)

    def test_an_unrecognised_statement_is_not_guessed_at(self) -> None:
        n = normalise_outcome("behaved as expected for this scaffold")
        self.assertIs(n.outcome, OutcomeClass.NOT_TESTED)
        self.assertIsNone(n.proposed_outcome)
        self.assertTrue(any("no rule matched" in u for u in n.uncertainties))

    def test_a_missing_statement_is_not_an_absence_of_activity(self) -> None:
        for raw in (None, "", "   "):
            n = normalise_outcome(raw)
            self.assertIs(n.outcome, OutcomeClass.NOT_TESTED)
            self.assertTrue(n.uncertainties)

    def test_conflicting_statements_resolve_to_not_tested(self) -> None:
        n = normalise_outcome("no product detected; inclusion bodies")
        self.assertIs(n.outcome, OutcomeClass.NOT_TESTED)
        self.assertIsNone(n.proposed_outcome)
        self.assertTrue(any("more than one outcome class" in u
                            for u in n.uncertainties))

    def test_an_ambiguous_token_poisons_an_otherwise_clear_statement(self) -> None:
        n = normalise_outcome("no activity (n.d.)")
        self.assertIs(n.outcome, OutcomeClass.NOT_TESTED)
        self.assertIs(n.proposed_outcome, OutcomeClass.NO_TARGET_PRODUCT_DETECTED)
        self.assertFalse(n.is_storable)

    def test_negation_is_read_before_the_positive_inside_it(self) -> None:
        n = normalise_outcome("not active")
        self.assertIs(n.proposed_outcome, OutcomeClass.NO_TARGET_PRODUCT_DETECTED)

    def test_explicit_not_tested_is_confident(self) -> None:
        n = normalise_outcome("not tested")
        self.assertIs(n.outcome, OutcomeClass.NOT_TESTED)
        self.assertTrue(n.confident)
        self.assertTrue(n.is_storable)

    def test_expression_and_computational_failures_are_distinct(self) -> None:
        self.assertIs(normalise_outcome("inclusion bodies").outcome,
                      OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE)
        self.assertIs(normalise_outcome("docking failed").outcome,
                      OutcomeClass.COMPUTATIONAL_FAILURE)
        self.assertIs(normalise_outcome("predicted inactive").outcome,
                      OutcomeClass.COMPUTATIONAL_NEGATIVE)
        self.assertIs(normalise_outcome("wrong enantiomer").outcome,
                      OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION)


class TestNegativesNeedALimit(unittest.TestCase):
    """A negative without a detection limit is not interpretable."""

    def test_negative_without_a_detection_limit_is_blocked(self) -> None:
        n = normalise_outcome("no product detected")
        self.assertIs(n.proposed_outcome, OutcomeClass.NO_TARGET_PRODUCT_DETECTED)
        self.assertIs(n.outcome, OutcomeClass.NOT_TESTED)
        self.assertFalse(n.is_storable)
        self.assertIn("detection limit", n.blocked_reason or "")

    def test_negative_with_a_limit_but_no_unit_is_blocked(self) -> None:
        n = normalise_outcome("no product detected",
                              detection=Detection(method="GC-MS",
                                                  limit_of_detection=0.5))
        self.assertIs(n.outcome, OutcomeClass.NOT_TESTED)
        self.assertFalse(n.is_storable)
        self.assertIn("unit", n.blocked_reason or "")

    def test_negative_with_a_full_detection_is_accepted(self) -> None:
        n = normalise_outcome(
            "no product detected",
            detection=Detection(method="chiral GC-MS", limit_of_detection=0.5,
                                limit_unit="uM"))
        self.assertIs(n.outcome, OutcomeClass.NO_TARGET_PRODUCT_DETECTED)
        self.assertTrue(n.is_storable)
        self.assertTrue(n.confident)

    def test_the_record_model_also_refuses_a_limitless_negative(self) -> None:
        """Defence in depth: the schema refuses it too."""
        with self.assertRaises(Exception):
            _record(outcome=OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    detection=Detection(method="GC-MS"))

    def test_positive_without_product_identification_is_blocked(self) -> None:
        n = normalise_outcome("activity detected",
                              detection=Detection(method="NADPH A340"))
        self.assertIs(n.proposed_outcome, OutcomeClass.CONFIRMED_TARGET_PRODUCT)
        self.assertIs(n.outcome, OutcomeClass.NOT_TESTED)
        self.assertIn("identifies the product", n.blocked_reason or "")

    def test_positive_with_product_identification_is_accepted(self) -> None:
        n = normalise_outcome(
            "product confirmed",
            detection=Detection(method="chiral HPLC",
                                confirms_product_identity=True))
        self.assertIs(n.outcome, OutcomeClass.CONFIRMED_TARGET_PRODUCT)
        self.assertTrue(n.is_storable)


# ---------------------------------------------------------------------------
# direction
# ---------------------------------------------------------------------------

class TestDirectionCheck(unittest.TestCase):
    """An oxidation measurement is not reduction evidence."""

    target = ReactionSpec(reaction_class=ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)

    def test_reverse_class_pairs_are_symmetric(self) -> None:
        self.assertIn(ReactionClass.ALCOHOL_OXIDATION,
                      reverse_class_of(ReactionClass.KETONE_TO_SECONDARY_ALCOHOL))
        self.assertIn(ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
                      reverse_class_of(ReactionClass.ALCOHOL_OXIDATION))

    def test_declared_reverse_direction_is_non_supporting(self) -> None:
        rec = _record(direction=ReactionDirection.REVERSE_OF_TARGET)
        v = direction_check(rec, self.target)
        self.assertFalse(v.supports)
        self.assertTrue(v.non_supporting)
        self.assertTrue(v.is_reverse)
        self.assertTrue(any("reverse_of_target" in r for r in v.reasons))

    def test_an_oxidation_record_is_flagged_even_when_it_claims_forward(self) -> None:
        rec = _record(reaction_class=ReactionClass.ALCOHOL_OXIDATION,
                      direction=ReactionDirection.FORWARD_AS_TARGET)
        v = direction_check(rec, self.target)
        self.assertFalse(v.supports)
        self.assertTrue(v.is_reverse)
        self.assertTrue(any("chemical reverse" in r for r in v.reasons))
        self.assertTrue(any("contradict" in r for r in v.reasons))

    def test_unspecified_direction_is_not_counted_as_forward(self) -> None:
        v = direction_check(_record(), self.target)
        self.assertFalse(v.supports)
        self.assertTrue(v.is_unspecified)
        self.assertFalse(v.is_reverse)

    def test_forward_and_reversible_records_support(self) -> None:
        for d in (ReactionDirection.FORWARD_AS_TARGET,
                  ReactionDirection.REVERSIBLE_BOTH_SHOWN):
            v = direction_check(_record(direction=d), self.target)
            self.assertTrue(v.supports, d.value)

    def test_direction_check_accepts_an_intake_wrapper(self) -> None:
        reg = _registry()
        item = ingest(_record(direction=ReactionDirection.REVERSE_OF_TARGET),
                      tier=EvidenceTier.CURATED_DATABASE,
                      source_id="ec_level_db", registry=reg,
                      extraction_method=ExtractionMethod.DATABASE_EXPORT)
        self.assertFalse(direction_check(item, self.target).supports)

    def test_a_missing_target_class_is_reported_not_assumed(self) -> None:
        v = direction_check(_record(direction=ReactionDirection.FORWARD_AS_TARGET),
                            None)
        self.assertIsNone(v.target_class)
        self.assertTrue(any("could not be read" in r for r in v.reasons))


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

class TestCoverage(unittest.TestCase):
    """Evidence or rows: the intersection is the number that matters."""

    def setUp(self) -> None:
        self.reg = _registry()

    def _full_record(self, rid: str) -> ExperimentRecord:
        return ExperimentRecord(
            record_id=rid,
            sequence=SEQ,
            construct_sequence=SEQ,
            substrate=SubstrateSpec(name="acetophenone",
                                    isomeric_smiles="CC(=O)c1ccccc1",
                                    inchikey="KWOLFJPFCHCOCG-UHFFFAOYSA-N"),
            reaction_class=ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
            reaction_direction=ReactionDirection.FORWARD_AS_TARGET,
            outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
            detection=Detection(method="chiral GC-MS",
                                confirms_product_identity=True,
                                authentic_standard=True),
            conditions=Conditions(pH=7.0, temperature_C=30.0),
            evidence=[EvidenceRef(source_type="publication",
                                  identifier="PMID:0000009",
                                  strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
                                  extracted_by="human", verified_by=REVIEWER)],
        )

    def test_counts_per_tier_and_the_fully_specified_intersection(self) -> None:
        good = ingest(self._full_record("good"),
                      tier=EvidenceTier.EXPERT_VERIFIED_PRIMARY,
                      source_id="primary_db", registry=self.reg,
                      reviewer=REVIEWER,
                      claimed_strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
                      extraction_method=ExtractionMethod.HUMAN_READING_PRIMARY)
        thin = ingest(_record("thin"), tier=EvidenceTier.CURATED_DATABASE,
                      source_id="ec_level_db", registry=self.reg,
                      extraction_method=ExtractionMethod.DATABASE_EXPORT)
        cov = coverage([good, thin],
                       target_reaction=ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)

        self.assertEqual(cov.n_records, 2)
        self.assertEqual(cov.by_tier[EvidenceTier.EXPERT_VERIFIED_PRIMARY], 1)
        self.assertEqual(cov.by_tier[EvidenceTier.CURATED_DATABASE], 1)
        self.assertEqual(cov.n_fully_specified, 1)
        self.assertEqual(cov.fully_specified_ids[0], good.intake_id)
        self.assertTrue(cov.has_evidence)

        missing = cov.missing_by_record[thin.intake_id]
        for name in ("substrate_structure", "reaction_direction", "conditions",
                     "detection", "informative_outcome"):
            self.assertIn(name, missing)
        self.assertIn(thin.intake_id, cov.direction_non_supporting_ids)
        self.assertNotIn(good.intake_id, cov.direction_non_supporting_ids)

    def test_a_pile_of_thin_rows_is_reported_as_rows_not_evidence(self) -> None:
        rows = [ingest(_record(f"r{i}"), tier=EvidenceTier.CURATED_DATABASE,
                       source_id="ec_level_db", registry=self.reg,
                       extraction_method=ExtractionMethod.DATABASE_EXPORT)
                for i in range(20)]
        cov = coverage(rows)
        self.assertEqual(cov.n_records, 20)
        self.assertEqual(cov.n_fully_specified, 0)
        self.assertFalse(cov.has_evidence)
        self.assertIn("rows, not evidence", cov.describe())

    def test_unknown_extraction_method_sets_a_curation_flag(self) -> None:
        rec = ingest(_record(), tier=EvidenceTier.CURATED_DATABASE,
                     source_id="ec_level_db", registry=self.reg)
        self.assertTrue(rec.needs_curation)
        self.assertTrue(any("extraction method not recorded" in u
                            for u in rec.uncertainties))
        self.assertIn(rec.intake_id, coverage([rec]).needs_curation_ids)


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
