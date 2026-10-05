"""Tests for :mod:`eagent.science.scorecard`.

The cases are chosen around the failures the module exists to prevent rather
than its happy path: a missing measurement treated as a bad one, an undecided
gate treated as a failed one, weak evidence treated as a negative result, a
docking score ranked in the wrong direction, and the weighted total that would
make all of the above invisible.
"""

from __future__ import annotations

import unittest

from eagent.errors import FabricationGuardError, TemplateError
from eagent.schemas import (
    Candidate,
    CatalyticMapping,
    CatalyticTemplate,
    CofactorSpec,
    CofactorState,
    ComplexPose,
    ConfidenceLevel,
    EvidenceRef,
    EvidenceStrength,
    FamilyAnnotation,
    GeometryConstraint,
    GeometryReport,
    LigandSource,
    ProductSpec,
    SCORE_DIMENSIONS,
    ScoreDimension,
    SequenceRecord,
    Stereochemistry,
    StructureRecord,
    SubstrateSpec,
    TaskSpec,
    TemplateProvenance,
    TemplateSourceType,
)
from eagent.science.scorecard import (
    DEFAULT_LEXICOGRAPHIC_ORDER,
    GATE_NAMES,
    ErrorKind,
    GateOutcome,
    FeasibilityGate,
    build_scorecard,
    comparable,
    dominates,
    evaluate_feasibility_gates,
    evidence_weaknesses,
    explain,
    gate_cofactor_compatibility,
    gate_input_integrity,
    gate_stereochemistry_resolved,
    gate_uncertainties,
    input_defects,
    lexicographic_rank,
    objective_value,
    pareto_front,
    rank_within_family,
    refuse_linear_blend,
    scorecard_qc_flags,
)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

SEQ = "MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHTDAYTLSGADPEGLFPVILGHEGAGIVESVGEGVT"


def candidate(
    cid: str,
    *,
    sequence: str = SEQ,
    family: str | None = "SDR",
    cluster: str | None = None,
    is_fragment: bool | None = False,
    percent_identity: float | None = 55.0,
    dims: dict[str, ScoreDimension] | None = None,
    gates_pass: bool | None = True,
    input_errors: list[str] | None = None,
) -> Candidate:
    """A candidate carrying only the fields a given test needs.

    ``gates_pass`` writes one synthetic gate so ``passes_gates`` /
    ``has_unresolved_gate`` behave as the test intends without running the real
    gate evaluators; ``None`` writes an UNEVALUATED gate.
    """
    cand = Candidate(
        candidate_id=cid,
        sequence_record=SequenceRecord(
            candidate_id=cid, sequence=sequence, is_fragment=is_fragment,
            percent_identity=percent_identity, search_method="mmseqs2",
            seed_accession="P00000", source_database="uniref90",
        ),
        family=FamilyAnnotation(family_name=family, sequence_cluster_id=cluster),
        input_errors=list(input_errors or []),
    )
    cand.set_dimension(ScoreDimension(
        name="catalytic_machinery_mappable", is_gate=True,
        gate_passed=gates_pass, direction="categorical",
        level=(ConfidenceLevel.STRONG if gates_pass
               else ConfidenceLevel.CONTRADICTORY if gates_pass is False
               else ConfidenceLevel.INSUFFICIENT),
    ))
    for dim in (dims or {}).values():
        cand.set_dimension(dim)
    return cand


def dim(name: str, level: ConfidenceLevel, value: float | None = None,
        direction: str = "higher_is_better") -> ScoreDimension:
    return ScoreDimension(name=name, level=level, value=value, direction=direction)


def catalytic_template(
    *,
    template_id: str = "cat:sdr:kred",
    cofactor: str | None = "NADPH",
    state: CofactorState = CofactorState.REDUCED,
    roles: tuple[str, ...] = ("catalytic_Tyr", "catalytic_Ser", "catalytic_Lys"),
) -> CatalyticTemplate:
    return CatalyticTemplate(
        template_id=template_id,
        family_name="SDR",
        mechanism_summary="Tyr/Ser/Lys triad, hydride from NADPH C4 to the carbonyl",
        catalytic_residues=[{"label": r, "residue_types": ["Y"], "role": r,
                             "functional_atoms": ["OH"], "evidence": "M-CSA"}
                            for r in roles],
        required_cofactor=cofactor,
        required_cofactor_state=state,
        cofactor_ligand_codes=["NDP"],
        geometry_constraints=[GeometryConstraint(
            name="hydride_transfer", kind="distance",
            atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
            min_value=2.5, max_value=3.6, severity="gating",
            calibrated_on=["1XYZ"], source="M-CSA 123",
        )],
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.MECHANISM_LITERATURE,
            identifiers=["M-CSA:123"],
        ),
    )


def task(
    *,
    creates_stereocenter: bool | None = True,
    target: Stereochemistry = Stereochemistry.R,
    cofactors: tuple[CofactorSpec, ...] | None = None,
) -> TaskSpec:
    if cofactors is None:
        cofactors = (CofactorSpec(name="NADPH", state=CofactorState.REDUCED,
                                  ligand_code="NDP"),)
    spec = TaskSpec(task_id="t1")
    spec.reaction.substrate = SubstrateSpec(
        name="acetophenone", isomeric_smiles="CC(=O)c1ccccc1", is_prochiral=True)
    spec.reaction.product = ProductSpec(
        name="(R)-1-phenylethanol", isomeric_smiles="C[C@@H](O)c1ccccc1",
        target_stereochemistry=target,
        creates_new_stereocenter=creates_stereocenter)
    spec.conditions.cofactor_options = list(cofactors)
    return spec


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

class TestFeasibilityGates(unittest.TestCase):

    def test_unevaluated_gate_is_not_a_failure(self) -> None:
        """The whole point of the third state: undecided is not rejected."""
        gate = FeasibilityGate("g", "q?", GateOutcome.UNEVALUATED, basis="not run")
        self.assertFalse(gate.failed)
        self.assertFalse(gate.passed)
        self.assertIsNone(gate.outcome.as_bool)
        self.assertIsNone(gate.to_dimension().gate_passed)

    def test_unevaluated_gate_becomes_an_uncertainty(self) -> None:
        gate = FeasibilityGate("g", "Can it be mapped?", GateOutcome.UNEVALUATED,
                               basis="no mapping attempted", remedy="run the mapper")
        unc = gate.to_uncertainty()
        self.assertIsNotNone(unc)
        assert unc is not None
        self.assertEqual(unc.resolvable_by, "run the mapper")
        self.assertIn("Can it be mapped?", unc.question)

    def test_decided_gates_produce_no_uncertainty(self) -> None:
        for outcome in (GateOutcome.PASS, GateOutcome.FAIL):
            self.assertIsNone(
                FeasibilityGate("g", "q?", outcome).to_uncertainty())

    def test_unevaluated_outranks_failed_when_only_levels_are_read(self) -> None:
        """A reader that only sees levels must still not confuse the two."""
        unevaluated = FeasibilityGate("g", "q?", GateOutcome.UNEVALUATED).to_dimension()
        failed = FeasibilityGate("g", "q?", GateOutcome.FAIL).to_dimension()
        self.assertGreater(unevaluated.level.rank, failed.level.rank)

    def test_missing_catalytic_role_fails_the_machinery_gate(self) -> None:
        cand = candidate("c1")
        cand.catalytic_mapping = CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Ser": "S142"},
            missing_roles=["catalytic_Tyr"],
        )
        gates = {g.name: g for g in evaluate_feasibility_gates(cand, task())}
        machinery = gates["catalytic_machinery_mappable"]
        self.assertTrue(machinery.failed)
        self.assertIs(machinery.kind, ErrorKind.INPUT_DEFECT)

    def test_recorded_substitution_is_accepted_not_failed(self) -> None:
        cand = candidate("c1")
        cand.catalytic_mapping = CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Ser": "S142", "catalytic_Lys": "K159"},
            missing_roles=["catalytic_Tyr"],
            substituted_roles={"catalytic_Tyr": "F155"},
        )
        gates = {g.name: g for g in evaluate_feasibility_gates(cand, task())}
        self.assertTrue(gates["catalytic_machinery_mappable"].passed)

    def test_no_mapping_attempted_is_unevaluated_not_failed(self) -> None:
        cand = candidate("c1")
        gates = {g.name: g for g in evaluate_feasibility_gates(cand, task())}
        self.assertTrue(gates["catalytic_machinery_mappable"].unevaluated)

    def test_wrong_cofactor_oxidation_state_fails(self) -> None:
        """NADP+ where the mechanism needs NADPH is a defect, not a weak score."""
        cand = candidate("c1")
        cand.catalytic_mapping = CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Tyr": "Y155"},
        )
        spec = task(cofactors=(CofactorSpec(name="NADP+",
                                            state=CofactorState.OXIDIZED,
                                            ligand_code="NAP"),))
        gate = gate_cofactor_compatibility(cand, spec,
                                           {"cat:sdr:kred": catalytic_template()})
        self.assertTrue(gate.failed)
        self.assertIs(gate.kind, ErrorKind.INPUT_DEFECT)
        self.assertIn("oxidation state", gate.basis)

    def test_correct_cofactor_and_state_passes(self) -> None:
        cand = candidate("c1")
        cand.catalytic_mapping = CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Tyr": "Y155"},
        )
        gate = gate_cofactor_compatibility(cand, task(),
                                           {"cat:sdr:kred": catalytic_template()})
        self.assertTrue(gate.passed)

    def test_unknown_cofactor_name_is_unevaluated_not_matched(self) -> None:
        cand = candidate("c1")
        cand.catalytic_mapping = CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Tyr": "Y155"},
        )
        spec = task(cofactors=(CofactorSpec(name="mystery cofactor X"),))
        gate = gate_cofactor_compatibility(cand, spec,
                                           {"cat:sdr:kred": catalytic_template()})
        self.assertTrue(gate.unevaluated)

    def test_missing_template_in_a_present_library_raises(self) -> None:
        cand = candidate("c1")
        cand.catalytic_mapping = CatalyticMapping(
            catalytic_template_id="cat:absent",
            role_to_residue={"catalytic_Tyr": "Y155"},
        )
        with self.assertRaises(TemplateError):
            evaluate_feasibility_gates(cand, task(),
                                       {"cat:sdr:kred": catalytic_template()})

    def test_unannotated_fragment_status_is_unevaluated_not_passed(self) -> None:
        cand = candidate("c1", is_fragment=None)
        gate = gate_input_integrity(cand, task())
        self.assertTrue(gate.unevaluated)

    def test_fragment_sequence_is_an_input_defect(self) -> None:
        cand = candidate("c1", is_fragment=True)
        gate = gate_input_integrity(cand, task())
        self.assertTrue(gate.failed)
        self.assertIs(gate.kind, ErrorKind.INPUT_DEFECT)

    def test_unresolved_target_configuration_fails_the_stereo_gate(self) -> None:
        gate = gate_stereochemistry_resolved(
            candidate("c1"), task(target=Stereochemistry.UNSPECIFIED))
        self.assertTrue(gate.failed)
        self.assertIs(gate.kind, ErrorKind.INPUT_DEFECT)

    def test_undecided_stereocentre_question_is_unevaluated(self) -> None:
        gate = gate_stereochemistry_resolved(
            candidate("c1"), task(creates_stereocenter=None))
        self.assertTrue(gate.unevaluated)

    def test_achiral_product_passes_without_a_configuration(self) -> None:
        gate = gate_stereochemistry_resolved(
            candidate("c1"), task(creates_stereocenter=False,
                                  target=Stereochemistry.UNSPECIFIED))
        self.assertTrue(gate.passed)

    def test_gate_uncertainties_cover_every_undecided_gate(self) -> None:
        cand = candidate("c1", is_fragment=None)
        gates = evaluate_feasibility_gates(cand, task())
        uncs = gate_uncertainties(gates, subject="c1")
        self.assertEqual(len(uncs),
                         sum(1 for g in gates if g.unevaluated))
        for unc in uncs:
            self.assertEqual(unc.affects, ["c1"])


# --------------------------------------------------------------------------
# The two error kinds
# --------------------------------------------------------------------------

class TestErrorKinds(unittest.TestCase):

    def test_weak_evidence_is_never_an_input_defect(self) -> None:
        """Low pocket confidence must not disqualify anything."""
        cand = candidate("c1")
        cand.structures = [StructureRecord(
            structure_id="af-1", source="afdb", pocket_plddt=42.0, mean_plddt=88.0)]
        self.assertEqual(input_defects(cand, task()), [])
        weaknesses = evidence_weaknesses(cand)
        self.assertTrue(weaknesses)
        self.assertFalse(ErrorKind.EVIDENCE_WEAKNESS.disqualifies())
        self.assertTrue(ErrorKind.INPUT_DEFECT.disqualifies())

    def test_nonstandard_residues_are_a_defect(self) -> None:
        cand = candidate("c1", sequence=SEQ[:-1] + "X")
        self.assertTrue(any("standard residues" in d
                            for d in input_defects(cand, task())))

    def test_unsourced_substrate_is_a_defect(self) -> None:
        spec = task()
        spec.reaction.substrate = SubstrateSpec(name="some ketone")
        self.assertTrue(any("isomeric SMILES" in d
                            for d in input_defects(candidate("c1"), spec)))

    def test_qc_flag_severity_separates_the_two_kinds(self) -> None:
        cand = candidate("c1", is_fragment=True)
        gates = evaluate_feasibility_gates(cand, task())
        flags = scorecard_qc_flags(cand, gates)
        blockers = [f for f in flags if f.severity.value == "blocker"]
        warns = [f for f in flags if f.severity.value == "warn"]
        self.assertTrue(blockers, "an input defect must block")
        self.assertTrue(all(f.code == "evidence_weakness" for f in warns))


# --------------------------------------------------------------------------
# build_scorecard
# --------------------------------------------------------------------------

class TestBuildScorecard(unittest.TestCase):

    def full_candidate(self) -> Candidate:
        cand = candidate("c1")
        cand.catalytic_mapping = CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Tyr": "Y155", "catalytic_Ser": "S142",
                             "catalytic_Lys": "K159"},
            alignment_quality=0.91,
        )
        cand.family = FamilyAnnotation(
            family_name="SDR", signals_supporting=["pfam", "motif", "clade"],
            confidence=ConfidenceLevel.STRONG)
        cand.structures = [StructureRecord(structure_id="af-1", source="afdb",
                                           pocket_plddt=90.0, mean_plddt=93.0,
                                           priority_rank=1)]
        cand.poses = [
            ComplexPose(pose_id="p1", method="template_docking", docking_score=-9.1,
                        docking_score_function="vina", cofactor_present=True,
                        cofactor_state=CofactorState.REDUCED,
                        substrate_source=LigandSource.DOCKING_PREDICTED),
            ComplexPose(pose_id="p2", method="af3", docking_score=-8.2,
                        docking_score_function="vina", cofactor_present=True,
                        cofactor_state=CofactorState.REDUCED,
                        substrate_source=LigandSource.JOINT_STRUCTURE_PREDICTION),
        ]
        cand.geometry = [
            GeometryReport(pose_id="p1", gating_passed=True,
                           independent_satisfied=2, independent_total=2),
            GeometryReport(pose_id="p2", gating_passed=True,
                           independent_satisfied=1, independent_total=2),
        ]
        cand.robustness_G = 0.8
        cand.evidence = [EvidenceRef(source_type="publication", identifier="PMID:1",
                                     strength=EvidenceStrength.HOMOLOG_EXPERIMENTAL)]
        return cand

    def test_every_axis_is_present(self) -> None:
        card = build_scorecard(self.full_candidate(), task(),
                               {"cat:sdr:kred": catalytic_template()},
                               {"docking_score_direction": {"vina": "lower_is_better"}})
        for name in SCORE_DIMENSIONS:
            self.assertIn(name, card)
            self.assertEqual(card[name].name, name)
        for name in GATE_NAMES:
            self.assertIn(name, card)
            self.assertTrue(card[name].is_gate)

    def test_absent_models_stay_insufficient_rather_than_guessed(self) -> None:
        card = build_scorecard(self.full_candidate(), task(),
                               {"cat:sdr:kred": catalytic_template()}, {})
        self.assertIs(card["substrate_specificity_model"].level,
                      ConfidenceLevel.INSUFFICIENT)
        self.assertIsNone(card["substrate_specificity_model"].value)
        self.assertIs(card["expression_developability_risk"].level,
                      ConfidenceLevel.INSUFFICIENT)

    def test_external_dimension_without_a_basis_is_refused(self) -> None:
        with self.assertRaises(FabricationGuardError):
            build_scorecard(
                self.full_candidate(), task(),
                {"cat:sdr:kred": catalytic_template()},
                {"external_dimensions": {"c1": {"substrate_specificity_model":
                                                {"level": "strong", "value": 0.9}}}},
            )

    def test_undeclared_docking_polarity_is_not_ranked(self) -> None:
        card = build_scorecard(self.full_candidate(), task(),
                               {"cat:sdr:kred": catalytic_template()}, {})
        self.assertIs(card["docking_result"].level, ConfidenceLevel.INSUFFICIENT)
        self.assertIsNone(card["docking_result"].value)

    def test_declared_polarity_picks_the_right_extreme(self) -> None:
        card = build_scorecard(self.full_candidate(), task(),
                               {"cat:sdr:kred": catalytic_template()},
                               {"docking_score_direction": {"vina": "lower_is_better"}})
        self.assertEqual(card["docking_result"].value, -9.1)
        self.assertEqual(card["docking_result"].direction, "lower_is_better")

    def test_mean_plddt_does_not_substitute_for_pocket_plddt(self) -> None:
        cand = self.full_candidate()
        cand.structures = [StructureRecord(structure_id="af-1", source="afdb",
                                           mean_plddt=93.0)]
        card = build_scorecard(cand, task(), {"cat:sdr:kred": catalytic_template()}, {})
        local = card["local_structure_confidence"]
        self.assertIs(local.level, ConfidenceLevel.INSUFFICIENT)
        self.assertIsNone(local.value)
        self.assertIn("mean", local.basis)

    def test_entirely_restrained_geometry_is_not_strong(self) -> None:
        cand = self.full_candidate()
        cand.geometry = [GeometryReport(
            pose_id="p1", gating_passed=True, independent_satisfied=0,
            independent_total=0, circular_constraints=["hydride_transfer"])]
        card = build_scorecard(cand, task(), {"cat:sdr:kred": catalytic_template()}, {})
        self.assertIs(card["catalytic_geometry"].level, ConfidenceLevel.WEAK)

    def test_four_databases_of_one_measurement_count_once(self) -> None:
        cand = self.full_candidate()
        cand.evidence = [
            EvidenceRef(source_type="database", identifier=f"db{i}",
                        strength=EvidenceStrength.HOMOLOG_EXPERIMENTAL,
                        experiment_activity_id="assay-7")
            for i in range(4)
        ]
        card = build_scorecard(cand, task(), {"cat:sdr:kred": catalytic_template()}, {})
        self.assertEqual(card["functional_literature_evidence"].value, 1.0)

    def test_explain_answers_the_six_questions(self) -> None:
        cand = self.full_candidate()
        cand.scorecard = build_scorecard(
            cand, task(), {"cat:sdr:kred": catalytic_template()},
            {"docking_score_direction": {"vina": "lower_is_better"}})
        text = explain(cand)
        for heading in ("Why it was retrieved", "Family and mechanism",
                        "Evidence for the target reaction",
                        "Substrate and cofactor placement", "What is uncertain",
                        "Why it deserves a slot"):
            self.assertIn(heading, text)
        # Ligand authority must be stated verbatim, never implied.
        self.assertIn("a docking pose", text)


# --------------------------------------------------------------------------
# Pareto
# --------------------------------------------------------------------------

class TestPareto(unittest.TestCase):

    def setUp(self) -> None:
        # a: good on both. b: dominated by a. c: better on y, worse on x.
        # d: x unknown (INSUFFICIENT) -- must be incomparable, not worst.
        # e: dimension entirely absent -- also incomparable.
        self.a = candidate("a", dims={
            "x": dim("catalytic_geometry", ConfidenceLevel.STRONG, 0.9),
            "y": dim("docking_result", ConfidenceLevel.WEAK, -9.0,
                     direction="lower_is_better"),
        })
        self.b = candidate("b", dims={
            "x": dim("catalytic_geometry", ConfidenceLevel.MODERATE, 0.5),
            "y": dim("docking_result", ConfidenceLevel.WEAK, -7.0,
                     direction="lower_is_better"),
        })
        self.c = candidate("c", dims={
            "x": dim("catalytic_geometry", ConfidenceLevel.WEAK, 0.3),
            "y": dim("docking_result", ConfidenceLevel.WEAK, -11.0,
                     direction="lower_is_better"),
        })
        self.d = candidate("d", dims={
            "x": dim("catalytic_geometry", ConfidenceLevel.INSUFFICIENT, None),
            "y": dim("docking_result", ConfidenceLevel.WEAK, -6.0,
                     direction="lower_is_better"),
        })
        self.e = candidate("e", dims={
            "y": dim("docking_result", ConfidenceLevel.WEAK, -6.5,
                     direction="lower_is_better"),
        })
        self.objectives = [("catalytic_geometry", "higher_is_better"),
                           ("docking_result", "lower_is_better")]

    def test_dominance(self) -> None:
        self.assertTrue(dominates(self.a, self.b, self.objectives))
        self.assertFalse(dominates(self.b, self.a, self.objectives))
        self.assertFalse(dominates(self.a, self.c, self.objectives))
        self.assertFalse(dominates(self.c, self.a, self.objectives))

    def test_an_unmeasured_axis_is_incomparable_not_worst(self) -> None:
        """d has a worse docking score than a and no geometry at all.

        If None sorted as the worst value, a would dominate d and d would
        vanish from the front -- a gap in the pipeline turned into a verdict on
        the enzyme.
        """
        self.assertFalse(dominates(self.a, self.d, self.objectives))
        self.assertFalse(dominates(self.d, self.a, self.objectives))
        self.assertIn(self.d, pareto_front(
            [self.a, self.b, self.c, self.d], self.objectives))

    def test_a_missing_dimension_is_also_incomparable(self) -> None:
        self.assertFalse(dominates(self.a, self.e, self.objectives))
        self.assertIn(self.e, pareto_front([self.a, self.e], self.objectives))

    def test_front_excludes_only_the_dominated(self) -> None:
        front = pareto_front([self.a, self.b, self.c, self.d, self.e],
                             self.objectives)
        ids = [c.candidate_id for c in front]
        self.assertNotIn("b", ids)
        self.assertEqual(ids, ["a", "c", "d", "e"], "input order must be preserved")

    def test_identical_candidates_do_not_dominate_each_other(self) -> None:
        twin = candidate("a2", dims={
            "x": dim("catalytic_geometry", ConfidenceLevel.STRONG, 0.9),
            "y": dim("docking_result", ConfidenceLevel.WEAK, -9.0,
                     direction="lower_is_better"),
        })
        self.assertFalse(dominates(self.a, twin, self.objectives))
        self.assertEqual(len(pareto_front([self.a, twin], self.objectives)), 2)

    def test_empty_objectives_raise(self) -> None:
        with self.assertRaises(ValueError):
            pareto_front([self.a], [])

    def test_wrong_direction_raises_instead_of_inverting(self) -> None:
        with self.assertRaises(ValueError):
            objective_value(self.a, "docking_result", "higher_is_better")

    def test_a_gate_is_not_an_objective(self) -> None:
        with self.assertRaises(ValueError):
            objective_value(self.a, "catalytic_machinery_mappable",
                            "higher_is_better")

    def test_contradictory_is_incomparable(self) -> None:
        contradictory = candidate("f", dims={
            "x": dim("catalytic_geometry", ConfidenceLevel.CONTRADICTORY, 0.9),
        })
        value, _ = comparable(contradictory.dimension("catalytic_geometry"))
        self.assertIsNone(value)

    def test_level_fallback_is_always_higher_is_better(self) -> None:
        """A lower_is_better axis with no scalar must not invert on the level."""
        risk = dim("expression_developability_risk", ConfidenceLevel.MODERATE,
                   None, direction="lower_is_better")
        value, direction = comparable(risk)
        self.assertEqual(direction, "higher_is_better")
        self.assertEqual(value, float(ConfidenceLevel.MODERATE.rank))


# --------------------------------------------------------------------------
# Lexicographic and within-family ranking
# --------------------------------------------------------------------------

class TestLexicographic(unittest.TestCase):

    def ranked_set(self) -> list[Candidate]:
        strong_lit = candidate("lit", dims={
            "a": dim("functional_literature_evidence", ConfidenceLevel.STRONG, 3.0),
            "b": dim("catalytic_geometry", ConfidenceLevel.WEAK, 0.1),
        })
        good_geom = candidate("geom", dims={
            "a": dim("functional_literature_evidence", ConfidenceLevel.WEAK, 1.0),
            "b": dim("catalytic_geometry", ConfidenceLevel.STRONG, 1.0),
        })
        nothing = candidate("none")
        return [good_geom, nothing, strong_lit]

    def test_first_dimension_decides(self) -> None:
        order = ["functional_literature_evidence", "catalytic_geometry"]
        ids = [c.candidate_id for c in lexicographic_rank(self.ranked_set(), order)]
        self.assertEqual(ids[:2], ["lit", "geom"])

    def test_reordering_the_priorities_reorders_the_result(self) -> None:
        order = ["catalytic_geometry", "functional_literature_evidence"]
        ids = [c.candidate_id for c in lexicographic_rank(self.ranked_set(), order)]
        self.assertEqual(ids[:2], ["geom", "lit"])

    def test_unknown_sorts_last_at_that_dimension(self) -> None:
        ids = [c.candidate_id for c in lexicographic_rank(self.ranked_set())]
        self.assertEqual(ids[-1], "none")

    def test_failed_gate_cannot_outrank_a_passing_candidate(self) -> None:
        failed = candidate("failed", gates_pass=False, dims={
            "a": dim("functional_literature_evidence", ConfidenceLevel.STRONG, 99.0),
        })
        weak_but_eligible = candidate("ok", dims={
            "a": dim("functional_literature_evidence", ConfidenceLevel.WEAK, 1.0),
        })
        ids = [c.candidate_id
               for c in lexicographic_rank([failed, weak_but_eligible])]
        self.assertEqual(ids, ["ok", "failed"])

    def test_undecided_gate_sits_between_pass_and_fail(self) -> None:
        passing = candidate("p")
        undecided = candidate("u", gates_pass=None)
        failed = candidate("f", gates_pass=False)
        ids = [c.candidate_id
               for c in lexicographic_rank([failed, undecided, passing])]
        self.assertEqual(ids, ["p", "u", "f"])

    def test_tie_break_is_the_candidate_id(self) -> None:
        twins = [candidate(cid) for cid in ("zz", "aa", "mm")]
        ids = [c.candidate_id for c in lexicographic_rank(twins)]
        self.assertEqual(ids, ["aa", "mm", "zz"])
        self.assertEqual(ids, [c.candidate_id
                               for c in lexicographic_rank(list(reversed(twins)))])

    def test_ungated_candidate_raises_rather_than_ranking_silently(self) -> None:
        bare = Candidate(candidate_id="bare",
                         sequence_record=SequenceRecord(candidate_id="bare",
                                                        sequence=SEQ))
        with self.assertRaises(ValueError):
            lexicographic_rank([bare])

    def test_default_order_covers_every_axis(self) -> None:
        self.assertEqual(set(DEFAULT_LEXICOGRAPHIC_ORDER), set(SCORE_DIMENSIONS))

    def test_rank_within_family_never_mixes_families(self) -> None:
        members = [candidate("s1", family="SDR"), candidate("a1", family="AKR"),
                   candidate("s2", family="SDR"), candidate("x1", family=None)]
        grouped = rank_within_family(members)
        self.assertEqual(set(grouped), {"SDR", "AKR", "unassigned"})
        self.assertEqual([c.candidate_id for c in grouped["SDR"]], ["s1", "s2"])
        self.assertEqual(len(grouped["AKR"]), 1)


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------

class TestRefuseLinearBlend(unittest.TestCase):

    def test_it_always_raises(self) -> None:
        with self.assertRaises(FabricationGuardError):
            refuse_linear_blend({"local_structure_confidence": 0.4,
                                 "docking_result": 0.3,
                                 "catalytic_geometry": 0.3})

    def test_it_raises_with_no_arguments_at_all(self) -> None:
        with self.assertRaises(FabricationGuardError):
            refuse_linear_blend()

    def test_it_raises_for_a_bare_weight_vector(self) -> None:
        with self.assertRaises(FabricationGuardError):
            refuse_linear_blend([0.4, 0.3, 0.3])

    def test_the_message_names_the_alternatives(self) -> None:
        with self.assertRaises(FabricationGuardError) as ctx:
            refuse_linear_blend({"a": 1.0})
        message = str(ctx.exception)
        for alternative in ("pareto_front", "lexicographic_rank",
                            "rank_within_family", "gates"):
            self.assertIn(alternative, message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
