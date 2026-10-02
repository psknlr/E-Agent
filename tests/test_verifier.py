"""Tests for :mod:`eagent.harness.verifier`.

Each case is a wrong thing that a run can produce while every artifact looks
right:

* a report pairing a candidate with somebody else's coordinates;
* a docked molecule that is not the one in the spec, or is its enantiomer;
* NADP+ in a complex whose template requires NADPH, so the modelled reaction
  cannot occur;
* a catalytic residue named ``Y155`` and measured at a different index;
* a sentence in a report that no artifact and no provenance entry supports;
* an ee in percent that nothing was calibrated on;
* a candidate deleted from the pool by a window nobody fitted;
* a restrained distance counted afterwards as independent corroboration;
* a crashed docking run written into the record layer as "no activity
  detected".

The verifier is given primary material -- sequences, a coordinate file, the
template, the pose's restraint list -- and has to find these without being
told. Where it is given a producing step's own verdict, the tests check that
it contradicts it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eagent.context import RunContext
from eagent.envelope import Artifact, Provenance, Severity, Status, ToolResult
from eagent.harness.verifier import (
    IGNORED_PRODUCER_CONCLUSIONS,
    Claim,
    IndependentVerifier,
    LigandDeclaration,
)
from eagent.provenance import RunManifest, utc_now
from eagent.schemas import (
    Candidate,
    CatalyticMapping,
    CatalyticTemplate,
    CofactorState,
    ComplexPose,
    Detection,
    EvidenceRef,
    EvidenceStrength,
    ExperimentRecord,
    FamilyAnnotation,
    GeometryConstraint,
    GeometryReport,
    OutcomeClass,
    ProductSpec,
    ReactionSpec,
    SequenceRecord,
    Stereochemistry,
    StereoCall,
    StructureRecord,
    SubstrateSpec,
    TaskSpec,
    TemplateProvenance,
    TemplateSourceType,
)
from eagent.science.numbering import ONE_TO_THREE

SEQ = "MKAYVLGSGYTKLDEWMNAQPFVRTICHADKLGEYNSPQ"
SUBSTRATE_SMILES = "CC(=O)c1ccc(Cl)cc1"
PRODUCT_SMILES = "C[C@H](O)c1ccc(Cl)cc1"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def pdb_text(sequence: str, chain: str = "A", start: int = 1) -> str:
    """One CA atom per residue: enough for an alignment, nothing more."""
    lines = []
    for i, letter in enumerate(sequence):
        resname = ONE_TO_THREE[letter]
        x, y, z = 1.5 * i, 0.0, 0.0
        lines.append(
            f"ATOM  {i + 1:5d} {' CA ':4}{' '}{resname:>3} {chain}"
            f"{start + i:4d}{' '}   {x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{20.00:6.2f}          {'C':>2}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def template(**over) -> CatalyticTemplate:
    kwargs = dict(
        template_id="cat.test.v1",
        family_name="TEST",
        required_cofactor="NADPH",
        required_cofactor_state=CofactorState.REDUCED,
        cofactor_ligand_codes=["NDP"],
        geometry_constraints=[
            GeometryConstraint(
                name="donor_to_electrophile", kind="distance",
                atom_a="cofactor.hydride_donor_C4",
                atom_b="substrate.electrophile",
                min_value=2.9, max_value=3.9, severity="scoring",
                calibrated_on=[], source="general chemistry, not fitted"),
            GeometryConstraint(
                name="carbonyl_O_to_tyr_OH", kind="distance",
                atom_a="substrate.carbonyl_O", atom_b="protein.tyr.OH",
                min_value=2.5, max_value=3.2, severity="scoring",
                calibrated_on=["internal:ternary_complex_set"],
                source="fitted on the set named above"),
        ],
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.MECHANISM_LITERATURE,
            identifiers=["textbook:test"], notes="unit test"),
    )
    kwargs.update(over)
    return CatalyticTemplate(**kwargs)


def candidate(cid: str = "C1", *, sequence: str = SEQ,
              roles: dict[str, tuple[str, int]] | None = None,
              **over) -> Candidate:
    roles = roles if roles is not None else {"catalytic_tyr": ("Y4", 3)}
    cand = Candidate(
        candidate_id=cid,
        sequence_record=SequenceRecord(candidate_id=cid, sequence=sequence),
        family=FamilyAnnotation(family_name="TEST"),
        catalytic_mapping=CatalyticMapping(
            catalytic_template_id="cat.test.v1",
            role_to_residue={k: v[0] for k, v in roles.items()},
            role_to_index={k: v[1] for k, v in roles.items()}),
        **over)
    return cand


def good_pose(pose_id: str = "P1", **over) -> ComplexPose:
    kwargs = dict(pose_id=pose_id, method="template_docking",
                  substrate_present=True, cofactor_present=True,
                  cofactor_state=CofactorState.REDUCED)
    kwargs.update(over)
    return ComplexPose(**kwargs)


def make_ctx(tmp: Path, *, stereo: bool = True) -> RunContext:
    task = TaskSpec(
        task_id="T1",
        reaction=ReactionSpec(
            substrate=SubstrateSpec(name="4'-chloroacetophenone",
                                    isomeric_smiles=SUBSTRATE_SMILES,
                                    inchikey="AAAAAAAAAAAAAA-BBBBBBBBBB-N"),
            product=ProductSpec(
                isomeric_smiles=PRODUCT_SMILES,
                target_stereochemistry=(Stereochemistry.S if stereo
                                        else Stereochemistry.UNSPECIFIED),
                creates_new_stereocenter=stereo)))
    return RunContext(task=task, workdir=tmp,
                      manifest=RunManifest(run_id="R1", task_id="T1"))


def record_a_step(ctx: RunContext, *, step_id: str = "evaluate",
                  interface: str = "evaluate_catalysis",
                  artifact_key: str = "catalytic_geometry") -> None:
    result = ToolResult(status=Status.SUCCESS,
                        provenance=Provenance(tool=interface),
                        artifacts=[Artifact(key=artifact_key,
                                            path="geometry.tsv")])
    ctx.manifest.record(step_id, interface, result, utc_now())


def codes(result: ToolResult, severity: Severity = Severity.BLOCKER) -> set[str]:
    return {f.code for f in result.qc_flags if f.severity is severity}


def verify(ctx: RunContext, **kwargs) -> ToolResult:
    kwargs.setdefault("catalytic_templates", {"cat.test.v1": template()})
    return IndependentVerifier().verify(ctx, **kwargs)


# ---------------------------------------------------------------------------
# sequence / structure pairing
# ---------------------------------------------------------------------------

class SequenceStructurePairingTests(unittest.TestCase):

    def _run(self, structure_sequence: str, **record_over):
        self._dir = tempfile.TemporaryDirectory()
        tmp = Path(self._dir.name)
        path = tmp / "model.pdb"
        path.write_text(pdb_text(structure_sequence), encoding="utf-8")
        record = StructureRecord(structure_id="S1", source="predicted",
                                 path=str(path), format="pdb", **record_over)
        cand = candidate(structures=[record])
        ctx = make_ctx(tmp)
        return verify(ctx, candidates=[cand])

    def tearDown(self):
        if hasattr(self, "_dir"):
            self._dir.cleanup()

    def test_an_undeclared_mismatch_is_a_blocker(self):
        wrong = SEQ[:10] + "W" + SEQ[11:]
        result = self._run(wrong)
        self.assertIn("sequence_structure_mismatch", codes(result))
        self.assertIs(result.status, Status.FAILED)
        self.assertTrue(result.blockers)

    def test_a_declared_mutation_is_accepted(self):
        wrong = SEQ[:10] + "W" + SEQ[11:]
        mutated = f"{SEQ[10]}11W"
        result = self._run(wrong, is_mutant_relative_to_candidate=True,
                           mutations_in_structure=[mutated])
        self.assertNotIn("sequence_structure_mismatch", codes(result))

    def test_a_matching_structure_passes(self):
        result = self._run(SEQ)
        self.assertNotIn("sequence_structure_mismatch", codes(result))

    def test_extra_residues_are_reported_as_a_warning(self):
        result = self._run("HHHHHH" + SEQ)
        self.assertIn("structure_has_extra_residues",
                      codes(result, Severity.WARN))

    def test_a_missing_file_is_unverified_not_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = StructureRecord(structure_id="S1", source="predicted",
                                     path=str(Path(tmp) / "absent.pdb"),
                                     format="pdb")
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, candidates=[candidate(structures=[record])])
            self.assertIn("structure_file_missing", codes(result, Severity.WARN))
            self.assertIs(result.status, Status.PARTIAL)
            self.assertTrue(result.data["verification"]["unverifiable"])

    def test_a_named_chain_the_file_does_not_hold_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pdb"
            path.write_text(pdb_text(SEQ, chain="A"), encoding="utf-8")
            record = StructureRecord(structure_id="S1", source="pdb_apo",
                                     path=str(path), format="pdb")
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, candidates=[candidate(structures=[record])],
                            structure_chains={"S1": "B"})
            self.assertIn("structure_chain_absent", codes(result))


# ---------------------------------------------------------------------------
# ligand identity and chirality
# ---------------------------------------------------------------------------

class LigandIdentityTests(unittest.TestCase):

    def _verify(self, declaration: LigandDeclaration, *, stereo: bool = True):
        self._dir = tempfile.TemporaryDirectory()
        ctx = make_ctx(Path(self._dir.name), stereo=stereo)
        cand = candidate(poses=[good_pose()])
        return verify(ctx, candidates=[cand],
                      ligands={"P1": declaration})

    def tearDown(self):
        if hasattr(self, "_dir"):
            self._dir.cleanup()

    def test_the_spec_substrate_passes(self):
        result = self._verify(LigandDeclaration(
            pose_id="P1", substrate_inchikey="AAAAAAAAAAAAAA-BBBBBBBBBB-N"))
        self.assertNotIn("ligand_identity_mismatch", codes(result))

    def test_a_different_compound_is_a_blocker(self):
        result = self._verify(LigandDeclaration(
            pose_id="P1", substrate_inchikey="ZZZZZZZZZZZZZZ-BBBBBBBBBB-N"))
        self.assertIn("ligand_identity_mismatch", codes(result))

    def test_the_same_skeleton_with_another_configuration_is_flagged_as_chirality(self):
        result = self._verify(LigandDeclaration(
            pose_id="P1", substrate_inchikey="AAAAAAAAAAAAAA-CCCCCCCCCC-N"))
        self.assertIn("ligand_chirality_mismatch", codes(result))
        self.assertNotIn("ligand_identity_mismatch", codes(result))

    def test_a_smiles_difference_that_is_only_stereo_is_flagged_as_chirality(self):
        self._dir = tempfile.TemporaryDirectory()
        ctx = make_ctx(Path(self._dir.name))
        ctx.task.reaction.substrate.inchikey = None
        cand = candidate(poses=[good_pose()])
        result = verify(ctx, candidates=[cand], ligands={
            "P1": LigandDeclaration(pose_id="P1",
                                    substrate_smiles="C[C@@H](O)c1ccccc1")})
        self.assertIn("ligand_identity_mismatch", codes(result))

    def test_the_wrong_product_configuration_is_a_blocker(self):
        result = self._verify(LigandDeclaration(
            pose_id="P1", substrate_inchikey="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            product_configuration=Stereochemistry.R))
        self.assertIn("product_configuration_mismatch", codes(result))

    def test_the_right_product_configuration_passes(self):
        result = self._verify(LigandDeclaration(
            pose_id="P1", substrate_inchikey="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            product_configuration=Stereochemistry.S))
        self.assertNotIn("product_configuration_mismatch", codes(result))

    def test_an_undeclared_ligand_is_unverified_not_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, candidates=[candidate(poses=[good_pose()])])
            self.assertTrue(any("identity" in u for u in
                                result.data["verification"]["unverifiable"]))


# ---------------------------------------------------------------------------
# cofactor
# ---------------------------------------------------------------------------

class CofactorTests(unittest.TestCase):

    def _verify(self, pose: ComplexPose, *, tpl: CatalyticTemplate | None = None,
                declaration: LigandDeclaration | None = None):
        self._dir = tempfile.TemporaryDirectory()
        ctx = make_ctx(Path(self._dir.name))
        cand = candidate(poses=[pose])
        return IndependentVerifier().verify(
            ctx, candidates=[cand],
            catalytic_templates={"cat.test.v1": tpl or template()},
            ligands={pose.pose_id: declaration} if declaration else None)

    def tearDown(self):
        if hasattr(self, "_dir"):
            self._dir.cleanup()

    def test_an_oxidised_cofactor_against_a_reduced_template_is_a_blocker(self):
        result = self._verify(good_pose(cofactor_state=CofactorState.OXIDIZED))
        self.assertIn("cofactor_oxidation_state_mismatch", codes(result))
        self.assertIs(result.status, Status.FAILED)

    def test_the_reduced_cofactor_the_template_requires_passes(self):
        result = self._verify(good_pose())
        self.assertNotIn("cofactor_oxidation_state_mismatch", codes(result))

    def test_an_unrecorded_state_is_a_warning_and_an_open_question(self):
        result = self._verify(good_pose(cofactor_state=CofactorState.UNKNOWN))
        self.assertIn("cofactor_state_unknown", codes(result, Severity.WARN))
        self.assertNotIn("cofactor_oxidation_state_mismatch", codes(result))

    def test_an_oxidised_ligand_code_contradicts_a_reduced_declaration(self):
        result = self._verify(
            good_pose(),
            declaration=LigandDeclaration(
                pose_id="P1", cofactor_ligand_code="NAP",
                cofactor_state=CofactorState.REDUCED))
        self.assertIn("cofactor_state_contradicts_ligand_code", codes(result))

    def test_the_wrong_nicotinamide_is_a_blocker(self):
        result = self._verify(
            good_pose(),
            declaration=LigandDeclaration(pose_id="P1", cofactor_name="NADH"))
        self.assertIn("cofactor_identity_mismatch", codes(result))

    def test_a_missing_cofactor_is_a_blocker_when_the_template_needs_one(self):
        result = self._verify(good_pose(cofactor_present=False))
        self.assertIn("required_cofactor_absent", codes(result))

    def test_a_missing_catalytic_metal_is_a_blocker(self):
        result = self._verify(good_pose(), tpl=template(metals=["ZN"]))
        self.assertIn("required_metal_absent", codes(result))

    def test_no_template_means_unverified_rather_than_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            result = IndependentVerifier().verify(
                ctx, candidates=[candidate(poses=[good_pose()])],
                catalytic_templates={})
            self.assertTrue(any("catalytic template" in u for u in
                                result.data["verification"]["unverifiable"]))


# ---------------------------------------------------------------------------
# numbering
# ---------------------------------------------------------------------------

class NumberingRoundTripTests(unittest.TestCase):

    def _verify(self, roles):
        self._dir = tempfile.TemporaryDirectory()
        ctx = make_ctx(Path(self._dir.name))
        return verify(ctx, candidates=[candidate(roles=roles)])

    def tearDown(self):
        if hasattr(self, "_dir"):
            self._dir.cleanup()

    def test_a_consistent_mapping_round_trips(self):
        index = SEQ.index("Y")
        result = self._verify({"catalytic_tyr": (f"Y{index + 1}", index)})
        self.assertNotIn("numbering_round_trip_failed", codes(result))

    def test_an_off_by_a_tag_mapping_is_a_blocker(self):
        index = SEQ.index("Y")
        result = self._verify({"catalytic_tyr": (f"Y{index + 1}", index + 6)})
        self.assertIn("numbering_round_trip_failed", codes(result))

    def test_a_role_with_no_index_is_a_blocker(self):
        self._dir = tempfile.TemporaryDirectory()
        ctx = make_ctx(Path(self._dir.name))
        cand = candidate()
        cand.catalytic_mapping.role_to_index = {}
        result = verify(ctx, candidates=[cand])
        self.assertIn("role_without_index", codes(result))

    def test_an_unparseable_token_is_a_blocker(self):
        result = self._verify({"catalytic_tyr": ("Tyr-155", 3)})
        self.assertIn("residue_token_unparseable", codes(result))

    def test_an_index_past_the_end_is_a_blocker(self):
        result = self._verify({"catalytic_tyr": ("Y4", 9999)})
        self.assertIn("residue_index_out_of_range", codes(result))


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------

class ClaimBackingTests(unittest.TestCase):

    def test_a_backed_claim_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record_a_step(ctx)
            result = verify(ctx, claims=[Claim(
                claim_id="K1", text="the mapping is complete",
                artifact_key="catalytic_geometry", step_id="evaluate")])
            self.assertEqual(codes(result), set())

    def test_a_claim_with_no_artifact_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record_a_step(ctx)
            result = verify(ctx, claims=[Claim(
                claim_id="K1", text="C1 is the most promising candidate",
                step_id="evaluate")])
            self.assertIn("claim_without_artifact", codes(result))

    def test_a_claim_citing_an_artifact_nobody_produced_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record_a_step(ctx)
            result = verify(ctx, claims=[Claim(
                claim_id="K1", text="see the table",
                artifact_key="imaginary_table", step_id="evaluate")])
            self.assertIn("claim_artifact_missing", codes(result))

    def test_a_claim_with_no_provenance_entry_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record_a_step(ctx)
            result = verify(ctx, claims=[Claim(
                claim_id="K1", text="see the table",
                artifact_key="catalytic_geometry", step_id="never_ran")])
            self.assertIn("claim_without_provenance", codes(result))

    def test_a_supplied_artifact_list_also_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record_a_step(ctx)
            result = verify(
                ctx,
                claims=[Claim(claim_id="K1", artifact_key="extra",
                              interface="evaluate_catalysis")],
                artifacts=[Artifact(key="extra", path="extra.tsv")])
            self.assertEqual(codes(result), set())


# ---------------------------------------------------------------------------
# numeric ee
# ---------------------------------------------------------------------------

class NumericEeTests(unittest.TestCase):

    def test_a_predicted_ee_without_a_calibration_source_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            cand = candidate()
            cand.stereo = StereoCall.model_construct(
                call="favors_target", target_face_poses=8,
                opposite_face_poses=0, undetermined_poses=0,
                basis="pose counting", predicted_ee_pct=96.0,
                calibration_source=None)
            result = verify(ctx, candidates=[cand])
            self.assertIn("uncited_numeric_ee", codes(result))

    def test_a_predicted_ee_with_a_named_calibration_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            cand = candidate()
            cand.stereo = StereoCall(
                call="favors_target", predicted_ee_pct=96.0,
                calibration_source="internal:kred_ee_model@2026-01")
            result = verify(ctx, candidates=[cand])
            self.assertNotIn("uncited_numeric_ee", codes(result))

    def test_an_ee_in_prose_needs_a_source_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record_a_step(ctx)
            result = verify(ctx, claims=[Claim(
                claim_id="K1", text="C1 should reach about 98% ee",
                artifact_key="catalytic_geometry", step_id="evaluate")])
            self.assertIn("uncited_numeric_ee", codes(result))

    def test_a_numeric_ee_claim_field_needs_a_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record_a_step(ctx)
            result = verify(ctx, claims=[Claim(
                claim_id="K1", metric="ee_target_pct", value=91.0, unit="%",
                artifact_key="catalytic_geometry", step_id="evaluate")])
            self.assertIn("uncited_numeric_ee", codes(result))

    def test_an_ee_measured_on_a_validated_chiral_method_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record_a_step(ctx)
            measured = ExperimentRecord(
                record_id="R1", sequence=SEQ,
                outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                ee_target_pct=91.0,
                detection=Detection(method="chiral GC-MS",
                                    confirms_product_identity=True,
                                    authentic_standard=True,
                                    chiral_method_validated=True),
                evidence=[EvidenceRef(source_type="internal_experiment",
                                      identifier="RUN-1",
                                      strength=EvidenceStrength.
                                      SEQUENCE_LEVEL_EXPERIMENTAL)])
            result = verify(ctx, records=[measured], claims=[Claim(
                claim_id="K1", subject="R1", metric="ee_target_pct",
                value=91.0, artifact_key="catalytic_geometry",
                step_id="evaluate")])
            self.assertNotIn("uncited_numeric_ee", codes(result))

    def test_an_ee_from_an_unvalidated_separation_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            loose = ExperimentRecord(
                record_id="R2", sequence=SEQ,
                outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                ee_target_pct=55.0,
                detection=Detection(method="chiral HPLC",
                                    confirms_product_identity=True),
                evidence=[EvidenceRef(source_type="internal_experiment",
                                      identifier="RUN-2")])
            result = verify(ctx, records=[loose])
            self.assertIn("ee_without_validated_chiral_method", codes(result))


# ---------------------------------------------------------------------------
# uncalibrated disqualification
# ---------------------------------------------------------------------------

class UncalibratedDisqualificationTests(unittest.TestCase):

    def _verify(self, *, constraint: str, tpl: CatalyticTemplate | None = None,
                explicit: bool = False):
        self._dir = tempfile.TemporaryDirectory()
        ctx = make_ctx(Path(self._dir.name))
        cand = candidate(
            disqualified=True,
            disqualification_reason=f"failed {constraint} in every pose")
        kwargs = {"disqualifying_constraints": {"C1": [constraint]}} \
            if explicit else {}
        return IndependentVerifier().verify(
            ctx, candidates=[cand],
            catalytic_templates={"cat.test.v1": tpl or template()}, **kwargs)

    def tearDown(self):
        if hasattr(self, "_dir"):
            self._dir.cleanup()

    def test_disqualifying_on_an_uncalibrated_window_alone_is_a_blocker(self):
        result = self._verify(constraint="donor_to_electrophile")
        self.assertIn("disqualified_by_uncalibrated_window", codes(result))

    def test_the_same_through_the_explicit_constraint_list(self):
        result = self._verify(constraint="donor_to_electrophile",
                              explicit=True)
        self.assertIn("disqualified_by_uncalibrated_window", codes(result))

    def test_disqualifying_on_a_calibrated_window_is_allowed(self):
        result = self._verify(constraint="carbonyl_O_to_tyr_OH")
        self.assertNotIn("disqualified_by_uncalibrated_window", codes(result))

    def test_a_calibrated_window_on_a_theoretical_template_still_may_not_reject(self):
        theoretical = template(provenance=TemplateProvenance(
            source_type=TemplateSourceType.THEORETICAL_MODEL,
            identifiers=["theozyme:test"], notes="computed, not observed"))
        result = self._verify(constraint="carbonyl_O_to_tyr_OH",
                              tpl=theoretical)
        self.assertIn("disqualified_by_uncalibrated_window", codes(result))

    def test_a_candidate_that_was_not_disqualified_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, candidates=[candidate()])
            self.assertNotIn("disqualified_by_uncalibrated_window",
                             codes(result))


# ---------------------------------------------------------------------------
# circular evidence
# ---------------------------------------------------------------------------

class RestrainedEvidenceTests(unittest.TestCase):

    def _verify(self, pose: ComplexPose, geometry: GeometryReport):
        self._dir = tempfile.TemporaryDirectory()
        ctx = make_ctx(Path(self._dir.name))
        cand = candidate(poses=[pose], geometry=[geometry])
        return verify(ctx, candidates=[cand])

    def tearDown(self):
        if hasattr(self, "_dir"):
            self._dir.cleanup()

    def test_a_restrained_constraint_counted_as_independent_is_a_blocker(self):
        pose = good_pose(restrained_constraints=["donor_to_electrophile"])
        geometry = GeometryReport(
            pose_id="P1",
            satisfied={"donor_to_electrophile": True,
                       "carbonyl_O_to_tyr_OH": True},
            circular_constraints=[], independent_satisfied=2,
            independent_total=2)
        result = self._verify(pose, geometry)
        self.assertIn("restrained_counted_as_independent", codes(result))
        self.assertIn("independent_evidence_overcounted", codes(result))

    def test_an_honestly_annotated_report_passes(self):
        from eagent.science.robustness import CircularityGuard

        pose = good_pose(restrained_constraints=["donor_to_electrophile"])
        raw = GeometryReport(pose_id="P1",
                             satisfied={"donor_to_electrophile": True,
                                        "carbonyl_O_to_tyr_OH": True})
        annotated = CircularityGuard.for_report(pose, raw).annotate(raw)
        result = self._verify(pose, annotated)
        self.assertNotIn("restrained_counted_as_independent", codes(result))
        self.assertNotIn("independent_evidence_overcounted", codes(result))
        self.assertEqual(annotated.independent_satisfied, 1)

    def test_an_entirely_circular_pose_is_a_blocker(self):
        pose = good_pose(restrained_constraints=["donor_to_electrophile"])
        geometry = GeometryReport(
            pose_id="P1", satisfied={"donor_to_electrophile": True},
            circular_constraints=["donor_to_electrophile"],
            independent_satisfied=0, independent_total=0)
        result = self._verify(pose, geometry)
        self.assertIn("evidence_entirely_circular", codes(result))

    def test_a_geometry_report_for_an_unknown_pose_is_a_blocker(self):
        result = self._verify(good_pose("P1"),
                              GeometryReport(pose_id="P9"))
        self.assertIn("geometry_report_without_pose", codes(result))


# ---------------------------------------------------------------------------
# experimental labels
# ---------------------------------------------------------------------------

class ExperimentalLabelTests(unittest.TestCase):

    @staticmethod
    def negative(**over) -> ExperimentRecord:
        kwargs = dict(
            record_id="C1", sequence=SEQ,
            outcome=OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
            detection=Detection(method="chiral GC-MS",
                                limit_of_detection=0.5, limit_unit="uM",
                                confirms_product_identity=True),
            evidence=[EvidenceRef(source_type="internal_experiment",
                                  identifier="RUN-1")])
        kwargs.update(over)
        return ExperimentRecord(**kwargs)

    def test_a_computational_failure_may_not_wear_an_experimental_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, records=[self.negative()],
                            computational_failures=["C1"])
            self.assertIn("experimental_label_on_computational_failure",
                          codes(result))

    def test_a_real_negative_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, records=[self.negative()])
            self.assertEqual(codes(result), set())

    def test_a_negative_with_no_detection_method_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record = self.negative(detection=Detection(
                limit_of_detection=0.5, limit_unit="uM"))
            result = verify(ctx, records=[record])
            self.assertIn("negative_without_detection_method", codes(result))

    def test_an_experimental_record_with_no_evidence_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, records=[self.negative(evidence=[])])
            self.assertIn("experimental_record_without_evidence", codes(result))

    def test_a_computational_outcome_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            record = ExperimentRecord(
                record_id="C1", sequence=SEQ,
                outcome=OutcomeClass.COMPUTATIONAL_FAILURE)
            result = verify(ctx, records=[record],
                            computational_failures=["C1"])
            self.assertEqual(codes(result), set(),
                             "labelling a computational failure as one is the "
                             "correct behaviour, not a finding")


# ---------------------------------------------------------------------------
# envelope and independence
# ---------------------------------------------------------------------------

class EnvelopeTests(unittest.TestCase):

    def test_blockers_fail_the_envelope_and_ask_for_a_human(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, candidates=[
                candidate(poses=[good_pose(
                    cofactor_state=CofactorState.OXIDIZED)])])
            self.assertIs(result.status, Status.FAILED)
            self.assertTrue(result.blockers)
            self.assertFalse(result.ok)
            self.assertTrue(result.next_actions[0].requires_human)

    def test_a_clean_run_reports_the_checks_it_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, candidates=[candidate()])
            self.assertIs(result.status, Status.SUCCESS)
            self.assertEqual(len(result.data["verification"]["checks_run"]), 9)

    def test_the_report_artifact_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            result = verify(ctx, candidates=[candidate()])
            artifact = result.artifact("verification_report")
            self.assertIsNotNone(artifact)
            self.assertTrue(Path(artifact.path).exists())

    def test_the_fields_it_refuses_to_trust_are_real_fields(self):
        """A stale ignore-list would quietly stop meaning anything."""
        import eagent.schemas as schemas

        for dotted in IGNORED_PRODUCER_CONCLUSIONS:
            model_name, field_name = dotted.split(".")
            model = getattr(schemas, model_name)
            self.assertTrue(
                field_name in getattr(model, "model_fields", {})
                or hasattr(model, field_name),
                f"{dotted} is not a field or property of {model_name}")

    def test_it_contradicts_a_producing_step_that_marked_its_own_work_clean(self):
        """The producer says the geometry was independent; the restraints say no."""
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            pose = good_pose(restrained_constraints=["donor_to_electrophile"])
            geometry = GeometryReport(
                pose_id="P1", satisfied={"donor_to_electrophile": True},
                gating_passed=True, circular_constraints=[],
                independent_satisfied=1, independent_total=1)
            result = verify(ctx, candidates=[candidate(poses=[pose],
                                                       geometry=[geometry])])
            self.assertIn("restrained_counted_as_independent", codes(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
