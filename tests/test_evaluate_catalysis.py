"""Tests for :mod:`eagent.tools.evaluate_catalysis`.

The cases are chosen around the failures the interface exists to prevent, not
around its happy path. In order:

* the oxidised cofactor slipped into a pose whose geometry is otherwise
  perfect, which a name-matching pipeline would accept;
* a substrate whose *other* carbonyl is the one presented to the hydride
  donor, which every distance and angle check passes;
* a pose whose every satisfied constraint had been restrained during
  modelling, which looks identical to an honest one in every artifact;
* a constraint whose atoms cannot be resolved, which must stay ``None`` rather
  than becoming a pass or a fail;
* a modelling failure, which must never be recorded as an experimental
  negative and must stay distinguishable from a genuine ``G = 0``;
* an uncalibrated template, which must downgrade confidence rather than reject
  candidates en masse.

Everything is built in memory. No file is read, no network is touched, and the
only coordinates are the ones these tests place by hand, so every expected
number can be recomputed from the constructors below.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterable, Sequence

from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Severity, Status
from eagent.provenance import RunManifest
from eagent.schemas import (
    AtomRef,
    Candidate,
    CatalyticMapping,
    CatalyticTemplate,
    CofactorSpec,
    CofactorState,
    ComplexPose,
    Conditions,
    ConfidenceLevel,
    FamilyAnnotation,
    GeometryConstraint,
    LigandSource,
    OutcomeClass,
    ProductSpec,
    ReactionSpec,
    SequenceRecord,
    Stereochemistry,
    SubstrateSpec,
    TaskSpec,
    TemplateProvenance,
    TemplateSourceType,
)
from eagent.science.stereo import CIPPriority
from eagent.science.structure_io import Atom, Chain, Residue, Structure
from eagent.tools.evaluate_catalysis import (
    DEFAULT_POCKET_LOCALISATION_A,
    CandidateEvaluation,
    ChemoselectivityCheck,
    EvaluateCatalysis,
    PoseBinding,
    PoseOutcome,
    PoseEvaluation,
    ProteinAtomRef,
    ResidueSelector,
    WindowAuthority,
    build_role_context,
    classify_aspect,
    pocket_localisation_window,
    template_authority,
)

# --------------------------------------------------------------------------
# Hand-placed geometry.
#
# The carbonyl carbon sits at the origin with its three sp2 ligands in the
# z = 0 plane, so the plane normal is +/- z and every elevation is readable by
# eye. The hydride donor sits 3.4 A away at a 107 degree Burgi-Dunitz angle to
# the C=O vector, which is the arrangement a competent ketoreductase pose has.
# --------------------------------------------------------------------------

CARBONYL_C = (0.0, 0.0, 0.0)
CARBONYL_O = (1.23, 0.0, 0.0)
SUBST_METHYL = (-0.75, 1.30, 0.0)
SUBST_ARYL = (-0.75, -1.30, 0.0)
#: 3.4 A from the carbonyl carbon at 107 degrees from the C=O vector.
HYDRIDE_DONOR = (3.4 * math.cos(math.radians(107.0)), 0.0,
                 3.4 * math.sin(math.radians(107.0)))
#: Tyrosine hydroxyl 2.7 A beyond the carbonyl oxygen, along the C=O vector.
TYR_OH = (CARBONYL_O[0] + 2.70, 0.0, 0.0)
LYS_NZ = (-4.0, 2.0, 4.0)
#: A second, non-target carbonyl carbon sitting 3.0 A from the donor, i.e.
#: closer than the designated reactive atom at 3.4 A.
COMPETING_C = (HYDRIDE_DONOR[0], HYDRIDE_DONOR[1] + 3.0, HYDRIDE_DONOR[2])


def _atom(serial: int, name: str, element: str, resname: str, chain: str,
          resseq: int, xyz: tuple[float, float, float],
          hetatm: bool = False) -> Atom:
    """One coordinate record with no fabricated occupancy or B-factor."""
    return Atom(serial=serial, name=name, element=element, resname=resname,
                chain=chain, resseq=resseq, icode="", altloc="",
                x=xyz[0], y=xyz[1], z=xyz[2], is_hetatm=hetatm)


def _residue(resname: str, chain: str, resseq: int, atoms: Sequence[Atom],
             hetatm: bool = False) -> Residue:
    return Residue(chain=chain, resname=resname, resseq=resseq, icode="",
                   atoms=list(atoms), is_hetatm=hetatm)


def make_structure(
    *,
    cofactor_component: str = "NDP",
    with_competing_carbonyl: bool = False,
    structure_id: str = "pose",
) -> Structure:
    """A minimal ketoreductase active site: Tyr, Lys, the cofactor and the ketone.

    Deliberately tiny. The point of these tests is the decision logic, and a
    four-residue site makes every distance recomputable by hand from the module
    constants above.
    """
    serial = 1

    def nxt() -> int:
        nonlocal serial
        serial += 1
        return serial - 1

    tyr = _residue("TYR", "A", 155, [_atom(nxt(), "OH", "O", "TYR", "A", 155, TYR_OH)])
    lys = _residue("LYS", "A", 159, [_atom(nxt(), "NZ", "N", "LYS", "A", 159, LYS_NZ)])
    cof = _residue(cofactor_component, "A", 301,
                   [_atom(nxt(), "C4N", "C", cofactor_component, "A", 301,
                          HYDRIDE_DONOR, hetatm=True)],
                   hetatm=True)
    lig_atoms = [
        _atom(nxt(), "C1", "C", "LIG", "A", 401, CARBONYL_C, hetatm=True),
        _atom(nxt(), "O1", "O", "LIG", "A", 401, CARBONYL_O, hetatm=True),
        _atom(nxt(), "C2", "C", "LIG", "A", 401, SUBST_METHYL, hetatm=True),
        _atom(nxt(), "C3", "C", "LIG", "A", 401, SUBST_ARYL, hetatm=True),
    ]
    if with_competing_carbonyl:
        lig_atoms.append(
            _atom(nxt(), "C9", "C", "LIG", "A", 401, COMPETING_C, hetatm=True))
    lig = _residue("LIG", "A", 401, lig_atoms, hetatm=True)
    chain = Chain(chain_id="A", residues=[tyr, lys, cof, lig])
    return Structure(structure_id=structure_id, chains=[chain],
                     source_format="mmcif", models_present=[1], model_selected=1)


def _provenance() -> TemplateProvenance:
    return TemplateProvenance(
        source_type=TemplateSourceType.EXPERIMENTAL_STRUCTURE,
        identifiers=["1E3W"], curated_by="test",
    )


def make_template(
    *,
    calibrated: bool = True,
    theoretical: bool = False,
    required_state: CofactorState = CofactorState.REDUCED,
    ligand_codes: Sequence[str] = ("NDP",),
    hydride_window: tuple[float, float] = (2.8, 4.0),
) -> CatalyticTemplate:
    """An SDR-style catalytic template whose windows bracket the geometry above."""
    calib = ["1E3W", "4NBU"] if calibrated else []
    provenance = (
        TemplateProvenance(source_type=TemplateSourceType.THEORETICAL_MODEL,
                           identifiers=["theozyme-draft-1"], curated_by="test")
        if theoretical else _provenance()
    )
    return CatalyticTemplate(
        template_id="ct_sdr_ketoreductase_v1",
        family_name="SDR",
        mechanism_summary="Tyr-assisted hydride transfer from NADPH C4 to the ketone.",
        catalytic_residues=[
            {"label": "catalytic_Tyr", "residue_types": ["TYR"],
             "role": "proton donor to the developing alkoxide",
             "functional_atoms": ["OH"], "evidence": "PDB 1E3W"},
            {"label": "catalytic_Lys", "residue_types": ["LYS"],
             "role": "lowers the tyrosine pKa and binds the nicotinamide ribose",
             "functional_atoms": ["NZ"], "evidence": "PDB 1E3W"},
        ],
        required_cofactor="NADPH",
        required_cofactor_state=required_state,
        cofactor_ligand_codes=list(ligand_codes),
        geometry_constraints=[
            GeometryConstraint(
                name="hydride_transfer_distance", kind="distance",
                atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
                min_value=hydride_window[0], max_value=hydride_window[1],
                severity="gating", calibrated_on=calib, source="PDB 1E3W",
            ),
            GeometryConstraint(
                name="burgi_dunitz_angle", kind="angle",
                atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
                atom_c="substrate.carbonyl_O", unit="degree",
                min_value=95.0, max_value=120.0,
                severity="gating", calibrated_on=calib, source="PDB 1E3W",
            ),
            GeometryConstraint(
                name="oxyanion_Tyr_OH", kind="distance",
                atom_a="protein.catalytic_Tyr.OH", atom_b="substrate.carbonyl_O",
                min_value=2.4, max_value=3.2,
                severity="gating", calibrated_on=calib, source="PDB 1E3W",
            ),
            GeometryConstraint(
                name="cofactor_anchor_Lys_NZ", kind="distance",
                atom_a="protein.catalytic_Lys.NZ",
                atom_b="cofactor.hydride_donor_C4",
                min_value=3.0, max_value=9.0,
                severity="scoring", calibrated_on=calib, source="PDB 1E3W",
            ),
        ],
        reference_structures=["1E3W"],
        provenance=provenance,
    )


def make_task(target: Stereochemistry = Stereochemistry.S,
              cofactor_state: CofactorState = CofactorState.REDUCED) -> TaskSpec:
    """A confirmed reaction spec for acetophenone -> (S)-1-phenylethanol."""
    task = TaskSpec(
        task_id="t1",
        reaction=ReactionSpec(
            substrate=SubstrateSpec(name="acetophenone",
                                    isomeric_smiles="CC(=O)c1ccccc1",
                                    is_prochiral=True),
            product=ProductSpec(name="1-phenylethanol",
                                isomeric_smiles="C[C@H](O)c1ccccc1",
                                target_stereochemistry=target,
                                creates_new_stereocenter=True),
            atom_mapped_reaction_smiles="[CH3:1][C:2](=[O:3])[c:4]>>[CH3:1][CH:2]([OH:3])[c:4]",
        ),
        conditions=Conditions(cofactor_options=[
            CofactorSpec(name="NADPH", state=cofactor_state, ligand_code="NDP",
                         transfer_atom=AtomRef(atom_map_id=100, element="C",
                                               role="hydride_donor_C4"),
                         source=LigandSource.EXPERIMENTAL_OBSERVED),
        ]),
    )
    task.approval.reaction_spec_confirmed = True
    return task


def make_binding(
    pose_id: str = "p1",
    *,
    cofactor_component: str = "NDP",
    bind_tyr: bool = True,
    tyr_expected_resname: str | None = "TYR",
    tyr_resseq: int = 155,
    competing: bool = False,
) -> PoseBinding:
    """The role-token -> coordinate-atom map the pose builder would supply."""
    protein: dict[str, ProteinAtomRef] = {
        "catalytic_Lys.NZ": ProteinAtomRef(chain="A", resseq=159, atom="NZ",
                                           expected_resname="LYS"),
    }
    if bind_tyr:
        protein["catalytic_Tyr.OH"] = ProteinAtomRef(
            chain="A", resseq=tyr_resseq, atom="OH",
            expected_resname=tyr_expected_resname,
        )
    return PoseBinding(
        pose_id=pose_id,
        substrate=ResidueSelector(chain="A", resname="LIG", resseq=401),
        substrate_atoms={"electrophile": "C1", "carbonyl_O": "O1",
                         "methyl": "C2", "aryl": "C3"},
        cofactor=ResidueSelector(chain="A", resname=cofactor_component, resseq=301),
        cofactor_atoms={"hydride_donor_C4": "C4N"},
        protein_atoms=protein,
        competing_electrophiles={"second_carbonyl_C9": "C9"} if competing else {},
        reactive_atom_role="electrophile",
        carbonyl_oxygen_role="carbonyl_O",
        prochiral_substituent_roles=["aryl", "methyl"],
        hydride_donor_role="hydride_donor_C4",
    )


def make_candidate(
    candidate_id: str = "cand1",
    *,
    template_id: str | None = "ct_sdr_ketoreductase_v1",
    poses: Iterable[ComplexPose] = (),
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        sequence_record=SequenceRecord(
            candidate_id=candidate_id,
            sequence="MKAAVLTGAASGIGLATAKRFAEEGAKVVLADLNEEGAKAVAEEINKAYGEG",
            accession="TEST0001", source_database="uniprot",
            database_version="2024_01", search_method="blastp",
            seed_accession="P00001", percent_identity=48.0, is_fragment=False,
        ),
        family=FamilyAnnotation(family_name="SDR",
                                confidence=ConfidenceLevel.MODERATE),
        catalytic_mapping=CatalyticMapping(
            catalytic_template_id=template_id,
            role_to_residue={"catalytic_Tyr": "Y155", "catalytic_Lys": "K159"},
        ),
        poses=list(poses),
    )


def make_pose(
    pose_id: str = "p1",
    *,
    cofactor_state: CofactorState = CofactorState.UNKNOWN,
    restrained: Sequence[str] = (),
    is_valid: bool = True,
    invalid_reason: str | None = None,
) -> ComplexPose:
    return ComplexPose(
        pose_id=pose_id, method="template_docking", rank=1,
        substrate_present=True, cofactor_present=True,
        cofactor_state=cofactor_state,
        substrate_source=LigandSource.DOCKING_PREDICTED,
        cofactor_source=LigandSource.HOMOLOGY_TRANSPLANTED,
        restrained_constraints=list(restrained),
        is_valid=is_valid, invalid_reason=invalid_reason,
    )


CIP = CIPPriority(
    ranks={"carbonyl_O": 1, "aryl": 2, "methyl": 3},
    source="operator:test-suite", incoming_rank=4, scope="product",
)


class _Harness(unittest.TestCase):
    """Shared set-up: a temporary working directory and a confirmed task."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        self.iface = EvaluateCatalysis()

    def context(self, task: TaskSpec | None = None, **config: Any) -> RunContext:
        task = task or make_task()
        return RunContext(
            task=task, workdir=self.workdir / "run",
            manifest=RunManifest(run_id="r1", task_id=task.task_id),
            policy=ExecutionPolicy(allow_network=False),
            config=dict(config),
        )

    def run_step(self, candidates, bindings, structures, template=None,
                 task=None, **kwargs):
        ctx = self.context(task)
        return ctx, self.iface.run(
            ctx, candidates=candidates, bindings=bindings, structures=structures,
            catalytic_templates=template or make_template(), cip_ranks=CIP,
            **kwargs,
        )

    @staticmethod
    def pose_eval(result, candidate_id: str = "cand1",
                  pose_id: str = "p1") -> dict[str, Any]:
        for evaluation in result.data["evaluations"]:
            if evaluation["candidate_id"] != candidate_id:
                continue
            for pose in evaluation["poses"]:
                if pose["pose_id"] == pose_id:
                    return pose
        raise AssertionError(f"no pose {candidate_id}/{pose_id} in the result")

    @staticmethod
    def candidate_eval(result, candidate_id: str = "cand1") -> dict[str, Any]:
        for evaluation in result.data["evaluations"]:
            if evaluation["candidate_id"] == candidate_id:
                return evaluation
        raise AssertionError(f"no evaluation for {candidate_id}")

    @staticmethod
    def codes(result) -> list[str]:
        return [f.code for f in result.qc_flags]


# ==========================================================================
# The happy path, so the failure tests mean something
# ==========================================================================

class TestCompetentPose(_Harness):
    """A pose that genuinely satisfies the template, to anchor the rest."""

    def test_a_competent_pose_satisfies_every_gating_constraint(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()})
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"], PoseOutcome.MECHANISM_SATISFIED.value,
                         msg=pose["reason"])
        self.assertTrue(pose["gating_passed"])
        self.assertAlmostEqual(pose["measurements"]["hydride_transfer_distance"],
                               3.4, places=6)
        self.assertAlmostEqual(pose["measurements"]["burgi_dunitz_angle"],
                               107.0, places=4)
        self.assertAlmostEqual(pose["measurements"]["oxyanion_Tyr_OH"],
                               2.70, places=6)

    def test_the_face_call_is_directional_and_carries_no_numeric_ee(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()})
        evaluation = self.candidate_eval(result)
        self.assertEqual(evaluation["poses"][0]["face"], "re")
        self.assertEqual(evaluation["poses"][0]["product_configuration"], "S")
        self.assertEqual(evaluation["stereo"]["call"], "favors_target")
        self.assertIsNone(evaluation["stereo"]["predicted_ee_pct"])
        self.assertIsNone(evaluation["stereo"]["calibration_source"])
        self.assertIsNone(candidate.stereo.predicted_ee_pct)

    def test_an_opposite_target_flips_the_call_without_changing_the_geometry(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        _, result = self.run_step(
            [candidate], [make_binding()], {"p1": make_structure()},
            task=make_task(target=Stereochemistry.R),
        )
        self.assertEqual(self.candidate_eval(result)["stereo"]["call"],
                         "favors_opposite")

    def test_the_scorecard_has_no_total_score_column(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        ctx, result = self.run_step([candidate], [make_binding()],
                                    {"p1": make_structure()})
        path = Path(result.artifact("candidate_scorecards").path)
        header = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
        for forbidden in ("total", "score", "weighted", "composite"):
            self.assertFalse(
                [c for c in header if forbidden in c.lower()
                 and not c.startswith("level:") and not c.startswith("value:")
                 and c not in ("stereo_call",)],
                msg=f"a column named like a total score appeared: {header}",
            )
        self.assertIsNone(result.provenance.parameters["weighted_total_score"])

    def test_both_artifacts_are_written_with_one_row_per_pose(self) -> None:
        candidate = make_candidate(poses=[
            make_pose("p1", cofactor_state=CofactorState.REDUCED),
            make_pose("p2", cofactor_state=CofactorState.REDUCED),
        ])
        _, result = self.run_step(
            [candidate],
            [make_binding("p1"), make_binding("p2")],
            {"p1": make_structure(), "p2": make_structure()},
        )
        geometry = Path(result.artifact("catalytic_geometry").path)
        lines = geometry.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)           # header + two poses
        header = lines[0].split("\t")
        for required in ("measure:hydride_transfer_distance",
                         "satisfied:hydride_transfer_distance",
                         "circular_constraints", "clash_count"):
            self.assertIn(required, header)
        self.assertTrue(Path(result.artifact("candidate_scorecards").path).is_file())


# ==========================================================================
# 1. The wrong-oxidation-state cofactor
# ==========================================================================

class TestWrongCofactorOxidationState(_Harness):
    """NADP+ where the mechanism needs NADPH: an input defect, not a geometry fail."""

    def _run_with_oxidised_cofactor(self):
        candidate = make_candidate(poses=[make_pose()])   # state left UNKNOWN
        template = make_template(ligand_codes=("NDP", "NAP"))
        return candidate, self.run_step(
            [candidate],
            [make_binding(cofactor_component="NAP")],
            {"p1": make_structure(cofactor_component="NAP")},
            template=template,
        )

    def test_the_oxidised_cofactor_is_caught_although_the_geometry_is_perfect(self) -> None:
        candidate, (_, result) = self._run_with_oxidised_cofactor()
        pose = self.pose_eval(result)
        # The geometry is untouched: every window is still satisfied.
        self.assertTrue(pose["satisfied"]["hydride_transfer_distance"])
        self.assertTrue(pose["satisfied"]["burgi_dunitz_angle"])
        # And yet the pose is an input error, not a satisfied pose.
        self.assertEqual(pose["outcome"], PoseOutcome.INPUT_ERROR.value)
        self.assertIn("oxidized", pose["reason"])
        self.assertIn("no hydride to transfer", pose["reason"])

    def test_the_state_is_inferred_from_the_component_id_not_assumed(self) -> None:
        _, (_, result) = self._run_with_oxidised_cofactor()
        self.assertIn("cofactor_wrong_oxidation_state", self.codes(result))
        flag = [f for f in result.qc_flags
                if f.code == "cofactor_wrong_oxidation_state"][0]
        self.assertIs(flag.severity, Severity.BLOCKER)
        self.assertIn("NAP", flag.message)

    def test_it_is_routed_to_input_errors_and_not_to_a_geometry_axis(self) -> None:
        candidate, (_, result) = self._run_with_oxidised_cofactor()
        self.assertTrue(candidate.input_errors)
        self.assertFalse(candidate.passes_gates)
        self.assertIs(candidate.scorecard["inputs_free_of_defects"].gate_passed, False)
        # The pose contributed no geometric verdict at all.
        self.assertIsNone(candidate.geometry[0].gating_passed)
        self.assertEqual(self.candidate_eval(result)["counts"]["input_error"], 1)
        self.assertEqual(self.candidate_eval(result)["counts"]["mechanism_violated"], 0)

    def test_a_wrong_input_is_never_counted_as_a_tested_pose(self) -> None:
        _, (_, result) = self._run_with_oxidised_cofactor()
        evaluation = self.candidate_eval(result)
        self.assertFalse(evaluation["was_tested"])
        self.assertIsNone(evaluation["robustness_G"])

    def test_an_unknown_state_is_flagged_rather_than_assumed_reduced(self) -> None:
        candidate = make_candidate(poses=[make_pose()])
        # "XYZ" is in no oxidation-state table, so the state stays UNKNOWN.
        _, result = self.run_step(
            [candidate], [make_binding(cofactor_component="XYZ")],
            {"p1": make_structure(cofactor_component="XYZ")},
            template=make_template(ligand_codes=("NDP", "XYZ")),
        )
        self.assertIn("cofactor_state_unknown", self.codes(result))
        pose = self.pose_eval(result)
        # Every window is satisfied and the pose is still not promoted: an
        # unrecorded oxidation state is not the one the mechanism needs.
        self.assertTrue(pose["satisfied"]["hydride_transfer_distance"])
        self.assertEqual(pose["outcome"], PoseOutcome.NOT_MEASURABLE.value)
        self.assertIn("never tested", pose["reason"])
        self.assertFalse(candidate.input_errors,
                         "an unknown state is unchecked, not a defect")
        self.assertFalse(self.candidate_eval(result)["was_tested"])


# ==========================================================================
# 2. The non-target carbonyl in the reactive position
# ==========================================================================

class TestChemoselectivity(_Harness):
    """The wrong carbonyl presented to the donor is not a target-reaction pose."""

    #: What a calibrated margin looks like: a statement of what it was fitted
    #: to. The value is the same 0.2 A either way; the difference is whether
    #: anybody measured it.
    SOURCE = "positional scatter of the in-house pose generator, n=240 redocks"

    def _run_with_competing_carbonyl(self, margin: float = 0.2,
                                     source: str = SOURCE):
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        return candidate, self.run_step(
            [candidate], [make_binding(competing=True)],
            {"p1": make_structure(with_competing_carbonyl=True)},
            competing_group_margin_angstrom=margin,
            competing_group_margin_source=source,
        )

    def test_a_closer_non_target_carbonyl_rejects_the_pose(self) -> None:
        _, (_, result) = self._run_with_competing_carbonyl()
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"], PoseOutcome.MECHANISM_VIOLATED.value)
        self.assertIn("second_carbonyl_C9", pose["reason"])
        self.assertIn("different product", pose["reason"])
        self.assertIs(pose["gating_passed"], False)

    def test_an_uncalibrated_margin_reports_but_does_not_reject(self) -> None:
        """The margin is the whole content of the comparison.

        Both distances carry the pose generator's positional scatter, so "is
        the competitor closer" has no answer until somebody says how much
        closer counts. Rejecting on the module default turned a number nobody
        measured into a hard computational negative -- while every template
        window in this module with the same provenance is explicitly not
        allowed to reject.
        """
        _, (_, result) = self._run_with_competing_carbonyl(source="")
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"],
                         PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW.value)
        self.assertIsNot(pose["gating_passed"], False)

    def test_an_uncalibrated_displacement_is_still_measured_and_reported(self) -> None:
        """Not rejecting is not the same as not noticing."""
        _, (_, result) = self._run_with_competing_carbonyl(source="")
        pose = self.pose_eval(result)
        self.assertIn("second_carbonyl_C9", pose["reason"])
        self.assertIs(pose["target_in_reactive_position"], False)

    def test_an_uncalibrated_displacement_is_not_a_computational_negative(self) -> None:
        _, (_, result) = self._run_with_competing_carbonyl(source="")
        pose = self.pose_eval(result)
        self.assertNotEqual(pose["record_outcome"],
                            OutcomeClass.COMPUTATIONAL_NEGATIVE.value)

    def test_every_distance_and_angle_still_passes_which_is_the_point(self) -> None:
        _, (_, result) = self._run_with_competing_carbonyl()
        pose = self.pose_eval(result)
        self.assertTrue(pose["satisfied"]["hydride_transfer_distance"])
        self.assertTrue(pose["satisfied"]["burgi_dunitz_angle"])
        self.assertTrue(pose["satisfied"]["oxyanion_Tyr_OH"])

    def test_the_rejection_is_a_mechanism_negative_not_a_modelling_failure(self) -> None:
        _, (_, result) = self._run_with_competing_carbonyl()
        pose = self.pose_eval(result)
        self.assertEqual(pose["record_outcome"],
                         OutcomeClass.COMPUTATIONAL_NEGATIVE.value)
        evaluation = self.candidate_eval(result)
        self.assertTrue(evaluation["was_tested"])
        self.assertEqual(evaluation["robustness_G"], 0.0)

    def test_a_margin_wider_than_the_gap_leaves_the_target_in_place(self) -> None:
        _, (_, result) = self._run_with_competing_carbonyl(margin=2.0)
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"], PoseOutcome.MECHANISM_SATISFIED.value)

    def test_an_undeclared_competing_group_is_untested_not_passed(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        _, result = self.run_step(
            [candidate], [make_binding(competing=False)],
            {"p1": make_structure(with_competing_carbonyl=True)},
        )
        self.assertIn("chemoselectivity_untested", self.codes(result))
        check = ChemoselectivityCheck(tested=False, target_distance_A=3.4)
        self.assertIsNone(check.target_in_reactive_position)


# ==========================================================================
# 3. The fully restrained pose
# ==========================================================================

class TestCircularEvidence(_Harness):
    """A pose built to satisfy the constraints cannot corroborate itself."""

    def _run_fully_restrained(self):
        every = ["hydride_transfer_distance", "burgi_dunitz_angle",
                 "oxyanion_Tyr_OH", "cofactor_anchor_Lys_NZ"]
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED, restrained=every)])
        return candidate, self.run_step([candidate], [make_binding()],
                                        {"p1": make_structure()})

    def test_a_fully_restrained_pose_yields_zero_independent_evidence(self) -> None:
        _, (_, result) = self._run_fully_restrained()
        pose = self.pose_eval(result)
        self.assertEqual(pose["independent_satisfied"], 0)
        self.assertEqual(pose["independent_total"], 0)
        self.assertTrue(pose["entirely_circular"])
        self.assertEqual(len(pose["circular_constraints"]), 4)

    def test_it_says_so_loudly_rather_than_silently(self) -> None:
        _, (_, result) = self._run_fully_restrained()
        self.assertIn("evidence_entirely_circular", self.codes(result))
        self.assertIn("evidence_entirely_circular",
                      [u.code for u in result.uncertainty])
        self.assertTrue(any("release_restraints" in str(n.params)
                            for n in result.next_actions))

    def test_the_geometry_axis_is_not_strong_on_circular_evidence(self) -> None:
        candidate, (_, result) = self._run_fully_restrained()
        axis = candidate.scorecard["catalytic_geometry"]
        self.assertIsNot(axis.level, ConfidenceLevel.STRONG)
        self.assertIn("restrained", axis.basis)

    def test_a_partly_restrained_pose_keeps_its_independent_constraints(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED,
            restrained=["hydride_transfer_distance"])])
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()})
        pose = self.pose_eval(result)
        self.assertFalse(pose["entirely_circular"])
        self.assertEqual(pose["circular_constraints"],
                         ["hydride_transfer_distance"])
        self.assertEqual(pose["independent_satisfied"], 3)
        self.assertEqual(pose["independent_total"], 3)

    def test_a_restraint_name_that_matches_nothing_is_reported(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED,
            restrained=["a_name_the_template_does_not_use"])])
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()})
        self.assertIn("restraint_name_mismatch", self.codes(result))


# ==========================================================================
# 4. The unmeasurable constraint
# ==========================================================================

class TestUnmeasurableConstraint(_Harness):
    """An unresolvable atom yields None -- not a pass, and not a failure."""

    def _run_without_tyrosine(self):
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        return candidate, self.run_step(
            [candidate], [make_binding(bind_tyr=False)],
            {"p1": make_structure()},
        )

    def test_an_unresolvable_role_leaves_the_measurement_none(self) -> None:
        _, (_, result) = self._run_without_tyrosine()
        pose = self.pose_eval(result)
        self.assertIsNone(pose["measurements"]["oxyanion_Tyr_OH"])
        self.assertIsNone(pose["satisfied"]["oxyanion_Tyr_OH"])
        self.assertIsNot(pose["satisfied"]["oxyanion_Tyr_OH"], False)
        self.assertIsNot(pose["satisfied"]["oxyanion_Tyr_OH"], True)

    def test_the_pose_is_unmeasurable_rather_than_violating(self) -> None:
        _, (_, result) = self._run_without_tyrosine()
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"], PoseOutcome.NOT_MEASURABLE.value)
        self.assertIsNone(pose["gating_passed"])
        self.assertIn("oxyanion_Tyr_OH", pose["reason"])

    def test_the_other_constraints_are_still_measured(self) -> None:
        _, (_, result) = self._run_without_tyrosine()
        pose = self.pose_eval(result)
        self.assertAlmostEqual(pose["measurements"]["hydride_transfer_distance"],
                               3.4, places=6)
        self.assertTrue(pose["satisfied"]["burgi_dunitz_angle"])

    def test_the_reason_the_role_did_not_resolve_is_recorded(self) -> None:
        _, (_, result) = self._run_without_tyrosine()
        pose = self.pose_eval(result)
        joined = " ".join(pose["unresolved_roles"].values())
        self.assertIn("catalytic_Tyr.OH", " ".join(pose["unresolved_roles"]))
        self.assertTrue(joined.strip())

    def test_sequence_numbering_used_as_author_numbering_is_caught(self) -> None:
        """Expecting TYR where the file holds LYS is the off-by-N signature."""
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        _, result = self.run_step(
            [candidate],
            [make_binding(tyr_resseq=159)],   # 159 is the lysine
            {"p1": make_structure()},
        )
        pose = self.pose_eval(result)
        self.assertIsNone(pose["satisfied"]["oxyanion_Tyr_OH"])
        reasons = " ".join(pose["unresolved_roles"].values())
        self.assertIn("off-by-N", reasons)

    def test_the_unevaluated_constraint_does_not_make_the_axis_a_negative(self) -> None:
        candidate, (_, result) = self._run_without_tyrosine()
        axis = candidate.scorecard["catalytic_geometry"]
        self.assertIs(axis.level, ConfidenceLevel.INSUFFICIENT)


# ==========================================================================
# 5. A modelling failure is not an experimental negative
# ==========================================================================

class TestModellingFailureIsNotInactivity(_Harness):
    """The distinction the whole module is organised around."""

    def test_a_pose_with_no_binding_is_unmeasurable(self) -> None:
        candidate = make_candidate(poses=[make_pose()])
        _, result = self.run_step([candidate], [], {})
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"], PoseOutcome.NOT_MEASURABLE.value)
        self.assertEqual(pose["record_outcome"],
                         OutcomeClass.COMPUTATIONAL_FAILURE.value)

    def test_an_invalid_pose_carries_the_builders_reason_forward(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            is_valid=False, invalid_reason="the sampler produced no converged pose")])
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()})
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"], PoseOutcome.NOT_MEASURABLE.value)
        self.assertIn("no converged pose", pose["reason"])

    def test_no_pose_outcome_can_become_an_experimental_result(self) -> None:
        for outcome in PoseOutcome:
            record = outcome.as_record_outcome()
            self.assertFalse(
                record.is_experimental,
                msg=f"{outcome.value} mapped to the experimental {record.value}",
            )
        self.assertIs(PoseOutcome.NOT_MEASURABLE.as_record_outcome(),
                      OutcomeClass.COMPUTATIONAL_FAILURE)
        self.assertIs(PoseOutcome.MECHANISM_VIOLATED.as_record_outcome(),
                      OutcomeClass.COMPUTATIONAL_NEGATIVE)
        self.assertIs(PoseOutcome.MECHANISM_SATISFIED.as_record_outcome(),
                      OutcomeClass.NOT_TESTED)

    def test_a_failed_run_and_a_genuine_zero_are_distinguishable(self) -> None:
        """The case a naive pipeline collapses into one column of zeros."""
        failed = make_candidate("failed", poses=[make_pose("pf")])
        measured = make_candidate("measured", poses=[make_pose(
            "pm", cofactor_state=CofactorState.REDUCED)])
        # A template whose hydride window the pose cannot meet: a real negative.
        narrow = make_template(hydride_window=(1.0, 2.0))
        ctx = self.context()
        result = self.iface.run(
            ctx,
            candidates=[failed, measured],
            bindings=[make_binding("pm")],           # "pf" has no binding at all
            structures={"pm": make_structure()},
            catalytic_templates=narrow, cip_ranks=CIP,
        )
        failed_eval = self.candidate_eval(result, "failed")
        measured_eval = self.candidate_eval(result, "measured")

        self.assertIsNone(failed_eval["robustness_G"])
        self.assertFalse(failed_eval["was_tested"])
        self.assertEqual(failed_eval["counts"]["not_measurable"], 1)
        self.assertEqual(failed_eval["counts"]["mechanism_violated"], 0)

        self.assertEqual(measured_eval["robustness_G"], 0.0)
        self.assertTrue(measured_eval["was_tested"])
        self.assertEqual(measured_eval["counts"]["mechanism_violated"], 1)
        self.assertEqual(measured_eval["counts"]["not_measurable"], 0)

    def test_a_modelling_failure_is_flagged_as_not_a_negative(self) -> None:
        candidate = make_candidate(poses=[make_pose()])
        _, result = self.run_step([candidate], [], {})
        flags = [f for f in result.qc_flags
                 if f.code == "modelling_failure_not_a_negative"]
        self.assertTrue(flags)
        self.assertIs(flags[0].severity, Severity.INFO)
        self.assertIn("says nothing about the enzyme", flags[0].message)

    def test_untested_candidates_make_the_step_partial_and_ask_for_a_rebuild(self) -> None:
        candidate = make_candidate(poses=[make_pose()])
        _, result = self.run_step([candidate], [], {})
        self.assertIs(result.status, Status.PARTIAL)
        self.assertIn("candidates_not_tested", self.codes(result))
        self.assertTrue(any(n.action == "model_complexes" for n in result.next_actions))

    def test_a_candidate_with_no_template_is_unmeasured_not_rejected(self) -> None:
        candidate = make_candidate(template_id=None, poses=[make_pose()])
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()})
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"], PoseOutcome.NOT_MEASURABLE.value)
        self.assertIn("no_catalytic_template", [u.code for u in result.uncertainty])


# ==========================================================================
# 6. An untrustworthy template downgrades rather than rejects
# ==========================================================================

class TestUncalibratedTemplate(_Harness):
    """A window nobody fitted may not throw a family away."""

    def test_an_uncalibrated_failing_window_does_not_reject(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        template = make_template(calibrated=False, hydride_window=(1.0, 2.0))
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()}, template=template)
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"],
                         PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW.value)
        self.assertIsNone(pose["gating_passed"])
        self.assertEqual(self.candidate_eval(result)["counts"]["mechanism_violated"], 0)

    def test_the_uncertainty_is_raised_instead(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        template = make_template(calibrated=False, hydride_window=(1.0, 2.0))
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()}, template=template)
        self.assertIn("uncalibrated_geometry_window",
                      [u.code for u in result.uncertainty])
        self.assertIn("template_window_provisional", self.codes(result))

    def test_a_pass_against_an_uncalibrated_window_is_downgraded(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        template = make_template(calibrated=False)
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()}, template=template)
        axis = candidate.scorecard["catalytic_geometry"]
        self.assertIn("downgraded", axis.basis)
        self.assertIsNot(axis.level, ConfidenceLevel.STRONG)

    def test_a_theoretical_template_is_provisional_however_calibrated(self) -> None:
        template = make_template(calibrated=True, theoretical=True)
        self.assertIs(template_authority(template),
                      WindowAuthority.THEORETICAL_MODEL)
        self.assertFalse(WindowAuthority.THEORETICAL_MODEL.may_reject)
        self.assertTrue(WindowAuthority.CALIBRATED.may_reject)


# ==========================================================================
# The 10 angstrom screen, and the other refusals
# ==========================================================================

class TestPocketLocalisationIsQCOnly(_Harness):
    """The familiar number lives here and decides nothing."""

    def test_the_default_window_is_ten_angstroms_from_the_module(self) -> None:
        threshold, source = pocket_localisation_window(make_template())
        self.assertEqual(threshold, DEFAULT_POCKET_LOCALISATION_A)
        self.assertIn("needs per-family calibration", source)

    def test_a_template_constraint_overrides_the_default(self) -> None:
        template = make_template()
        template.geometry_constraints.append(GeometryConstraint(
            name="pocket_localisation", kind="distance",
            atom_a="substrate.electrophile", atom_b="cofactor.hydride_donor_C4",
            min_value=0.0, max_value=7.5, severity="advisory",
            calibrated_on=["1E3W"], source="PDB 1E3W",
        ))
        threshold, source = pocket_localisation_window(template)
        self.assertEqual(threshold, 7.5)
        self.assertIn("template:ct_sdr_ketoreductase_v1", source)

    def test_it_is_measured_from_the_reactive_atom_to_catalytic_atoms(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()})
        row = self._geometry_row(result)
        self.assertAlmostEqual(float(row["pocket_localisation_A"]), 3.4, places=4)
        self.assertEqual(row["pocket_within"], "true")
        self.assertIn("hydride_donor_C4", row["pocket_nearest_role"])

    def test_failing_the_screen_flags_but_does_not_reject(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        _, result = self.run_step(
            [candidate], [make_binding()], {"p1": make_structure()},
            pocket_localisation_angstrom=1.0,
        )
        self.assertIn("pocket_localisation_outside_window", self.codes(result))
        flag = [f for f in result.qc_flags
                if f.code == "pocket_localisation_outside_window"][0]
        self.assertIs(flag.severity, Severity.WARN)
        self.assertIn("QC screen", flag.message)
        # The verdict is unchanged: the mechanism constraints decide.
        self.assertEqual(self.pose_eval(result)["outcome"],
                         PoseOutcome.MECHANISM_SATISFIED.value)

    @staticmethod
    def _geometry_row(result) -> dict[str, str]:
        path = Path(result.artifact("catalytic_geometry").path)
        lines = path.read_text(encoding="utf-8").splitlines()
        return dict(zip(lines[0].split("\t"), lines[1].split("\t")))


class TestRefusals(_Harness):
    """What the interface will not do, whatever it is asked."""

    def test_it_refuses_to_submit_a_sequence_to_an_external_service(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        ctx = self.context()
        result = self.iface.run(
            ctx, candidates=[candidate], bindings=[make_binding()],
            structures={"p1": make_structure()},
            catalytic_templates=make_template(),
            submit_to="https://example.org/fold",
        )
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("external_submission_refused", self.codes(result))
        self.assertIn("allow_network=False", result.message)

    def test_it_refuses_to_run_without_candidates(self) -> None:
        ctx = self.context()
        result = self.iface.run(ctx, candidates=[])
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("no_candidates", self.codes(result))

    def test_an_unconfirmed_reaction_spec_blocks_the_step(self) -> None:
        task = make_task()
        task.approval.reaction_spec_confirmed = False
        ctx = self.context(task)
        result = self.iface.run(ctx, candidates=[make_candidate()])
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("approval_required", self.codes(result))

    def test_an_unresolved_substrate_structure_blocks_the_step(self) -> None:
        task = make_task()
        task.reaction.substrate.isomeric_smiles = None
        ctx = self.context(task)
        result = self.iface.run(ctx, candidates=[make_candidate()])
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("unresolved_fields", self.codes(result))

    def test_without_cip_ranks_no_configuration_is_invented(self) -> None:
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        ctx = self.context()
        result = self.iface.run(
            ctx, candidates=[candidate], bindings=[make_binding()],
            structures={"p1": make_structure()},
            catalytic_templates=make_template(),   # no cip_ranks
        )
        evaluation = self.candidate_eval(result)
        self.assertIsNone(evaluation["poses"][0]["product_configuration"])
        self.assertEqual(evaluation["stereo"]["call"], "insufficient_evidence")
        self.assertIn("cip_ranks_absent", [u.code for u in result.uncertainty])


# ==========================================================================
# Unit-level checks on the pieces
# ==========================================================================

class TestRoleContext(_Harness):
    """The binding layer reports gaps instead of substituting for them."""

    def test_an_ambiguous_selector_resolves_to_nothing(self) -> None:
        structure = make_structure()
        binding = make_binding()
        binding.substrate = ResidueSelector(chain="A")   # matches every residue
        context = build_role_context(structure, binding, make_template())
        self.assertIsNone(context.substrate_residue)
        self.assertTrue(any("residues match the substrate selector" in p
                            for p in context.problems))

    def test_a_missing_atom_name_is_named_in_the_reason(self) -> None:
        structure = make_structure()
        binding = make_binding()
        binding.substrate_atoms = dict(binding.substrate_atoms)
        binding.substrate_atoms["electrophile"] = "CX"
        context = build_role_context(structure, binding, make_template())
        resolver = context.resolver()
        self.assertFalse(resolver.resolve("substrate.electrophile").found)
        self.assertIn("has no atom named 'CX'",
                      context.gaps()["substrate.electrophile"])

    def test_the_donor_atom_name_falls_back_only_for_known_components(self) -> None:
        structure = make_structure(cofactor_component="NDP")
        binding = make_binding()
        binding.cofactor_atoms = {}
        context = build_role_context(structure, binding, make_template())
        self.assertIn("hydride_donor_C4", context.cofactor_role_atoms)
        self.assertTrue(any("Chemical Component Dictionary" in p
                            for p in context.problems))

        unknown = make_structure(cofactor_component="XYZ")
        unknown_binding = make_binding(cofactor_component="XYZ")
        unknown_binding.cofactor_atoms = {}
        context2 = build_role_context(unknown, unknown_binding, make_template())
        self.assertNotIn("hydride_donor_C4", context2.cofactor_role_atoms)
        self.assertIn("will not be guessed",
                      context2.gaps()["cofactor.hydride_donor_C4"])


class TestAspectClassification(unittest.TestCase):
    """Constraint grouping is for reporting and never touches pass/fail."""

    def test_each_mechanism_question_is_recognised(self) -> None:
        template = make_template()
        by_name = {c.name: classify_aspect(c) for c in template.geometry_constraints}
        self.assertEqual(by_name["hydride_transfer_distance"], "hydride_transfer")
        self.assertEqual(by_name["burgi_dunitz_angle"], "approach_trajectory")
        self.assertEqual(by_name["oxyanion_Tyr_OH"], "oxyanion_stabilisation")
        self.assertEqual(by_name["cofactor_anchor_Lys_NZ"], "cofactor_placement")


class TestCountsStaySeparate(unittest.TestCase):
    """The roll-up keeps the three failure kinds in three different fields."""

    def test_counts_do_not_leak_into_each_other(self) -> None:
        def pose(pose_id: str, outcome: PoseOutcome) -> PoseEvaluation:
            return PoseEvaluation(candidate_id="c", pose_id=pose_id,
                                  method="template_docking", outcome=outcome)

        evaluation = CandidateEvaluation(
            candidate_id="c", template_id="t",
            authority=WindowAuthority.CALIBRATED,
            poses=[
                pose("a", PoseOutcome.MECHANISM_SATISFIED),
                pose("b", PoseOutcome.MECHANISM_VIOLATED),
                pose("c", PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW),
                pose("d", PoseOutcome.NOT_MEASURABLE),
                pose("e", PoseOutcome.INPUT_ERROR),
            ],
        )
        self.assertEqual(evaluation.n_poses, 5)
        self.assertEqual(evaluation.n_decided, 2)
        self.assertEqual(evaluation.n_satisfied, 1)
        self.assertEqual(evaluation.n_violated, 1)
        self.assertEqual(evaluation.n_provisional, 1)
        self.assertEqual(evaluation.n_not_measurable, 1)
        self.assertEqual(evaluation.n_input_error, 1)
        self.assertTrue(evaluation.was_tested)


if __name__ == "__main__":
    unittest.main(verbosity=2)
