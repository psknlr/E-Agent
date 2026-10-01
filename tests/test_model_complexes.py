"""Tests for the ``model_complexes`` interface.

The fixtures are the same 30-residue synthetic enzyme used by the structure
tests, plus fake runners that write real coordinate files. Faking the runner
rather than the result is deliberate: the properties under test are about what
this module does with coordinates a tool produced -- does it notice the
cofactor is missing, does it keep every pose, does it record the restraints --
and a fake that returned ready-made ``ComplexPose`` objects would test none of
them.

The scientific errors under test, in order of how expensive they are to miss:

* a protein+substrate complex built for a cofactor-dependent reaction;
* an oxidised NAD(P)+ accepted as the hydride donor;
* a required catalytic metal silently absent;
* a pocket completed by a subunit that is not in the file;
* one pose kept and the rest discarded;
* an enforced restraint later counted as independent evidence;
* a ranking score read as an activity ranking;
* non-commercial model weights used in a commercial run.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eagent.connectors.base import AccessPolicy, SubmissionAuthorization
from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Severity, Status
from eagent.errors import CircularEvidenceError, LicenseError, TemplateError
from eagent.provenance import RunManifest, sequence_hash
from eagent.schemas import (
    Budget,
    Candidate,
    CatalyticMapping,
    CatalyticTemplate,
    CofactorState,
    ComplexPose,
    FamilyAnnotation,
    GeometryConstraint,
    LigandSource,
    ReactionSpec,
    SequenceRecord,
    SubstrateSpec,
    TaskSpec,
    TemplateProvenance,
    TemplateSourceType,
)
from eagent.science.numbering import one_to_three
from eagent.science.structure_io import read_mmcif, read_pdb
from eagent.tools.model_complexes import (
    ComplexPolicy,
    ComplexRequest,
    JointPredictionJob,
    ModelComplexes,
    RawPose,
    Route,
    ToolKind,
    ToolRegistry,
    ToolRegistryEntry,
    ToolRegistryError,
    UnavailableComplexPredictor,
    UnavailableDockingRunner,
    assert_restraints_recorded,
    catalytic_site_box,
    check_license,
    cluster_poses,
    pose_rmsd,
    trusted_site_atoms,
    validate_assembly,
)
from eagent.tools.prepare_structures import select_chain

CAND = "MKAIVTGGAQGIGRAIAERLAADGYNVAVL"
FIRST_RESSEQ = 101
POCKET_CENTRE_INDEX = 15
SUBSTRATE_SMILES = "CC(=O)c1ccccc1"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _atom_line(serial: int, name: str, resname: str, chain: str, resseq: int,
               x: float, y: float, z: float, element: str,
               bfactor: float = 20.0, hetatm: bool = False) -> str:
    record = "HETATM" if hetatm else "ATOM"
    return (f"{record:<6}{serial:>5} {name:<4}{'':1}{resname:>3} {chain:1}"
            f"{resseq:>4}{'':1}   {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}"
            f"{bfactor:6.2f}{'':10}{element:>2}")


def _receptor_pdb(sequence: str = CAND, chain: str = "A",
                  ligands: tuple[tuple[str, int], ...] = (("NDP", 3),),
                  extra_chain: str | None = None) -> str:
    lines: list[str] = []
    serial = 1
    for i, letter in enumerate(sequence):
        lines.append(_atom_line(serial, "CA", one_to_three(letter), chain,
                                FIRST_RESSEQ + i, 3.0 * i, 0.0, 0.0, "C"))
        serial += 1
    if extra_chain:
        for i, letter in enumerate(sequence):
            lines.append(_atom_line(serial, "CA", one_to_three(letter),
                                    extra_chain, FIRST_RESSEQ + i,
                                    3.0 * i, 20.0, 0.0, "C"))
            serial += 1
    cx = 3.0 * POCKET_CENTRE_INDEX
    for n, (code, n_atoms) in enumerate(ligands):
        resseq = 900 + n
        if n_atoms == 1:
            lines.append(_atom_line(serial, code, code, chain, resseq,
                                    cx, 2.2, 0.0, code, hetatm=True))
            serial += 1
            continue
        for k in range(n_atoms):
            lines.append(_atom_line(serial, "C4N" if k == 0 else f"C{k}", code,
                                    chain, resseq, cx + 0.8 * k, 2.4, 0.4 * k,
                                    "C", hetatm=True))
            serial += 1
    return "\n".join(lines) + "\nEND\n"


def _pose_pdb(receptor_text: str, substrate_code: str = "LIG",
              offset: float = 0.0, n_atoms: int = 3,
              include_substrate: bool = True) -> str:
    """Receptor coordinates plus a substrate placed beside the cofactor."""
    body = receptor_text.replace("END\n", "").rstrip("\n")
    if not include_substrate:
        return body + "\nEND\n"
    cx = 3.0 * POCKET_CENTRE_INDEX
    lines = [body]
    serial = 5000
    for k in range(n_atoms):
        lines.append(_atom_line(serial + k, f"C{k + 1}", substrate_code, "A", 950,
                                cx + 0.9 * k + offset, 4.6, 0.3 * k, "C",
                                hetatm=True))
    return "\n".join(lines) + "\nEND\n"


def _constraints() -> list[GeometryConstraint]:
    return [
        GeometryConstraint(name="hydride_distance", kind="distance",
                           atom_a="cofactor.hydride_donor_C4",
                           atom_b="substrate.electrophile",
                           target=3.5, tolerance=0.6, severity="gating",
                           calibrated_on=["1XYZ"], source="test"),
        GeometryConstraint(name="tyr_oh_to_carbonyl_O", kind="distance",
                           atom_a="protein.catalytic_Tyr.OH",
                           atom_b="substrate.carbonyl_O",
                           min_value=2.4, max_value=3.4, severity="scoring",
                           calibrated_on=["1XYZ"], source="test"),
    ]


def _template(required: str | None = "NADPH",
              state: CofactorState = CofactorState.REDUCED,
              codes: tuple[str, ...] = ("NDP", "NAP"),
              metals: tuple[str, ...] = (),
              assembly: str | None = "monomer") -> CatalyticTemplate:
    return CatalyticTemplate(
        template_id="CT1", family_name="SDR",
        mechanism_summary="hydride from NADPH C4 to the ketone carbon",
        catalytic_residues=[{"label": "catalytic_Tyr", "residue_types": ["TYR"],
                             "role": "general acid",
                             "functional_atoms": ["OH"], "evidence": "M-CSA"}],
        required_cofactor=required, required_cofactor_state=state,
        cofactor_ligand_codes=list(codes), metals=list(metals),
        assembly_state=assembly, geometry_constraints=_constraints(),
        reference_structures=["1XYZ"],
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.EXPERIMENTAL_STRUCTURE,
            identifiers=["1XYZ"]))


def _candidate(cid: str = "c1") -> Candidate:
    return Candidate(
        candidate_id=cid,
        sequence_record=SequenceRecord(candidate_id=cid, sequence=CAND),
        family=FamilyAnnotation(family_name="SDR"),
        catalytic_mapping=CatalyticMapping(catalytic_template_id="CT1",
                                           role_to_index={"catalytic_Tyr": 24}))


def _substrate() -> SubstrateSpec:
    return SubstrateSpec(name="acetophenone", isomeric_smiles=SUBSTRATE_SMILES)


def _ctx(tmp: Path, **policy: object) -> RunContext:
    task = TaskSpec(task_id="t-complex",
                    reaction=ReactionSpec(substrate=_substrate()),
                    budget=Budget(initial_sequence_target=5))
    manifest = RunManifest(run_id="r1", task_id="t-complex", global_seed=3)
    return RunContext(task=task, workdir=tmp, manifest=manifest,
                      policy=ExecutionPolicy(**policy))  # type: ignore[arg-type]


def _db_license(permits_commercial: bool | None) -> str | None:
    """A licence string only when the fixture claims to know the terms."""
    if permits_commercial is None:
        return None
    return "operator-recorded terms"


def _db_source(permits_commercial: bool | None) -> str | None:
    if permits_commercial is None:
        return None
    return "terms accepted at download"


def _registry(permits_commercial: bool | None = None) -> ToolRegistry:
    return ToolRegistry([
        ToolRegistryEntry(key="dock_code", kind=ToolKind.CODE,
                          display_name="fake docking engine", version="1.2",
                          license="Apache-2.0",
                          license_source="LICENSE shipped with the binary",
                          permits_commercial_use=True, needs_legal_review=False),
        ToolRegistryEntry(key="fold_code", kind=ToolKind.CODE,
                          display_name="fake co-folding code", version="0.4",
                          license="MIT", license_source="repository LICENSE",
                          permits_commercial_use=True, needs_legal_review=False),
        ToolRegistryEntry(key="fold_weights", kind=ToolKind.MODEL_WEIGHTS,
                          display_name="fake co-folding weights", version="w1",
                          license="CC-BY-NC-4.0 (as recorded by the operator)",
                          license_source="weights terms accepted at download",
                          permits_commercial_use=permits_commercial,
                          output_terms="outputs inherit the weights' terms"),
        ToolRegistryEntry(key="fold_msa_db", kind=ToolKind.INPUT_DATABASE,
                          display_name="fake MSA database", version="2024_01",
                          license=_db_license(permits_commercial),
                          license_source=_db_source(permits_commercial),
                          permits_commercial_use=permits_commercial),
        ToolRegistryEntry(key="fold_outputs", kind=ToolKind.OUTPUT,
                          display_name="predicted complex coordinates",
                          license=_db_license(permits_commercial),
                          license_source=_db_source(permits_commercial),
                          permits_commercial_use=permits_commercial,
                          output_terms="outputs follow the weights' terms"),
    ])


class _FakeDockingRunner:
    """Writes `n_poses` real pose files, two of which are the same mode."""

    name = "fake_dock"
    version = "1.2"
    registry_keys = ("dock_code",)
    uses_model_weights = False

    def __init__(self, receptor_text: str, n_poses: int = 4,
                 include_substrate: bool = True,
                 report_restraints: tuple[str, ...] | None = None,
                 available: bool = True) -> None:
        self.receptor_text = receptor_text
        self.n_poses = n_poses
        self.include_substrate = include_substrate
        self.report_restraints = report_restraints
        self._available = available
        self.jobs: list[object] = []

    def is_available(self) -> bool:
        return self._available

    def dock(self, job):  # type: ignore[no-untyped-def]
        self.jobs.append(job)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        poses = []
        # Two poses at the same place (one binding mode) and the rest apart.
        offsets = [0.0, 0.1, 6.0, 12.0, 18.0][:self.n_poses]
        for i, off in enumerate(offsets):
            path = job.out_dir / f"pose_{i}.pdb"
            path.write_text(_pose_pdb(self.receptor_text, offset=off,
                                      include_substrate=self.include_substrate),
                            encoding="utf-8")
            poses.append(RawPose(
                path=path, score=-9.0 + i, score_function="fake_vina_score",
                conformer_id=f"conf{i % 2}",
                restrained_constraints=tuple(self.report_restraints)
                if self.report_restraints is not None else job.restraint_names))
        return poses


class _FakeComplexPredictor:
    """Co-folding stand-in: several samples, verbatim confidence fields."""

    name = "fake_cofolder"
    version = "0.4"
    registry_keys = ("fold_code", "fold_weights", "fold_msa_db", "fold_outputs")
    uses_model_weights = True

    def __init__(self, receptor_text: str, n_samples: int = 3,
                 runs_remotely: bool = False, available: bool = True,
                 registry_keys: tuple[str, ...] | None = None) -> None:
        self.receptor_text = receptor_text
        self.n_samples = n_samples
        self.runs_remotely = runs_remotely
        self._available = available
        if registry_keys is not None:
            self.registry_keys = registry_keys
        self.jobs: list[JointPredictionJob] = []

    def is_available(self) -> bool:
        return self._available

    def predict(self, job: JointPredictionJob):  # type: ignore[no-untyped-def]
        self.jobs.append(job)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for i in range(min(self.n_samples, job.n_samples)):
            path = job.out_dir / f"sample_{i}.pdb"
            path.write_text(_pose_pdb(self.receptor_text, offset=2.0 * i),
                            encoding="utf-8")
            out.append(RawPose(
                path=path, sample_index=i,
                model_confidence={"ranking_score": 0.81 - 0.1 * i,
                                  "iptm": 0.74, "pae_interface": 3.2,
                                  "pae_matrix": [[0.1, 0.2], [0.2, 0.1]]},
                restrained_constraints=job.restraint_names))
        return out


# ---------------------------------------------------------------------------
# assembly validation: the complex must be the whole catalytic system
# ---------------------------------------------------------------------------


class TestAssemblyValidation(unittest.TestCase):
    def _request(self, **kw: object) -> ComplexRequest:
        base: dict[str, object] = {
            "candidate_id": "c1", "structure_id": "holo",
            "receptor_path": "receptor.pdb", "substrate_ligand_code": "LIG",
            "substrate_conformer_paths": ("conf0.sdf",),
            "cofactor_ligand_code": "NDP",
        }
        base.update(kw)
        return ComplexRequest(**base)  # type: ignore[arg-type]

    def test_complete_system_validates(self) -> None:
        receptor = read_pdb(_receptor_pdb(), "holo")
        v = validate_assembly(self._request(), _template(), _substrate(),
                              receptor, Route.TEMPLATE_DOCKING)
        self.assertTrue(v.ok, [f for f in v.flags])
        self.assertEqual(v.missing, [])

    def test_protein_plus_substrate_only_is_rejected(self) -> None:
        receptor = read_pdb(_receptor_pdb(ligands=()), "apo")
        v = validate_assembly(self._request(cofactor_ligand_code=None),
                              _template(), _substrate(), receptor,
                              Route.TEMPLATE_DOCKING)
        self.assertFalse(v.ok)
        codes = {c for c, _s, _m in v.flags}
        self.assertIn("cofactor_missing_from_assembly", codes)
        self.assertIn("cofactor", v.missing)
        blockers = [m for _c, s, m in v.flags if s is Severity.BLOCKER]
        self.assertTrue(any("cannot occur" in m for m in blockers))

    def test_oxidised_cofactor_is_treated_as_a_missing_cofactor(self) -> None:
        receptor = read_pdb(_receptor_pdb(ligands=(("NAP", 3),)), "holo_ox")
        v = validate_assembly(self._request(cofactor_ligand_code="NAP"),
                              _template(), _substrate(), receptor,
                              Route.TEMPLATE_DOCKING)
        self.assertFalse(v.ok)
        codes = {c for c, _s, _m in v.flags}
        self.assertIn("cofactor_wrong_oxidation_state", codes)

    def test_unknown_cofactor_state_is_not_assumed_reduced(self) -> None:
        receptor = read_pdb(_receptor_pdb(ligands=(("XYZ", 3),)), "holo_x")
        v = validate_assembly(self._request(cofactor_ligand_code="XYZ"),
                              _template(codes=("XYZ",)), _substrate(), receptor,
                              Route.TEMPLATE_DOCKING)
        self.assertFalse(v.ok)
        self.assertIn("cofactor_state_not_derivable",
                      {c for c, _s, _m in v.flags})

    def test_cofactor_absent_from_the_receptor_blocks_route_a(self) -> None:
        receptor = read_pdb(_receptor_pdb(ligands=()), "apo")
        v = validate_assembly(self._request(), _template(), _substrate(),
                              receptor, Route.TEMPLATE_DOCKING)
        self.assertIn("cofactor_absent_from_receptor",
                      {c for c, _s, _m in v.flags})
        # Route B builds the cofactor itself, so the same request is fine there.
        vb = validate_assembly(self._request(), _template(), _substrate(),
                               receptor, Route.JOINT_PREDICTION)
        self.assertTrue(vb.ok, [f for f in vb.flags])

    def test_required_metal_must_be_present(self) -> None:
        receptor = read_pdb(_receptor_pdb(), "holo")
        v = validate_assembly(self._request(), _template(metals=("ZN",)),
                              _substrate(), receptor, Route.TEMPLATE_DOCKING)
        self.assertFalse(v.ok)
        self.assertIn("required_metal_absent", {c for c, _s, _m in v.flags})
        receptor_zn = read_pdb(_receptor_pdb(ligands=(("NDP", 3), ("ZN", 1))),
                               "holo_zn")
        v2 = validate_assembly(self._request(), _template(metals=("ZN",)),
                               _substrate(), receptor_zn,
                               Route.TEMPLATE_DOCKING)
        self.assertTrue(v2.ok, [f for f in v2.flags])

    def test_subunit_deficient_assembly_blocks_unless_allowed(self) -> None:
        receptor = read_pdb(_receptor_pdb(), "holo")
        v = validate_assembly(self._request(), _template(assembly="homodimer"),
                              _substrate(), receptor, Route.TEMPLATE_DOCKING)
        self.assertFalse(v.ok)
        self.assertIn("assembly_state_incomplete", {c for c, _s, _m in v.flags})
        v2 = validate_assembly(
            self._request(), _template(assembly="homodimer"), _substrate(),
            receptor, Route.TEMPLATE_DOCKING,
            ComplexPolicy(allow_subunit_deficient_assembly=True))
        self.assertTrue(v2.ok)
        self.assertIn("assembly_state_incomplete", {c for c, _s, _m in v2.flags})

    def test_substrate_without_a_structure_is_rejected(self) -> None:
        receptor = read_pdb(_receptor_pdb(), "holo")
        v = validate_assembly(self._request(), _template(),
                              SubstrateSpec(name="acetophenone"), receptor,
                              Route.TEMPLATE_DOCKING)
        self.assertFalse(v.ok)
        self.assertIn("substrate_not_structurally_defined",
                      {c for c, _s, _m in v.flags})

    def test_route_a_without_conformers_is_rejected(self) -> None:
        receptor = read_pdb(_receptor_pdb(), "holo")
        v = validate_assembly(self._request(substrate_conformer_paths=()),
                              _template(), _substrate(), receptor,
                              Route.TEMPLATE_DOCKING)
        self.assertFalse(v.ok)
        self.assertIn("no_substrate_conformers", {c for c, _s, _m in v.flags})

    def test_a_restraint_the_template_does_not_define_is_a_template_error(self) -> None:
        receptor = read_pdb(_receptor_pdb(), "holo")
        with self.assertRaises(TemplateError):
            validate_assembly(self._request(restraint_names=("not_a_constraint",)),
                              _template(), _substrate(), receptor,
                              Route.TEMPLATE_DOCKING)


# ---------------------------------------------------------------------------
# the search box
# ---------------------------------------------------------------------------


class TestSearchBox(unittest.TestCase):
    def test_box_is_built_from_the_trusted_site(self) -> None:
        receptor = read_pdb(_receptor_pdb(), "holo")
        _chain, rmap, choice = select_chain(receptor, CAND)
        atoms, labels, basis = trusted_site_atoms(
            receptor, choice.chain_id, rmap, {"catalytic_Tyr": 24}, "NDP")
        self.assertTrue(any("NDP" in l for l in labels))
        self.assertTrue(any("catalytic_Tyr" in l for l in labels))
        self.assertIn("cofactor", basis)
        box = catalytic_site_box(atoms, 4.0, 12.0, labels, basis)
        self.assertGreaterEqual(min(box.size), 12.0)
        self.assertEqual(len(box.anchors), len(labels))

    def test_no_trusted_anchor_refuses_to_define_a_box(self) -> None:
        receptor = read_pdb(_receptor_pdb(ligands=()), "apo")
        _chain, rmap, choice = select_chain(receptor, CAND)
        with self.assertRaises(Exception):
            trusted_site_atoms(receptor, choice.chain_id, rmap, {}, None)


# ---------------------------------------------------------------------------
# poses: comparison, clustering, restraint bookkeeping
# ---------------------------------------------------------------------------


class TestPoseBookkeeping(unittest.TestCase):
    def _atoms(self, text: str, code: str = "LIG"):
        s = read_pdb(text, "p")
        return [a for r in s.ligands() if r.resname.strip() == code
                for a in r.heavy_atoms()]

    def test_rmsd_pairs_atoms_by_name_not_by_file_order(self) -> None:
        a = self._atoms(_pose_pdb(_receptor_pdb(), offset=0.0))
        b = self._atoms(_pose_pdb(_receptor_pdb(), offset=1.0))
        self.assertAlmostEqual(pose_rmsd(a, b) or 0.0, 1.0, places=6)
        self.assertAlmostEqual(pose_rmsd(a, list(reversed(b))) or 0.0, 1.0,
                               places=6)

    def test_rmsd_of_mismatched_atom_sets_is_none_not_a_big_number(self) -> None:
        a = self._atoms(_pose_pdb(_receptor_pdb(), offset=0.0, n_atoms=3))
        b = self._atoms(_pose_pdb(_receptor_pdb(), offset=0.0, n_atoms=2))
        self.assertIsNone(pose_rmsd(a, b))

    def test_clustering_groups_but_never_drops(self) -> None:
        poses = {
            "p1": self._atoms(_pose_pdb(_receptor_pdb(), offset=0.0)),
            "p2": self._atoms(_pose_pdb(_receptor_pdb(), offset=0.2)),
            "p3": self._atoms(_pose_pdb(_receptor_pdb(), offset=9.0)),
        }
        clusters = cluster_poses(poses, 2.0)
        self.assertEqual(set(clusters), {"p1", "p2", "p3"})
        self.assertEqual(clusters["p1"], clusters["p2"])
        self.assertNotEqual(clusters["p1"], clusters["p3"])

    def test_a_pose_that_forgot_its_restraints_raises(self) -> None:
        good = ComplexPose(pose_id="p1", method="template_docking",
                           restrained_constraints=["hydride_distance"])
        bad = ComplexPose(pose_id="p2", method="template_docking")
        assert_restraints_recorded([good], ["hydride_distance"])
        with self.assertRaises(CircularEvidenceError):
            assert_restraints_recorded([good, bad], ["hydride_distance"])


# ---------------------------------------------------------------------------
# licences
# ---------------------------------------------------------------------------


class TestLicensing(unittest.TestCase):
    def test_unregistered_tool_is_refused_before_it_runs(self) -> None:
        with self.assertRaises(ToolRegistryError):
            check_license(ToolRegistry(), ["dock_code"], False, "runner")

    def test_a_tool_with_no_registry_keys_is_refused(self) -> None:
        with self.assertRaises(ToolRegistryError):
            check_license(_registry(), [], False, "runner")

    def test_weights_must_be_registered_apart_from_code(self) -> None:
        with self.assertRaises(ToolRegistryError):
            check_license(_registry(), ["fold_code"], False, "predictor",
                          uses_model_weights=True)

    def test_commercial_run_is_blocked_by_non_commercial_weights(self) -> None:
        with self.assertRaises(LicenseError):
            check_license(_registry(permits_commercial=False),
                          ["fold_code", "fold_weights"], True, "predictor",
                          uses_model_weights=True)

    def test_unrecorded_terms_are_not_a_permission(self) -> None:
        with self.assertRaises(LicenseError):
            check_license(_registry(permits_commercial=None),
                          ["fold_code", "fold_weights"], True, "predictor",
                          uses_model_weights=True)

    def test_non_commercial_run_may_use_non_commercial_weights(self) -> None:
        facts = check_license(_registry(permits_commercial=False),
                              ["fold_code", "fold_weights", "fold_outputs"],
                              False, "predictor", uses_model_weights=True)
        kinds = {e["kind"] for e in facts["entries"]}
        self.assertEqual(kinds, {"code", "model_weights", "output"})

    def test_a_permission_claim_needs_a_licence_statement(self) -> None:
        with self.assertRaises(ToolRegistryError):
            ToolRegistryEntry(key="k", kind=ToolKind.MODEL_WEIGHTS,
                              display_name="w", permits_commercial_use=True)

    def test_a_licence_claim_needs_a_source(self) -> None:
        with self.assertRaises(ToolRegistryError):
            ToolRegistryEntry(key="k", kind=ToolKind.CODE, display_name="c",
                              license="MIT")

    def test_registry_is_loadable_from_config(self) -> None:
        registry = ToolRegistry.from_config({"tool_registry": [
            {"key": "x", "kind": "code", "display_name": "x"}]})
        self.assertEqual(registry.keys(), ["x"])
        self.assertEqual(registry.get("x").kind, ToolKind.CODE)


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------


class TestModelComplexes(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.iface = ModelComplexes()
        self.receptor_text = _receptor_pdb()
        self.receptor_path = self.tmp / "receptor.pdb"
        self.receptor_path.write_text(self.receptor_text, encoding="utf-8")
        self.conformer = self.tmp / "conf0.sdf"
        self.conformer.write_text("fake conformer\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _request(self, **kw: object) -> ComplexRequest:
        base: dict[str, object] = {
            "candidate_id": "c1", "structure_id": "holo",
            "receptor_path": str(self.receptor_path),
            "substrate_ligand_code": "LIG",
            "substrate_conformer_paths": (str(self.conformer),),
            "cofactor_ligand_code": "NDP",
            "restraint_names": ("hydride_distance",),
        }
        base.update(kw)
        return ComplexRequest(**base)  # type: ignore[arg-type]

    def _run(self, *, routes=(Route.TEMPLATE_DOCKING,), runner=None,
             predictor=None, template=None, request=None, registry=None,
             policy=None, access_policy=None, workdir="work", **ctx_policy):
        ctx = _ctx(self.tmp / workdir, **ctx_policy)
        result = self.iface.run(
            ctx, candidates=[_candidate()],
            requests=[request or self._request()],
            catalytic_templates=[template or _template()],
            routes=routes, docking_runner=runner, complex_predictor=predictor,
            tool_registry=registry or _registry(permits_commercial=False),
            policy=policy, access_policy=access_policy)
        return ctx, result

    # -- route A -----------------------------------------------------------
    def test_every_pose_is_kept_written_and_indexed(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=4)
        _ctx_, result = self._run(runner=runner)
        self.assertTrue(result.status.usable, result.message)
        self.assertEqual(len(result.data["poses"]), 4)

        index = Path(result.artifact("pose_index").path)
        rows = index.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(rows) - 1, 4)
        complexes = Path(result.artifact("complexes_dir").path)
        self.assertEqual(len(list(complexes.glob("*.cif"))), 4)
        # every stored pose is a readable mmCIF carrying substrate and cofactor
        for cif in complexes.glob("*.cif"):
            s = read_mmcif(cif, "pose")
            names = {r.resname.strip() for r in s.ligands()}
            self.assertIn("LIG", names)
            self.assertIn("NDP", names)

        clusters = result.data["clusters"]
        self.assertEqual(len(clusters), 4)
        self.assertLess(len(set(clusters.values())), 4,
                        "the two near-identical poses should share a cluster")

    def test_restraints_are_recorded_on_every_pose(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=2)
        _ctx_, result = self._run(runner=runner)
        for pose in result.data["poses"]:
            self.assertIn("hydride_distance", pose["restrained_constraints"])
        self.assertEqual(
            result.provenance.parameters["restrained_constraints"],
            ["hydride_distance"])

    def test_undeclared_runner_restraints_are_recorded_and_flagged(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=1,
                                    report_restraints=("hydride_distance",
                                                       "tyr_oh_to_carbonyl_O"))
        _ctx_, result = self._run(runner=runner)
        self.assertIn("undeclared_restraints", [f.code for f in result.qc_flags])
        self.assertIn("tyr_oh_to_carbonyl_O",
                      result.data["poses"][0]["restrained_constraints"])

    def test_single_pose_is_flagged_as_not_a_sampling_result(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=1)
        _ctx_, result = self._run(runner=runner)
        self.assertIn("single_pose_returned", [f.code for f in result.qc_flags])

    def test_a_pose_without_the_substrate_is_invalid_but_kept(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=2,
                                    include_substrate=False)
        _ctx_, result = self._run(runner=runner)
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(len(result.data["poses"]), 2)
        self.assertFalse(any(p["is_valid"] for p in result.data["poses"]))
        self.assertIn("incomplete_catalytic_system",
                      [f.code for f in result.blockers])
        rows = Path(result.artifact("pose_index").path).read_text(
            encoding="utf-8").strip().split("\n")
        self.assertEqual(len(rows) - 1, 2)

    def test_docking_score_is_kept_as_a_score_not_a_ranking_of_activity(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=2)
        _ctx_, result = self._run(runner=runner)
        first = result.data["poses"][0]
        self.assertEqual(first["docking_score"], -9.0)
        self.assertEqual(first["docking_score_function"], "fake_vina_score")
        self.assertIn("not a catalytic activity ranking",
                      result.data["ranking_score_caveat"])

    def test_cofactor_from_the_crystal_is_not_labelled_as_predicted(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=1)
        _ctx_, result = self._run(runner=runner)
        pose = result.data["poses"][0]
        self.assertEqual(pose["cofactor_source"],
                         LigandSource.EXPERIMENTAL_OBSERVED.value)
        self.assertEqual(pose["substrate_source"],
                         LigandSource.DOCKING_PREDICTED.value)

    def test_missing_runner_fails_with_an_install_hint(self) -> None:
        _ctx_, result = self._run(runner=UnavailableDockingRunner())
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("docking_runner", result.message)
        self.assertIn("docking engine", result.message)

    def test_external_binaries_disabled_stops_route_a(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text)
        _ctx_, result = self._run(runner=runner, allow_external_binaries=False)
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(runner.jobs, [])

    def test_the_box_is_recorded_in_the_job(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=1)
        self._run(runner=runner)
        job = runner.jobs[0]
        self.assertTrue(job.box.anchors)
        self.assertGreaterEqual(min(job.box.size), 12.0)
        self.assertEqual(job.conformer_paths, (str(self.conformer),))
        self.assertEqual(job.restraint_names, ("hydride_distance",))

    # -- route B -----------------------------------------------------------
    def test_joint_prediction_records_confidence_verbatim(self) -> None:
        predictor = _FakeComplexPredictor(self.receptor_text, n_samples=3)
        _ctx_, result = self._run(routes=(Route.JOINT_PREDICTION,),
                                  predictor=predictor)
        self.assertTrue(result.status.usable, result.message)
        poses = result.data["poses"]
        self.assertEqual(len(poses), 3)
        self.assertAlmostEqual(poses[0]["model_confidence"]["ranking_score"],
                               0.81)
        self.assertAlmostEqual(poses[0]["model_confidence"]["iptm"], 0.74)
        self.assertAlmostEqual(poses[0]["model_confidence"]["pae_interface"], 3.2)
        # the PAE matrix is not squeezed into the scalar confidence dict
        self.assertNotIn("pae_matrix", poses[0]["model_confidence"])
        index = Path(result.artifact("pose_index").path).read_text(
            encoding="utf-8")
        self.assertIn("0.81", index)
        notes = Path(result.artifact("pose_index_notes").path).read_text(
            encoding="utf-8")
        self.assertIn("not a catalytic activity ranking", notes)

    def test_joint_prediction_sources_both_ligands_as_predicted(self) -> None:
        predictor = _FakeComplexPredictor(self.receptor_text, n_samples=1)
        _ctx_, result = self._run(routes=(Route.JOINT_PREDICTION,),
                                  predictor=predictor)
        pose = result.data["poses"][0]
        self.assertEqual(pose["cofactor_source"],
                         LigandSource.JOINT_STRUCTURE_PREDICTION.value)
        self.assertEqual(pose["substrate_source"],
                         LigandSource.JOINT_STRUCTURE_PREDICTION.value)

    def test_commercial_run_against_non_commercial_weights_fails_closed(self) -> None:
        predictor = _FakeComplexPredictor(self.receptor_text)
        _ctx_, result = self._run(routes=(Route.JOINT_PREDICTION,),
                                  predictor=predictor,
                                  registry=_registry(permits_commercial=False),
                                  allow_commercial_use=True)
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(predictor.jobs, [])
        self.assertIn("commercial", result.message)

    def test_commercial_run_is_allowed_when_the_terms_say_so(self) -> None:
        predictor = _FakeComplexPredictor(self.receptor_text, n_samples=1)
        _ctx_, result = self._run(routes=(Route.JOINT_PREDICTION,),
                                  predictor=predictor,
                                  registry=_registry(permits_commercial=True),
                                  allow_commercial_use=True)
        self.assertTrue(result.status.usable, result.message)
        self.assertEqual(len(predictor.jobs), 1)
        models = result.provenance.models
        self.assertEqual(models["fold_weights"], "w1")
        dbs = result.provenance.databases
        self.assertEqual(dbs["fold_msa_db"], "2024_01")

    def test_remote_predictor_may_not_see_an_unauthorised_sequence(self) -> None:
        predictor = _FakeComplexPredictor(self.receptor_text, runs_remotely=True)
        _ctx_, result = self._run(routes=(Route.JOINT_PREDICTION,),
                                  predictor=predictor)
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(predictor.jobs, [])

    def test_remote_predictor_runs_with_a_named_authorisation(self) -> None:
        predictor = _FakeComplexPredictor(self.receptor_text, n_samples=1,
                                          runs_remotely=True)
        access = AccessPolicy(
            allow_network=True,
            authorizations=(SubmissionAuthorization(
                authorized_by="operator:alice", scope="*",
                justification="sequence already published",
                sequence_sha256=(sequence_hash(CAND),)),))
        _ctx_, result = self._run(routes=(Route.JOINT_PREDICTION,),
                                  predictor=predictor, access_policy=access,
                                  allow_network=True)
        self.assertTrue(result.status.usable, result.message)
        self.assertEqual(len(predictor.jobs), 1)
        self.assertIn("operator:alice",
                      " ".join(result.provenance.parameters["disclosure_notes"]))

    def test_missing_predictor_fails_with_an_install_hint(self) -> None:
        _ctx_, result = self._run(routes=(Route.JOINT_PREDICTION,),
                                  predictor=UnavailableComplexPredictor())
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("co-folding", result.message)

    # -- both routes -------------------------------------------------------
    def test_two_routes_produce_two_labelled_pose_sets(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=2)
        predictor = _FakeComplexPredictor(self.receptor_text, n_samples=2)
        _ctx_, result = self._run(
            routes=(Route.TEMPLATE_DOCKING, Route.JOINT_PREDICTION),
            runner=runner, predictor=predictor)
        routes = {p["method"] for p in result.data["poses"]}
        self.assertEqual(routes, {"template_docking", "fake_cofolder"})
        self.assertEqual(len(result.data["poses_by_candidate"]["c1"]), 4)
        self.assertNotIn("single_modelling_route",
                         [u.code for u in result.uncertainty])

    def test_one_route_is_recorded_as_having_no_second_opinion(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text, n_poses=2)
        _ctx_, result = self._run(runner=runner)
        self.assertIn("single_modelling_route",
                      [u.code for u in result.uncertainty])

    # -- refusals ----------------------------------------------------------
    def test_a_candidate_without_a_catalytic_template_is_not_modelled(self) -> None:
        runner = _FakeDockingRunner(self.receptor_text)
        ctx = _ctx(self.tmp / "w_no_template")
        result = self.iface.run(ctx, candidates=[_candidate()],
                                requests=[self._request()],
                                catalytic_templates=[], routes=(Route.TEMPLATE_DOCKING,),
                                docking_runner=runner,
                                tool_registry=_registry())
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(runner.jobs, [])
        self.assertIn("no_catalytic_template", [f.code for f in result.blockers])

    def test_apo_receptor_never_reaches_the_runner(self) -> None:
        apo_path = self.tmp / "apo.pdb"
        apo_path.write_text(_receptor_pdb(ligands=()), encoding="utf-8")
        runner = _FakeDockingRunner(self.receptor_text)
        _ctx_, result = self._run(
            runner=runner,
            request=self._request(receptor_path=str(apo_path),
                                  cofactor_ligand_code=None))
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(runner.jobs, [])
        self.assertIn("cofactor_missing_from_assembly",
                      [f.code for f in result.blockers])

    def test_no_requests_is_a_typed_refusal(self) -> None:
        ctx = _ctx(self.tmp / "w_empty")
        result = self.iface.run(ctx, candidates=[_candidate()], requests=[],
                                catalytic_templates=[_template()])
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("ComplexRequest", result.message)

    def test_unresolved_substrate_blocks_the_step_before_anything_runs(self) -> None:
        task = TaskSpec(task_id="t-none", budget=Budget())
        manifest = RunManifest(run_id="r", task_id="t-none")
        ctx = RunContext(task=task, workdir=self.tmp / "w_strict",
                         manifest=manifest, policy=ExecutionPolicy())
        runner = _FakeDockingRunner(self.receptor_text)
        result = self.iface.run(ctx, candidates=[_candidate()],
                                requests=[self._request()],
                                catalytic_templates=[_template()],
                                docking_runner=runner)
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(runner.jobs, [])
        self.assertIn("reaction.substrate.isomeric_smiles", result.message)


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
