"""Tests for the gnina and Boltz adapters.

No binary is installed here, so every test drives an adapter through a fake
:class:`CommandRunner` that writes output in the shape the tool's own source or
documentation describes (see the module docstring of
``eagent.tools.prediction_backends`` for what was read and what was not). That
exercises the adapter's argument construction, output parsing and refusals. It
says nothing about whether gnina or Boltz behave as documented, and the tests
say so rather than imply it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import yaml

from eagent.context import ExecutionPolicy, RunContext
from eagent.errors import LicenseError, ToolUnavailableError
from eagent.provenance import RunManifest
from eagent.schemas import (
    Budget, Candidate, CatalyticMapping, CatalyticTemplate, CofactorState,
    FamilyAnnotation, GeometryConstraint, ReactionSpec, SequenceRecord,
    SubstrateSpec, TaskSpec, TemplateProvenance, TemplateSourceType,
)
from eagent.science.structure_io import (
    Atom, Chain, Residue, Structure, read_mmcif, read_structure,
)
from eagent.tools.mine_sequences import CommandResult
from eagent.tools.model_complexes import (
    ComplexRequest, DockingJob, JointPredictionJob, ModelComplexes,
    Route, SearchBox, ToolRegistry, check_license,
)
from eagent.tools.prediction_backends import (
    BackendRunError, BoltzComplexPredictor, GninaDockingRunner, SdfPose,
    UnsupportedJobError, merge_ligand_into_receptor, parse_sdf,
    rename_residues,
)
from eagent.tools.prepare_structures import write_mmcif_text

REPO = Path(__file__).resolve().parents[1]
SEQ = "MKAIVTGGAQGIGRAIAERLAADGYNVAVL"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def sdf_record(title: str, atoms, props: dict[str, str]) -> str:
    """A V2000 record. ``atoms`` is a list of (symbol, x, y, z)."""
    lines = [title, "  fake-engine", "",
             f"{len(atoms):>3}{0:>3}  0  0  0  0  0  0  0  0999 V2000"]
    for sym, x, y, z in atoms:
        lines.append(f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {sym:<3} 0  0  0  0  0  0  0  0  0  0  0  0")
    lines.append("M  END")
    for key, value in props.items():
        lines += [f"> <{key}>", value, ""]
    lines.append("$$$$")
    return "\n".join(lines) + "\n"


LIGAND_ATOMS = [("C", 0.0, 0.0, 0.0), ("O", 1.23, 0.0, 0.0),
                ("C", -0.75, 1.30, 0.0), ("C", -0.75, -1.30, 0.0)]


def make_receptor(with_cofactor: bool = True, extra_resname: str | None = None
                  ) -> Structure:
    residues = []
    serial = 0
    for i, letter in enumerate("MKAIV"):
        serial += 1
        residues.append(Residue(
            chain="A", resname="ALA", resseq=10 + i, icode="",
            atoms=[Atom(serial, "CA", "C", "ALA", "A", 10 + i, "", "",
                        3.0 * i, 0.0, 0.0, 1.0, 20.0)]))
    if with_cofactor:
        serial += 1
        residues.append(Residue(
            chain="A", resname="NDP", resseq=301, icode="", is_hetatm=True,
            atoms=[Atom(serial, "C4N", "C", "NDP", "A", 301, "", "",
                        6.0, 2.4, 0.4, 1.0, 20.0, True)]))
    if extra_resname:
        serial += 1
        residues.append(Residue(
            chain="A", resname=extra_resname, resseq=302, icode="",
            is_hetatm=True,
            atoms=[Atom(serial, "C1", "C", extra_resname, "A", 302, "", "",
                        7.0, 0.0, 0.0, 1.0, 20.0, True)]))
    return Structure(structure_id="rec", chains=[Chain("A", residues)],
                     source_format="mmcif", models_present=[1],
                     model_selected=1)


class FakeRunner:
    """Records every call and lets a test script the engine's behaviour."""

    def __init__(self, behaviour=None, returncode: int = 0, stderr: str = "",
                 version_line: str = "fake 1.0") -> None:
        self.calls: list[list[str]] = []
        self.behaviour = behaviour
        self.returncode = returncode
        self.stderr = stderr
        self.version_line = version_line

    def __call__(self, argv: Sequence[str], *, cwd=None, timeout=None):
        argv = list(argv)
        if argv[-1] == "--version":
            return CommandResult(tuple(argv), 0, self.version_line + "\n", "")
        self.calls.append(argv)
        if self.returncode == 0 and self.behaviour:
            self.behaviour(argv)
        return CommandResult(tuple(argv), self.returncode, "", self.stderr)


def finder(name: str) -> str | None:
    return f"/usr/bin/{name}"


def none_finder(name: str) -> str | None:
    return None


def arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# --------------------------------------------------------------------------
# SD files and structure assembly
# --------------------------------------------------------------------------

class TestSdf(unittest.TestCase):

    def test_atoms_and_data_items_are_read(self) -> None:
        text = sdf_record("pose1", LIGAND_ATOMS,
                          {"minimizedAffinity": "-6.20", "CNNscore": "0.81"})
        text += sdf_record("pose2", LIGAND_ATOMS, {"minimizedAffinity": "-5.9"})
        poses = parse_sdf(text)
        self.assertEqual(len(poses), 2)
        self.assertEqual(poses[0].elements, ("C", "O", "C", "C"))
        self.assertAlmostEqual(poses[0].coords[1][0], 1.23)
        self.assertAlmostEqual(poses[0].number("minimizedAffinity"), -6.2)
        self.assertAlmostEqual(poses[0].number("CNNscore"), 0.81)

    def test_a_missing_property_is_none_not_zero(self) -> None:
        pose = parse_sdf(sdf_record("p", LIGAND_ATOMS, {}))[0]
        self.assertIsNone(pose.number("minimizedAffinity"))

    def test_a_non_numeric_property_is_none(self) -> None:
        pose = parse_sdf(sdf_record("p", LIGAND_ATOMS, {"CNNscore": "n/a"}))[0]
        self.assertIsNone(pose.number("CNNscore"))

    def test_two_letter_elements_are_normalised(self) -> None:
        pose = parse_sdf(sdf_record("p", [("CL", 0, 0, 0)], {}))[0]
        self.assertEqual(pose.elements, ("Cl",))

    def test_a_truncated_record_is_refused_not_shortened(self) -> None:
        text = sdf_record("p", LIGAND_ATOMS, {})
        lines = text.split("\n")
        broken = "\n".join(lines[:6] + lines[8:])   # drops two atom lines
        with self.assertRaises(BackendRunError):
            parse_sdf(broken)

    def test_v3000_is_refused_not_half_read(self) -> None:
        text = sdf_record("p", LIGAND_ATOMS, {}).replace("V2000", "V3000")
        with self.assertRaises(BackendRunError):
            parse_sdf(text)


class TestAssembly(unittest.TestCase):

    def pose(self) -> SdfPose:
        return parse_sdf(sdf_record("p", LIGAND_ATOMS, {}))[0]

    def test_the_ligand_joins_the_receptor_and_keeps_the_cofactor(self) -> None:
        merged, selector = merge_ligand_into_receptor(
            make_receptor(), self.pose(), "LIG", chain_id="A")
        names = {r.resname for c in merged.chains for r in c.residues}
        self.assertIn("NDP", names)
        self.assertIn("LIG", names)
        self.assertEqual(selector["resname"], "LIG")
        self.assertEqual(selector["chain"], "A")
        self.assertEqual(selector["resseq"], 302)       # after the cofactor at 301
        self.assertFalse(selector["atom_names_follow_reaction_atom_map"])

    def test_atoms_are_named_by_element_and_ordinal(self) -> None:
        merged, _ = merge_ligand_into_receptor(
            make_receptor(), self.pose(), "LIG", chain_id="A")
        lig = next(r for c in merged.chains for r in c.residues
                   if r.resname == "LIG")
        self.assertEqual([a.name for a in lig.atoms], ["C1", "O1", "C2", "C3"])

    def test_the_receptor_is_not_mutated(self) -> None:
        receptor = make_receptor()
        merge_ligand_into_receptor(receptor, self.pose(), "LIG", chain_id="A")
        self.assertNotIn("LIG", {r.resname for r in receptor.chains[0].residues})

    def test_a_receptor_already_holding_that_name_is_refused(self) -> None:
        with self.assertRaises(BackendRunError) as caught:
            merge_ligand_into_receptor(make_receptor(extra_resname="LIG"),
                                       self.pose(), "LIG", chain_id="A")
        self.assertIn("indistinguishable", str(caught.exception))

    def test_an_unknown_chain_is_refused(self) -> None:
        with self.assertRaises(BackendRunError):
            merge_ligand_into_receptor(make_receptor(), self.pose(), "LIG",
                                       chain_id="Z")

    def test_rename_changes_only_the_named_component(self) -> None:
        merged, _ = merge_ligand_into_receptor(
            make_receptor(), self.pose(), "LIG1", chain_id="A")
        renamed = rename_residues(merged, {"LIG1": "LIG"})
        names = [r.resname for r in renamed.chains[0].residues]
        self.assertIn("LIG", names)
        self.assertNotIn("LIG1", names)
        self.assertIn("NDP", names)
        for atom in next(r for r in renamed.chains[0].residues
                         if r.resname == "LIG").atoms:
            self.assertEqual(atom.resname, "LIG")
        # per-atom confidence on untouched residues survives
        self.assertEqual(renamed.chains[0].residues[0].atoms[0].bfactor_or_plddt,
                         20.0)
        self.assertTrue(any("rewritten" in n for n in renamed.parse_notes))


# --------------------------------------------------------------------------
# gnina
# --------------------------------------------------------------------------

class _GninaCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.receptor_path = self.tmp / "receptor.cif"
        self.receptor_path.write_text(
            write_mmcif_text(make_receptor(), "rec"), encoding="utf-8")
        self.conformer = self.tmp / "conf0.sdf"
        self.conformer.write_text("conformer\n", encoding="utf-8")

    def job(self, **kw) -> DockingJob:
        base = dict(
            candidate_id="c1", receptor_path=self.receptor_path,
            receptor_chain="A",
            box=SearchBox(center=(1.5, 2.0, -3.25), size=(14.0, 15.0, 16.0),
                          anchors=("catalytic_Tyr",), basis="test"),
            conformer_paths=(str(self.conformer),),
            substrate_ligand_code="LIG", cofactor_ligand_code="NDP",
            restraints=(), seed=77, out_dir=self.tmp / "out")
        base.update(kw)
        return DockingJob(**base)

    @staticmethod
    def writes_poses(*, props=None, n=3):
        props = props if props is not None else [
            {"minimizedAffinity": f"-{7 - i}.0", "CNNscore": f"0.{9 - i}",
             "CNNaffinity": "5.1", "CNN_VS": "4.6"} for i in range(n)]

        def behaviour(argv):
            out = Path(arg(argv, "-o"))
            out.write_text("".join(sdf_record(f"m{i}", LIGAND_ATOMS, p)
                                   for i, p in enumerate(props)),
                           encoding="utf-8")
        return behaviour


class TestGnina(_GninaCase):

    def runner(self, behaviour=None, **kw) -> tuple[GninaDockingRunner, FakeRunner]:
        fake = FakeRunner(behaviour or self.writes_poses())
        return GninaDockingRunner(runner=fake, executable_finder=finder, **kw), fake

    def test_the_command_line_carries_the_job(self) -> None:
        engine, fake = self.runner()
        engine.dock(self.job())
        argv = fake.calls[0]
        self.assertEqual(argv[0], "/usr/bin/gnina")
        self.assertEqual(arg(argv, "--center_x"), "1.500")
        self.assertEqual(arg(argv, "--center_z"), "-3.250")
        self.assertEqual(arg(argv, "--size_y"), "15.000")
        self.assertEqual(arg(argv, "--seed"), "77")
        self.assertEqual(arg(argv, "-l"), str(self.conformer))
        self.assertEqual(arg(argv, "--cnn_scoring"), "rescore")
        self.assertIn("--no_gpu", argv)

    def test_a_cif_receptor_is_handed_over_as_pdb(self) -> None:
        engine, fake = self.runner()
        engine.dock(self.job())
        received = Path(arg(fake.calls[0], "-r"))
        self.assertEqual(received.suffix, ".pdb")
        self.assertTrue(received.is_file())

    def test_every_pose_is_returned_in_engine_order(self) -> None:
        engine, _ = self.runner()
        poses = engine.dock(self.job())
        self.assertEqual(len(poses), 3)
        self.assertEqual([p.sample_index for p in poses], [1, 2, 3])
        self.assertEqual([p.score for p in poses], [-7.0, -6.0, -5.0])

    def test_the_score_is_labelled_as_a_scoring_function_value(self) -> None:
        engine, _ = self.runner()
        pose = engine.dock(self.job())[0]
        self.assertIn("minimizedAffinity", pose.score_function)
        self.assertIn("comparable only within", pose.score_function)

    def test_cnn_values_get_explicit_names_and_never_become_ranking_score(self) -> None:
        engine, _ = self.runner()
        confidence = engine.dock(self.job())[0].model_confidence
        self.assertEqual(set(confidence),
                         {"cnn_pose_score", "cnn_affinity_pK", "cnn_vs"})
        self.assertNotIn("ranking_score", confidence)

    def test_a_property_gnina_did_not_write_is_none_not_zero(self) -> None:
        engine, _ = self.runner(self.writes_poses(props=[{}]))
        pose = engine.dock(self.job())[0]
        self.assertIsNone(pose.score)
        self.assertEqual(pose.model_confidence, {})
        self.assertFalse(pose.extra["score_property_present"])

    def test_each_pose_is_a_readable_complex_with_substrate_and_cofactor(self) -> None:
        engine, _ = self.runner()
        pose = engine.dock(self.job())[0]
        structure = read_mmcif(pose.path, "p")
        names = {r.resname for r in structure.ligands()}
        self.assertEqual(names, {"LIG", "NDP"})

    def test_the_pose_records_where_the_ligand_is_and_how_atoms_are_named(self) -> None:
        engine, _ = self.runner()
        extra = engine.dock(self.job())[0].extra
        self.assertEqual(extra["ligand_selector"]["resname"], "LIG")
        self.assertFalse(extra["ligand_selector"]
                         ["atom_names_follow_reaction_atom_map"])
        self.assertEqual(len(extra["raw_output_sha256"]), 64)

    def test_one_run_per_conformer(self) -> None:
        second = self.tmp / "conf1.sdf"
        second.write_text("conformer\n", encoding="utf-8")
        engine, fake = self.runner()
        poses = engine.dock(self.job(
            conformer_paths=(str(self.conformer), str(second))))
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual({p.conformer_id for p in poses}, {"conf0", "conf1"})

    def test_a_restraint_is_refused_before_anything_runs(self) -> None:
        engine, fake = self.runner()
        restraint = GeometryConstraint(
            name="hydride_distance", kind="distance",
            atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
            target=3.5, tolerance=0.6, severity="gating", source="test")
        with self.assertRaises(UnsupportedJobError):
            engine.dock(self.job(restraints=(restraint,)))
        self.assertEqual(fake.calls, [])

    def test_a_nonzero_exit_carries_the_engines_own_message(self) -> None:
        fake = FakeRunner(returncode=2, stderr="CUDA error: no device")
        engine = GninaDockingRunner(runner=fake, executable_finder=finder)
        with self.assertRaises(BackendRunError) as caught:
            engine.dock(self.job())
        self.assertIn("CUDA error: no device", str(caught.exception))

    def test_success_without_an_output_file_is_an_error(self) -> None:
        engine, _ = self.runner(lambda argv: None)
        with self.assertRaises(BackendRunError):
            engine.dock(self.job())

    def test_zero_poses_is_an_empty_result_not_an_invented_one(self) -> None:
        engine, _ = self.runner(lambda argv: Path(arg(argv, "-o")).write_text(
            "", encoding="utf-8"))
        self.assertEqual(list(engine.dock(self.job())), [])

    def test_a_missing_conformer_is_an_error(self) -> None:
        engine, _ = self.runner()
        with self.assertRaises(BackendRunError):
            engine.dock(self.job(conformer_paths=(str(self.tmp / "nope.sdf"),)))

    def test_no_conformers_is_an_unsupported_job(self) -> None:
        engine, _ = self.runner()
        with self.assertRaises(UnsupportedJobError):
            engine.dock(self.job(conformer_paths=()))

    def test_an_absent_binary_is_unavailable_with_a_hint(self) -> None:
        engine = GninaDockingRunner(runner=FakeRunner(),
                                    executable_finder=none_finder)
        self.assertFalse(engine.is_available())
        self.assertEqual(engine.version, "absent")
        with self.assertRaises(ToolUnavailableError) as caught:
            engine.dock(self.job())
        self.assertIn("gnina", str(caught.exception))

    def test_the_version_is_what_the_binary_printed(self) -> None:
        engine = GninaDockingRunner(
            runner=FakeRunner(version_line="gnina v1.3 fake-build"),
            executable_finder=finder)
        self.assertEqual(engine.version, "gnina v1.3 fake-build")

    def test_a_weights_free_run_registers_no_weights(self) -> None:
        with_cnn = GninaDockingRunner(runner=FakeRunner(), executable_finder=finder)
        without = GninaDockingRunner(runner=FakeRunner(), executable_finder=finder,
                                     cnn_scoring="none")
        self.assertTrue(with_cnn.uses_model_weights)
        self.assertIn("gnina.model_weights", with_cnn.registry_keys)
        self.assertFalse(without.uses_model_weights)
        self.assertNotIn("gnina.model_weights", without.registry_keys)

    def test_an_unknown_cnn_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            GninaDockingRunner(cnn_scoring="turbo")


# --------------------------------------------------------------------------
# Boltz
# --------------------------------------------------------------------------

def boltz_job(tmp: Path, **kw) -> JointPredictionJob:
    base = dict(
        candidate_id="cand 1", sequence=SEQ, substrate_smiles="CC(=O)c1ccccc1",
        substrate_ligand_code="LIG", cofactor_ligand_code="NDP",
        cofactor_smiles=None, metals=(), n_samples=3, seed=5,
        out_dir=tmp / "out")
    base.update(kw)
    return JointPredictionJob(**base)


def boltz_output(argv: list[str], *, n: int = 3, with_confidence: bool = True,
                 substrate_name: str = "LIG1") -> None:
    """What the Boltz writer produces, per boltz/data/write/writer.py."""
    yaml_path = Path(argv[2])
    stem = yaml_path.stem
    out = (Path(arg(argv, "--out_dir")) / f"boltz_results_{stem}"
           / "predictions" / stem)
    out.mkdir(parents=True)
    for k in range(n):
        structure = Structure(
            structure_id="m", source_format="mmcif", models_present=[1],
            model_selected=1,
            chains=[
                Chain("A", [Residue("A", "ALA", 1, "", [Atom(
                    1, "CA", "C", "ALA", "A", 1, "", "", 0, 0, 0, 1.0, 88.0)])]),
                Chain("B", [Residue("B", substrate_name, 1, "", [Atom(
                    2, "C7", "C", substrate_name, "B", 1, "", "",
                    3.0 + k, 1, 1, 1.0, 80.0, True)], is_hetatm=True)]),
                Chain("C", [Residue("C", "NDP", 1, "", [Atom(
                    3, "C4N", "C", "NDP", "C", 1, "", "",
                    6.0, 2.4, 0.4, 1.0, 85.0, True)], is_hetatm=True)]),
            ])
        (out / f"{stem}_model_{k}.cif").write_text(
            write_mmcif_text(structure, "m"), encoding="utf-8")
        if with_confidence:
            (out / f"confidence_{stem}_model_{k}.json").write_text(json.dumps({
                "confidence_score": 0.9 - 0.1 * k, "ptm": 0.8, "iptm": 0.7,
                "ligand_iptm": 0.6, "protein_iptm": 0.75,
                "complex_plddt": 0.82, "complex_iplddt": 0.7,
                "complex_pde": 0.5, "complex_ipde": 0.9,
                "chains_ptm": {"0": 0.8, "1": 0.3},
                "pair_chains_iptm": {"0": {"0": 0.8, "1": 0.4},
                                     "1": {"0": 0.4, "1": 0.3}}}), encoding="utf-8")


class TestBoltzInput(unittest.TestCase):

    def test_the_document_has_protein_substrate_and_ccd_cofactor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = BoltzComplexPredictor.build_input(boltz_job(Path(tmp)))
        kinds = [next(iter(item)) for item in doc["sequences"]]
        self.assertEqual(kinds, ["protein", "ligand", "ligand"])
        protein = doc["sequences"][0]["protein"]
        self.assertEqual((protein["id"], protein["sequence"]), ("A", SEQ))
        self.assertEqual(doc["sequences"][1]["ligand"],
                         {"id": "B", "smiles": "CC(=O)c1ccccc1"})
        self.assertEqual(doc["sequences"][2]["ligand"], {"id": "C", "ccd": "NDP"})

    def test_without_an_msa_the_protein_runs_single_sequence_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = BoltzComplexPredictor.build_input(boltz_job(Path(tmp)))
        self.assertEqual(doc["sequences"][0]["protein"]["msa"], "empty")

    def test_a_supplied_msa_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = BoltzComplexPredictor.build_input(
                boltz_job(Path(tmp), params={"msa_path": "/data/c1.a3m"}))
        self.assertEqual(doc["sequences"][0]["protein"]["msa"], "/data/c1.a3m")

    def test_metals_become_ccd_ligands_with_fresh_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = BoltzComplexPredictor.build_input(
                boltz_job(Path(tmp), metals=("zn", "MG")))
        ligands = [i["ligand"] for i in doc["sequences"][1:]]
        self.assertEqual([l["id"] for l in ligands], ["B", "C", "D", "E"])
        self.assertEqual([l.get("ccd") for l in ligands[2:]], ["ZN", "MG"])

    def test_a_substrate_with_no_smiles_is_refused_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnsupportedJobError):
                BoltzComplexPredictor.build_input(
                    boltz_job(Path(tmp), substrate_smiles=None))

    def test_a_cofactor_given_only_as_smiles_is_a_smiles_ligand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = BoltzComplexPredictor.build_input(boltz_job(
                Path(tmp), cofactor_ligand_code=None, cofactor_smiles="CCO"))
        self.assertEqual(doc["sequences"][2]["ligand"],
                         {"id": "C", "smiles": "CCO"})


class TestBoltz(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def predictor(self, behaviour=boltz_output, **kw):
        fake = FakeRunner(behaviour)
        return BoltzComplexPredictor(runner=fake, executable_finder=finder,
                                     **kw), fake

    def test_the_command_line_matches_the_cli_that_was_read(self) -> None:
        engine, fake = self.predictor()
        engine.predict(boltz_job(self.tmp))
        argv = fake.calls[0]
        self.assertEqual(argv[:2], ["/usr/bin/boltz", "predict"])
        self.assertEqual(arg(argv, "--diffusion_samples"), "3")
        self.assertEqual(arg(argv, "--seed"), "5")
        self.assertEqual(arg(argv, "--output_format"), "mmcif")
        self.assertEqual(arg(argv, "--model"), "boltz2")

    def test_the_input_file_is_valid_yaml_named_after_a_safe_stem(self) -> None:
        engine, fake = self.predictor()
        engine.predict(boltz_job(self.tmp))
        path = Path(fake.calls[0][2])
        self.assertEqual(path.name, "cand_1.yaml")
        self.assertEqual(yaml.safe_load(path.read_text())["version"], 1)

    def test_the_sequence_is_not_sent_to_a_remote_msa_server_by_default(self) -> None:
        engine, fake = self.predictor()
        engine.predict(boltz_job(self.tmp))
        self.assertNotIn("--use_msa_server", fake.calls[0])
        self.assertFalse(engine.runs_remotely)
        self.assertNotIn("boltz.input_database", engine.registry_keys)

    def test_opting_in_flags_the_predictor_as_remote_and_adds_the_database(self) -> None:
        engine, fake = self.predictor(use_msa_server=True)
        poses = engine.predict(boltz_job(self.tmp))
        self.assertIn("--use_msa_server", fake.calls[0])
        self.assertTrue(engine.runs_remotely)
        self.assertIn("boltz.input_database", engine.registry_keys)
        self.assertTrue(poses[0].extra["remote_msa_server_used"])

    def test_models_come_back_in_boltzs_rank_order(self) -> None:
        engine, _ = self.predictor()
        poses = engine.predict(boltz_job(self.tmp))
        self.assertEqual([p.extra["rank_by_confidence"] for p in poses], [0, 1, 2])
        self.assertEqual(len(poses), 3)

    def test_ranks_sort_numerically_not_lexically(self) -> None:
        engine, _ = self.predictor(lambda argv: boltz_output(argv, n=12))
        poses = engine.predict(boltz_job(self.tmp, n_samples=12))
        self.assertEqual([p.extra["rank_by_confidence"] for p in poses],
                         list(range(12)))

    def test_confidence_keeps_boltzs_own_numbers_under_the_indexed_names(self) -> None:
        engine, _ = self.predictor()
        conf = engine.predict(boltz_job(self.tmp))[1].model_confidence
        self.assertAlmostEqual(conf["ranking_score"], 0.8)
        self.assertAlmostEqual(conf["iptm"], 0.7)
        self.assertAlmostEqual(conf["ligand_iptm"], 0.6)
        self.assertIn("pair_chains_iptm", conf)
        self.assertNotIn("pae_interface", conf)       # Boltz does not write one

    def test_missing_confidence_is_empty_and_recorded_not_zero(self) -> None:
        engine, _ = self.predictor(
            lambda argv: boltz_output(argv, with_confidence=False))
        pose = engine.predict(boltz_job(self.tmp))[0]
        self.assertEqual(pose.model_confidence, {})
        self.assertFalse(pose.extra["confidence_file_present"])

    def test_the_smiles_ligand_is_renamed_to_the_requested_code_and_raw_kept(self) -> None:
        engine, _ = self.predictor()
        pose = engine.predict(boltz_job(self.tmp, substrate_ligand_code="KET"))[0]
        names = {r.resname for r in read_mmcif(pose.path, "p").ligands()}
        self.assertEqual(names, {"KET", "NDP"})
        self.assertEqual(pose.extra["renamed_residues"], {"LIG1": "KET"})
        raw = read_mmcif(pose.extra["raw_output"], "raw")
        self.assertIn("LIG1", {r.resname for r in raw.ligands()})
        self.assertEqual(len(pose.extra["raw_output_sha256"]), 64)

    def test_no_rename_when_the_code_is_already_boltzs(self) -> None:
        engine, _ = self.predictor()
        pose = engine.predict(boltz_job(self.tmp, substrate_ligand_code="LIG1"))[0]
        self.assertEqual(pose.extra["renamed_residues"], {})
        self.assertEqual(Path(pose.path), Path(pose.extra["raw_output"]))

    def test_a_missing_substrate_residue_is_an_error_not_a_guess(self) -> None:
        engine, _ = self.predictor(
            lambda argv: boltz_output(argv, substrate_name="XYZ"))
        with self.assertRaises(BackendRunError) as caught:
            engine.predict(boltz_job(self.tmp, substrate_ligand_code="KET"))
        self.assertIn("LIG1", str(caught.exception))

    def test_atom_naming_is_declared_not_mapped(self) -> None:
        engine, _ = self.predictor()
        extra = engine.predict(boltz_job(self.tmp))[0].extra
        self.assertFalse(extra["atom_names_follow_reaction_atom_map"])
        self.assertIn("canonical", extra["atom_naming"])
        self.assertIn("single-sequence", extra["msa"])

    def test_a_restraint_is_refused_before_anything_runs(self) -> None:
        engine, fake = self.predictor()
        restraint = GeometryConstraint(
            name="hydride_distance", kind="distance",
            atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
            target=3.5, tolerance=0.6, severity="gating", source="test")
        with self.assertRaises(UnsupportedJobError):
            engine.predict(boltz_job(self.tmp, restraints=(restraint,)))
        self.assertEqual(fake.calls, [])

    def test_a_nonzero_exit_carries_the_engines_own_message(self) -> None:
        fake = FakeRunner(returncode=1, stderr="Missing MSA's in input and "
                                               "--use_msa_server flag not set.")
        engine = BoltzComplexPredictor(runner=fake, executable_finder=finder)
        with self.assertRaises(BackendRunError) as caught:
            engine.predict(boltz_job(self.tmp))
        self.assertIn("Missing MSA", str(caught.exception))

    def test_success_without_models_is_an_error(self) -> None:
        engine, _ = self.predictor(lambda argv: None)
        with self.assertRaises(BackendRunError):
            engine.predict(boltz_job(self.tmp))

    def test_an_absent_binary_is_unavailable_with_a_hint(self) -> None:
        engine = BoltzComplexPredictor(runner=FakeRunner(),
                                       executable_finder=none_finder)
        self.assertFalse(engine.is_available())
        with self.assertRaises(ToolUnavailableError):
            engine.predict(boltz_job(self.tmp))

    def test_bad_constructor_arguments_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            BoltzComplexPredictor(accelerator="quantum")
        with self.assertRaises(ValueError):
            BoltzComplexPredictor(model="boltz9")


# --------------------------------------------------------------------------
# through model_complexes, with the shipped registry
# --------------------------------------------------------------------------

def shipped_registry() -> ToolRegistry:
    raw = yaml.safe_load((REPO / "configs" / "tool_registry.yaml").read_text())
    return ToolRegistry.from_config(raw)


class TestLicenceKeysResolve(unittest.TestCase):
    """The adapters' keys exist in the shipped registry, and gate as they should."""

    def test_every_key_an_adapter_declares_is_in_the_registry(self) -> None:
        registry = shipped_registry()
        adapters = [
            GninaDockingRunner(), GninaDockingRunner(cnn_scoring="none"),
            BoltzComplexPredictor(), BoltzComplexPredictor(use_msa_server=True)]
        for adapter in adapters:
            with self.subTest(adapter=type(adapter).__name__,
                              keys=adapter.registry_keys):
                for key in adapter.registry_keys:
                    self.assertIn(key, registry)
                check_license(registry, list(adapter.registry_keys), False,
                              "test", uses_model_weights=adapter.uses_model_weights)

    def test_a_commercial_run_is_blocked_for_gnina_and_for_boltz_outputs(self) -> None:
        registry = shipped_registry()
        for adapter in (GninaDockingRunner(), BoltzComplexPredictor()):
            with self.subTest(adapter=type(adapter).__name__):
                with self.assertRaises(LicenseError):
                    check_license(registry, list(adapter.registry_keys), True,
                                  "test", uses_model_weights=True)

    def test_what_was_read_is_recorded_with_where_and_when(self) -> None:
        registry = shipped_registry()
        code = registry.get("boltz.code")
        self.assertEqual(code.license, "MIT")
        self.assertIn("2026-10-07", code.license_source)
        self.assertIs(code.permits_commercial_use, True)
        self.assertTrue(code.needs_legal_review)
        # outputs: nothing read says anything, so nothing is claimed
        self.assertIsNone(registry.get("boltz.output").permits_commercial_use)
        self.assertIsNone(registry.get("gnina.code").permits_commercial_use)


def _template() -> CatalyticTemplate:
    return CatalyticTemplate(
        template_id="CT1", family_name="SDR",
        mechanism_summary="hydride from NADPH C4 to the ketone carbon",
        catalytic_residues=[{"label": "catalytic_Tyr", "residue_types": ["ALA"],
                             "role": "general acid", "functional_atoms": ["CA"],
                             "evidence": "test"}],
        required_cofactor="NADPH", required_cofactor_state=CofactorState.REDUCED,
        cofactor_ligand_codes=["NDP", "NAP"], assembly_state="monomer",
        geometry_constraints=[GeometryConstraint(
            name="hydride_distance", kind="distance",
            atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
            target=3.5, tolerance=0.6, severity="gating", source="test")],
        reference_structures=["1XYZ"],
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.EXPERIMENTAL_STRUCTURE,
            identifiers=["1XYZ"]))


class TestThroughModelComplexes(unittest.TestCase):
    """The adapters satisfy the seams: poses survive the pipeline's own checks."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        receptor = make_receptor()
        self.receptor_path = self.tmp / "receptor.cif"
        self.receptor_path.write_text(write_mmcif_text(receptor, "rec"),
                                      encoding="utf-8")
        self.conformer = self.tmp / "conf0.sdf"
        self.conformer.write_text("c\n", encoding="utf-8")

    def ctx(self) -> RunContext:
        task = TaskSpec(task_id="t", reaction=ReactionSpec(
            substrate=SubstrateSpec(name="acetophenone",
                                    isomeric_smiles="CC(=O)c1ccccc1")),
            budget=Budget(initial_sequence_target=5))
        return RunContext(task=task, workdir=self.tmp / "work",
                          manifest=RunManifest(run_id="r", task_id="t",
                                               global_seed=3),
                          policy=ExecutionPolicy())

    def candidate(self) -> Candidate:
        return Candidate(
            candidate_id="c1",
            sequence_record=SequenceRecord(candidate_id="c1", sequence=SEQ),
            family=FamilyAnnotation(family_name="SDR"),
            catalytic_mapping=CatalyticMapping(
                catalytic_template_id="CT1", role_to_index={"catalytic_Tyr": 2}))

    def request(self) -> ComplexRequest:
        return ComplexRequest(
            candidate_id="c1", structure_id="holo",
            receptor_path=str(self.receptor_path),
            substrate_ligand_code="LIG",
            substrate_conformer_paths=(str(self.conformer),),
            cofactor_ligand_code="NDP", restraint_names=())

    def test_gnina_poses_pass_the_pipelines_checks_and_are_labelled_predicted(self) -> None:
        engine = GninaDockingRunner(
            runner=FakeRunner(_GninaCase.writes_poses()),
            executable_finder=finder)
        result = ModelComplexes().run(
            self.ctx(), candidates=[self.candidate()], requests=[self.request()],
            catalytic_templates=[_template()], routes=(Route.TEMPLATE_DOCKING,),
            docking_runner=engine, tool_registry=shipped_registry())
        self.assertTrue(result.status.usable, result.message)
        poses = result.data["poses"]
        self.assertEqual(len(poses), 3)
        for pose in poses:
            self.assertTrue(pose["is_valid"], pose.get("invalid_reason"))
            self.assertTrue(pose["substrate_present"])
            self.assertTrue(pose["cofactor_present"])
        self.assertEqual(poses[0]["docking_score"], -7.0)

    def test_boltz_poses_pass_the_pipelines_checks(self) -> None:
        engine = BoltzComplexPredictor(runner=FakeRunner(boltz_output),
                                       executable_finder=finder)
        result = ModelComplexes().run(
            self.ctx(), candidates=[self.candidate()], requests=[self.request()],
            catalytic_templates=[_template()], routes=(Route.JOINT_PREDICTION,),
            complex_predictor=engine, tool_registry=shipped_registry())
        self.assertTrue(result.status.usable, result.message)
        poses = result.data["poses"]
        self.assertEqual(len(poses), 3)
        for pose in poses:
            self.assertTrue(pose["is_valid"], pose.get("invalid_reason"))
        self.assertAlmostEqual(poses[0]["model_confidence"]["ranking_score"], 0.9)

    def test_the_licence_check_uses_the_adapters_real_keys(self) -> None:
        engine = GninaDockingRunner(
            runner=FakeRunner(_GninaCase.writes_poses()),
            executable_finder=finder)
        ctx = self.ctx()
        ctx.policy = ExecutionPolicy(allow_commercial_use=True)
        result = ModelComplexes().run(
            ctx, candidates=[self.candidate()], requests=[self.request()],
            catalytic_templates=[_template()], routes=(Route.TEMPLATE_DOCKING,),
            docking_runner=engine, tool_registry=shipped_registry())
        self.assertFalse(result.status.usable)
        text = result.message + " ".join(f.message for f in result.qc_flags)
        self.assertIn("commercial", text.lower())
        self.assertIn("gnina", text.lower())


if __name__ == "__main__":
    unittest.main()
