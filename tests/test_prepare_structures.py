"""Tests for the ``prepare_structures`` interface.

Every fixture is generated inline from a 30-residue synthetic sequence, so the
expected answers can be counted by hand: residue ``i`` of the candidate is
author position ``101 + i``, its CA sits at ``(3*i, 0, 0)``, and the cofactor
sits beside residue 15. Nothing here reads a file from outside the repository
or touches the network.

The emphasis is on the errors that cannot be caught downstream: an oxidised
cofactor read as a hydride donor, a predicted model whose fold is confident
and whose pocket is not, a structure that is quietly a point mutant, a
homologue's crystal structure standing in for the candidate, and an
unpublished sequence being sent to a remote predictor.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eagent.connectors.base import (
    AccessPolicy,
    NetworkDisabledError,
    SubmissionAuthorization,
    UnauthorizedSubmissionError,
)
from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Severity, Status
from eagent.errors import ToolUnavailableError
from eagent.provenance import RunManifest, sequence_hash
from eagent.schemas import (
    Budget,
    Candidate,
    CatalyticMapping,
    CatalyticTemplate,
    CofactorState,
    ConfidenceLevel,
    FamilyAnnotation,
    GeometryConstraint,
    SequenceRecord,
    TaskSpec,
    TemplateProvenance,
    TemplateSourceType,
)
from eagent.science.numbering import one_to_three
from eagent.tools.handoff import CANDIDATES_KEY, HandoffError, serialise_candidates
from eagent.science.structure_io import read_mmcif, read_pdb
from eagent.tools.prepare_structures import (
    PRIORITY_EXPERIMENTAL_ENZYME,
    PRIORITY_HOMOLOGUE_SCAFFOLD,
    PRIORITY_MATCHING_COMPLEX,
    PRIORITY_REUSABLE_PREDICTION,
    PredictionOutcome,
    PredictionRequest,
    PrepareStructures,
    StructureIndex,
    StructureIndexEntry,
    StructurePolicy,
    StructureSource,
    UnavailablePredictor,
    assess_assembly,
    assess_cofactor,
    guard_sequence_submission,
    ligand_inventory,
    metals_in,
    mutation_tokens,
    select_chain,
    write_mmcif_text,
)

CAND = "MKAIVTGGAQGIGRAIAERLAADGYNVAVL"      # 30 residues; Y at index 24
FIRST_RESSEQ = 101                            # author numbering starts here
POCKET_CENTRE_INDEX = 15


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _atom_line(serial: int, name: str, resname: str, chain: str, resseq: int,
               x: float, y: float, z: float, element: str,
               occupancy: float | None = 1.0, bfactor: float | None = 20.0,
               hetatm: bool = False) -> str:
    """One PDB ATOM/HETATM line, columns exactly where the reader expects them."""
    record = "HETATM" if hetatm else "ATOM"
    occ = "      " if occupancy is None else f"{occupancy:6.2f}"
    bf = "      " if bfactor is None else f"{bfactor:6.2f}"
    return (f"{record:<6}{serial:>5} {name:<4}{'':1}{resname:>3} {chain:1}"
            f"{resseq:>4}{'':1}   {x:8.3f}{y:8.3f}{z:8.3f}{occ}{bf}"
            f"{'':10}{element:>2}")


def _protein_pdb(sequence: str = CAND, chain: str = "A",
                 first_resseq: int = FIRST_RESSEQ,
                 skip_indices: tuple[int, ...] = (),
                 bfactor: float = 20.0,
                 pocket_bfactor: float | None = None,
                 ligands: tuple[tuple[str, int], ...] = (),
                 extra_chain: str | None = None) -> str:
    """A CA-only polymer plus optional HETATM groups beside residue 15.

    CA-only is enough for everything under test: sequence agreement, numbering,
    pocket membership and B-factor/pLDDT averaging are all residue-level.
    """
    lines: list[str] = []
    serial = 1
    pocket = set(range(POCKET_CENTRE_INDEX - 2, POCKET_CENTRE_INDEX + 3))
    for i, letter in enumerate(sequence):
        if i in skip_indices:
            continue
        b = pocket_bfactor if (pocket_bfactor is not None and i in pocket) else bfactor
        lines.append(_atom_line(serial, "CA", one_to_three(letter), chain,
                                first_resseq + i, 3.0 * i, 0.0, 0.0, "C",
                                bfactor=b))
        serial += 1
    if extra_chain:
        for i, letter in enumerate(sequence):
            lines.append(_atom_line(serial, "CA", one_to_three(letter),
                                    extra_chain, first_resseq + i,
                                    3.0 * i, 20.0, 0.0, "C", bfactor=bfactor))
            serial += 1
    cx = 3.0 * POCKET_CENTRE_INDEX
    for n, (code, n_atoms) in enumerate(ligands):
        resseq = 900 + n
        if n_atoms == 1:
            lines.append(_atom_line(serial, code, code, chain, resseq,
                                    cx, 2.2, 0.0, code, bfactor=15.0,
                                    hetatm=True))
            serial += 1
            continue
        for k in range(n_atoms):
            name = "C4N" if k == 0 else f"C{k}"
            lines.append(_atom_line(serial, name, code, chain, resseq,
                                    cx + 0.8 * k, 2.2, 0.4 * k, "C",
                                    bfactor=15.0, hetatm=True))
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def _constraint() -> GeometryConstraint:
    return GeometryConstraint(
        name="hydride_distance", kind="distance",
        atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
        target=3.5, tolerance=0.6, severity="gating",
        calibrated_on=["1XYZ"], source="synthetic test template")


def _template(required: str | None = "NADPH",
              state: CofactorState = CofactorState.REDUCED,
              codes: tuple[str, ...] = ("NDP", "NAP"),
              metals: tuple[str, ...] = (),
              assembly: str | None = "homodimer") -> CatalyticTemplate:
    return CatalyticTemplate(
        template_id="CT1", family_name="SDR",
        mechanism_summary="hydride from NADPH C4 to the ketone carbon",
        catalytic_residues=[{"label": "catalytic_Tyr", "residue_types": ["TYR"],
                             "role": "general acid",
                             "functional_atoms": ["OH"], "evidence": "M-CSA"}],
        required_cofactor=required, required_cofactor_state=state,
        cofactor_ligand_codes=list(codes), metals=list(metals),
        assembly_state=assembly, geometry_constraints=[_constraint()],
        reference_structures=["1XYZ"],
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.EXPERIMENTAL_STRUCTURE,
            identifiers=["1XYZ"]))


def _candidate(sequence: str = CAND, cid: str = "c1") -> Candidate:
    return Candidate(
        candidate_id=cid,
        sequence_record=SequenceRecord(candidate_id=cid, sequence=sequence),
        family=FamilyAnnotation(family_name="SDR"),
        catalytic_mapping=CatalyticMapping(catalytic_template_id="CT1",
                                           role_to_index={"catalytic_Tyr": 24}))


def _ctx(tmp: Path, **policy: object) -> RunContext:
    task = TaskSpec(task_id="t-struct", budget=Budget(initial_sequence_target=5))
    manifest = RunManifest(run_id="r1", task_id="t-struct", global_seed=7)
    return RunContext(task=task, workdir=tmp, manifest=manifest,
                      policy=ExecutionPolicy(**policy))  # type: ignore[arg-type]


def _write(tmp: Path, name: str, text: str) -> Path:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


class _FakePredictor:
    """A predictor that writes a model of the sequence it was handed."""

    name = "fake_folder"
    version = "0.1"

    def __init__(self, runs_remotely: bool = False, available: bool = True,
                 pocket_bfactor: float | None = None) -> None:
        self.runs_remotely = runs_remotely
        self._available = available
        self.pocket_bfactor = pocket_bfactor
        self.calls: list[PredictionRequest] = []

    def is_available(self) -> bool:
        return self._available

    def predict(self, request: PredictionRequest) -> PredictionOutcome:
        self.calls.append(request)
        request.out_dir.mkdir(parents=True, exist_ok=True)
        path = request.out_dir / f"{request.candidate_id}.pdb"
        path.write_text(_protein_pdb(request.sequence, bfactor=92.0,
                                     pocket_bfactor=self.pocket_bfactor),
                        encoding="utf-8")
        return PredictionOutcome(path=path, model_name=self.name,
                                 model_version=self.version)


# ---------------------------------------------------------------------------
# cofactor oxidation state: the central trap
# ---------------------------------------------------------------------------


class TestCofactorState(unittest.TestCase):
    def test_oxidised_cofactor_where_reduced_required_is_a_blocker(self) -> None:
        structure = read_pdb(_protein_pdb(ligands=(("NAP", 3),)), "s")
        verdict = assess_cofactor(structure, _template())
        self.assertTrue(verdict.is_blocker)
        self.assertEqual(verdict.found_state, CofactorState.OXIDIZED)
        self.assertIn("no hydride", verdict.message)

    def test_reduced_cofactor_matches(self) -> None:
        structure = read_pdb(_protein_pdb(ligands=(("NDP", 3),)), "s")
        verdict = assess_cofactor(structure, _template())
        self.assertFalse(verdict.is_blocker)
        self.assertEqual(verdict.found_state, CofactorState.REDUCED)
        self.assertEqual(verdict.verdict, "cofactor_state_matches")

    def test_unknown_component_is_not_guessed(self) -> None:
        structure = read_pdb(_protein_pdb(ligands=(("XYZ", 3),)), "s")
        verdict = assess_cofactor(structure, _template(codes=("XYZ",)))
        self.assertEqual(verdict.found_state, CofactorState.UNKNOWN)
        self.assertEqual(verdict.verdict, "cofactor_state_unknown")
        self.assertFalse(verdict.is_blocker)

    def test_apo_structure_reports_absence_rather_than_a_default(self) -> None:
        structure = read_pdb(_protein_pdb(), "s")
        verdict = assess_cofactor(structure, _template())
        self.assertEqual(verdict.verdict, "cofactor_absent")
        self.assertIsNone(verdict.found_code)

    def test_no_template_means_unchecked_not_passed(self) -> None:
        structure = read_pdb(_protein_pdb(ligands=(("NAP", 3),)), "s")
        verdict = assess_cofactor(structure, None)
        self.assertFalse(verdict.is_blocker)
        self.assertIn("no catalytic template", verdict.message)


# ---------------------------------------------------------------------------
# ligands, metals, assembly
# ---------------------------------------------------------------------------


class TestInventoryAndAssembly(unittest.TestCase):
    def test_metal_is_a_ligand_not_noise(self) -> None:
        structure = read_pdb(_protein_pdb(ligands=(("NDP", 3), ("ZN", 1))), "s")
        self.assertEqual(ligand_inventory(structure), {"NDP": 1, "ZN": 1})
        self.assertEqual(metals_in(structure), ["ZN"])

    def test_monomer_against_homodimer_template_is_reported(self) -> None:
        structure = read_pdb(_protein_pdb(), "s")
        verdict = assess_assembly(structure, _template())
        self.assertEqual(verdict.verdict, "subunit_deficient")
        self.assertEqual(verdict.expected_chain_count, 2)

    def test_matching_assembly_passes(self) -> None:
        structure = read_pdb(_protein_pdb(extra_chain="B"), "s")
        verdict = assess_assembly(structure, _template())
        self.assertEqual(verdict.verdict, "assembly_matches")

    def test_unparsed_assembly_word_is_not_guessed(self) -> None:
        structure = read_pdb(_protein_pdb(), "s")
        verdict = assess_assembly(structure, _template(assembly="dimer of dimers"))
        self.assertIsNone(verdict.expected_chain_count)
        self.assertEqual(verdict.verdict, "assembly_unparsed")


# ---------------------------------------------------------------------------
# numbering and sequence agreement
# ---------------------------------------------------------------------------


class TestNumbering(unittest.TestCase):
    def test_mutation_is_reported_in_candidate_to_structure_direction(self) -> None:
        mutant = CAND[:14] + "F" + CAND[15:]
        structure = read_pdb(_protein_pdb(mutant), "s")
        _chain, rmap, choice = select_chain(structure, CAND)
        self.assertEqual(mutation_tokens(rmap), ["A115F"])
        self.assertAlmostEqual(choice.identity or 0.0, 29 / 30, places=6)

    def test_unobserved_loop_becomes_a_missing_region(self) -> None:
        structure = read_pdb(_protein_pdb(skip_indices=(9, 10, 11, 12, 13)), "s")
        _chain, rmap, _choice = select_chain(structure, CAND)
        self.assertEqual(rmap.unobserved_regions(), [(9, 13)])

    def test_named_chain_is_used_and_missing_chain_raises(self) -> None:
        structure = read_pdb(_protein_pdb(extra_chain="B"), "s")
        _chain, _rmap, choice = select_chain(structure, CAND, "B")
        self.assertEqual(choice.chain_id, "B")
        with self.assertRaises(Exception):
            select_chain(structure, CAND, "Z")


# ---------------------------------------------------------------------------
# mmCIF writing
# ---------------------------------------------------------------------------


class TestMmcifWriting(unittest.TestCase):
    def test_round_trip_preserves_atoms_elements_and_absent_occupancy(self) -> None:
        text = _protein_pdb(ligands=(("NDP", 3), ("ZN", 1)))
        # one atom with no occupancy and no B-factor at all
        text = text.replace("END\n", "")
        text += _atom_line(999, "O", "HOH", "A", 950, 1.0, 2.0, 3.0, "O",
                           occupancy=None, bfactor=None) + "\nEND\n"
        original = read_pdb(text, "orig")
        cif = write_mmcif_text(original, "orig")
        reread = read_mmcif(cif, "orig")
        self.assertEqual(reread.n_atoms(), original.n_atoms())
        self.assertEqual([a.element for a in reread.atoms()],
                         [a.element for a in original.atoms()])
        water = [a for a in reread.atoms() if a.resname == "HOH"][0]
        self.assertIsNone(water.occupancy)
        self.assertIsNone(water.bfactor_or_plddt)
        self.assertEqual(sorted(ligand_inventory(reread)), ["NDP", "ZN"])


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------


class TestPrepareStructures(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.iface = PrepareStructures()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # -- helpers ----------------------------------------------------------
    def _run(self, entries: list[dict[str, object]], *,
             candidates: list[Candidate] | None = None,
             templates: list[CatalyticTemplate] | None = None,
             policy: StructurePolicy | None = None,
             predictor: object | None = None,
             access_policy: AccessPolicy | None = None,
             **ctx_policy: object):
        index_path = self.tmp / "index.json"
        index_path.write_text(json.dumps({"cache_version": "test-1",
                                          "entries": entries}), encoding="utf-8")
        ctx = _ctx(self.tmp / "work", **ctx_policy)  # type: ignore[arg-type]
        result = self.iface.run(
            ctx, candidates=candidates or [_candidate()],
            index_path=index_path,
            catalytic_templates=templates or [_template()],
            policy=policy or StructurePolicy(),
            predictor=predictor, access_policy=access_policy)
        return ctx, result

    def _entry(self, name: str, text: str, **kw: object) -> dict[str, object]:
        _write(self.tmp, name, text)
        entry: dict[str, object] = {
            "structure_id": kw.pop("structure_id", Path(name).stem),
            "path": name, "candidate_id": "c1",
        }
        entry.update(kw)
        return entry

    # -- priority ---------------------------------------------------------
    def test_experimental_holo_beats_a_predicted_model(self) -> None:
        entries = [
            self._entry("pred.pdb", _protein_pdb(bfactor=90.0),
                        structure_id="pred", source="predicted"),
            self._entry("holo.pdb", _protein_pdb(ligands=(("NDP", 3),)),
                        structure_id="holo", source="pdb_complex"),
        ]
        _ctx_, result = self._run(entries)
        self.assertTrue(result.status.usable, result.message)
        selected = result.data["selected"]["c1"]
        self.assertEqual(selected["structure_id"], "holo")
        self.assertEqual(selected["priority_rank"], PRIORITY_MATCHING_COMPLEX)
        self.assertIn("cofactor", selected["reason"])

    def test_predicted_model_of_the_candidate_beats_a_homologue_crystal(self) -> None:
        homolog = CAND[:10] + "WWWWWWWWWW" + CAND[20:]     # 20/30 identity
        entries = [
            self._entry("homolog.pdb", _protein_pdb(homolog, ligands=(("NDP", 3),)),
                        structure_id="homolog", source="pdb_complex"),
            self._entry("pred.pdb", _protein_pdb(bfactor=90.0),
                        structure_id="pred", source="afdb"),
        ]
        _ctx_, result = self._run(entries)
        selected = result.data["selected"]["c1"]
        self.assertEqual(selected["structure_id"], "pred")
        self.assertEqual(selected["priority_rank"], PRIORITY_REUSABLE_PREDICTION)
        ranks = {a["structure_id"]: a["priority_rank"]
                 for a in result.data["assessed"]}
        self.assertEqual(ranks["homolog"], PRIORITY_HOMOLOGUE_SCAFFOLD)

    def test_apo_experimental_ranks_below_a_matching_complex(self) -> None:
        entries = [self._entry("apo.pdb", _protein_pdb(), structure_id="apo",
                               source="pdb_apo")]
        _ctx_, result = self._run(entries)
        self.assertEqual(result.data["selected"]["c1"]["priority_rank"],
                         PRIORITY_EXPERIMENTAL_ENZYME)

    def test_structure_below_the_identity_floor_is_rejected(self) -> None:
        unrelated = "W" * 30
        entries = [self._entry("other.pdb", _protein_pdb(unrelated),
                               structure_id="other", source="pdb_complex")]
        _ctx_, result = self._run(entries)
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("c1", result.message)

    def test_a_predicted_model_of_another_sequence_is_not_the_candidates(self) -> None:
        other = CAND[:10] + "WWWWWWWWWW" + CAND[20:]     # 20/30 identity
        entries = [self._entry("stale.pdb", _protein_pdb(other, bfactor=92.0),
                               structure_id="stale", source="afdb")]
        _ctx_, result = self._run(entries)
        ranks = {a["structure_id"]: a["priority_rank"]
                 for a in result.data["assessed"]}
        self.assertEqual(ranks["stale"], PRIORITY_HOMOLOGUE_SCAFFOLD)
        self.assertIn("homologue_structure", [f.code for f in result.qc_flags])

    # -- QC ---------------------------------------------------------------
    def test_oxidised_cofactor_blocks_the_selected_structure(self) -> None:
        entries = [self._entry("holo_ox.pdb", _protein_pdb(ligands=(("NAP", 3),)),
                               structure_id="holo_ox", source="pdb_complex")]
        _ctx_, result = self._run(entries)
        flags = {f.code: f for f in result.qc_flags}
        self.assertIn("cofactor_oxidation_state_mismatch", flags)
        self.assertIs(flags["cofactor_oxidation_state_mismatch"].severity,
                      Severity.BLOCKER)
        self.assertIs(result.status, Status.PARTIAL)
        self.assertFalse(result.ok)

    def test_pocket_plddt_is_recorded_separately_from_the_mean(self) -> None:
        entries = [self._entry("pred.pdb",
                               _protein_pdb(bfactor=95.0, pocket_bfactor=45.0,
                                            ligands=(("NDP", 3),)),
                               structure_id="pred", source="predicted")]
        _ctx_, result = self._run(entries)
        record = result.data["structures"]["c1"]
        self.assertGreater(record["mean_plddt"], 85.0)
        self.assertLess(record["pocket_plddt"], 60.0)
        self.assertIn("pocket_confidence_weak",
                      [f.code for f in result.qc_flags])

    def test_experimental_bfactors_are_never_read_as_plddt(self) -> None:
        entries = [self._entry("holo.pdb", _protein_pdb(ligands=(("NDP", 3),)),
                               structure_id="holo", source="pdb_complex")]
        _ctx_, result = self._run(entries)
        record = result.data["structures"]["c1"]
        self.assertIsNone(record["mean_plddt"])
        self.assertIsNone(record["pocket_plddt"])
        self.assertTrue(any("is not pLDDT" in note for note in record["qc_notes"]),
                        record["qc_notes"])
        conf_dir = Path(result.artifact("confidence_metrics_dir").path)
        payload = json.loads(next(conf_dir.glob("*.json")).read_text("utf-8"))
        self.assertEqual(payload["b_value_kind"], "bfactor")
        self.assertEqual(payload["pocket_confidence_level"],
                         ConfidenceLevel.INSUFFICIENT.value)

    def test_mutant_structure_is_flagged_and_named(self) -> None:
        mutant = CAND[:14] + "F" + CAND[15:]
        entries = [self._entry("mut.pdb", _protein_pdb(mutant,
                                                       ligands=(("NDP", 3),)),
                               structure_id="mut", source="pdb_complex")]
        _ctx_, result = self._run(entries)
        record = result.data["structures"]["c1"]
        self.assertTrue(record["is_mutant_relative_to_candidate"])
        self.assertEqual(record["mutations_in_structure"], ["A115F"])
        self.assertIn("structure_is_mutant", [f.code for f in result.qc_flags])

    def test_catalytic_residue_without_coordinates_is_flagged(self) -> None:
        entries = [self._entry("gap.pdb",
                               _protein_pdb(skip_indices=(23, 24, 25),
                                            ligands=(("NDP", 3),)),
                               structure_id="gap", source="pdb_complex")]
        _ctx_, result = self._run(entries)
        self.assertIn("catalytic_residue_unobserved",
                      [f.code for f in result.qc_flags])

    # -- artifacts --------------------------------------------------------
    def test_artifacts_and_residue_mapping_table(self) -> None:
        entries = [self._entry("holo.pdb", _protein_pdb(ligands=(("NDP", 3),)),
                               structure_id="holo", source="pdb_complex",
                               assembly_in_file="author assembly 1")]
        ctx, result = self._run(entries)
        keys = {a.key for a in result.artifacts}
        self.assertEqual(keys, {"structures_dir", "structure_qc",
                                "residue_atom_mapping", "confidence_metrics_dir"})

        mapping = Path(result.artifact("residue_atom_mapping").path)
        rows = mapping.read_text(encoding="utf-8").strip().split("\n")
        header = rows[0].split("\t")
        self.assertEqual(len(rows) - 1, len(CAND))
        first = dict(zip(header, rows[1].split("\t")))
        self.assertEqual(first["candidate_index0"], "0")
        self.assertEqual(first["candidate_resnum1"], "1")
        self.assertEqual(first["author_resseq"], str(FIRST_RESSEQ))
        self.assertEqual(first["candidate_letter"], CAND[0])
        in_pocket = [r for r in rows[1:]
                     if dict(zip(header, r.split("\t")))["in_pocket"] == "true"]
        self.assertTrue(in_pocket, "the cofactor should define a pocket")

        qc = Path(result.artifact("structure_qc").path).read_text(encoding="utf-8")
        self.assertIn("holo", qc)
        self.assertIn("cofactor_state_matches", qc)

        stored = Path(result.data["selected"]["c1"]["path"])
        self.assertTrue(stored.is_file())
        self.assertEqual(stored.suffix, ".cif")
        reread = read_mmcif(stored, "stored")
        self.assertIn("NDP", ligand_inventory(reread))

        conf_dir = Path(result.artifact("confidence_metrics_dir").path)
        self.assertTrue(any(conf_dir.glob("*.json")))

    def test_numbering_offset_is_recorded(self) -> None:
        entries = [self._entry("holo.pdb", _protein_pdb(ligands=(("NDP", 3),)),
                               structure_id="holo", source="pdb_complex")]
        _ctx_, result = self._run(entries)
        self.assertEqual(result.data["structures"]["c1"]["numbering_offset"],
                         FIRST_RESSEQ - 1)

    # -- cache misses and predictions -------------------------------------
    def test_cache_miss_names_exactly_what_is_needed(self) -> None:
        _ctx_, result = self._run([])
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("c1", result.message)
        self.assertIn(sequence_hash(CAND), result.message)
        codes = {u.code for u in result.uncertainty}
        self.assertIn("structure_unavailable", codes)
        self.assertIn("no_structure_predictor", codes)

    def test_absent_predictor_is_reported_not_substituted(self) -> None:
        _ctx_, result = self._run([], predictor=UnavailablePredictor())
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.data["structures"], {})
        with self.assertRaises(ToolUnavailableError):
            UnavailablePredictor().predict(
                PredictionRequest("c1", CAND, self.tmp, 1))

    def test_local_predictor_fills_the_gap_and_is_recorded(self) -> None:
        predictor = _FakePredictor(pocket_bfactor=50.0)
        _ctx_, result = self._run([], predictor=predictor)
        self.assertTrue(result.status.usable, result.message)
        self.assertEqual(len(predictor.calls), 1)
        selected = result.data["selected"]["c1"]
        self.assertEqual(selected["priority_rank"], PRIORITY_REUSABLE_PREDICTION)
        self.assertIn("fake_folder", str(result.provenance.parameters
                                         ["new_predictions"]))
        self.assertIn("ran locally",
                      " ".join(result.provenance.parameters["disclosure_notes"]))

    def test_a_model_may_be_requested_alongside_an_experimental_structure(self) -> None:
        predictor = _FakePredictor()
        entries = [self._entry("holo.pdb", _protein_pdb(ligands=(("NDP", 3),)),
                               structure_id="holo", source="pdb_complex")]
        _ctx_, result = self._run(
            entries, predictor=predictor,
            policy=StructurePolicy(predict_when_experimental_exists=True))
        self.assertEqual(len(predictor.calls), 1)
        ids = {a["structure_id"] for a in result.data["assessed"]}
        self.assertIn("holo", ids)
        self.assertTrue(any(i.startswith("c1__fake_folder") for i in ids))
        # the experimental complex still wins: a prediction is not promoted
        # for having been made in this run
        self.assertEqual(result.data["selected"]["c1"]["structure_id"], "holo")

    def test_remote_predictor_is_refused_offline_and_nothing_is_sent(self) -> None:
        predictor = _FakePredictor(runs_remotely=True)
        _ctx_, result = self._run([], predictor=predictor)
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(predictor.calls, [])
        self.assertIn("allow_network", result.message)

    def test_remote_predictor_needs_a_named_authorisation(self) -> None:
        predictor = _FakePredictor(runs_remotely=True)
        ctx = _ctx(self.tmp / "w2", allow_network=True)
        with self.assertRaises(UnauthorizedSubmissionError):
            guard_sequence_submission(predictor, CAND, ctx)

        authorised = AccessPolicy(
            allow_network=True,
            authorizations=(SubmissionAuthorization(
                authorized_by="operator:alice", scope="*",
                justification="published sequence, cleared for folding",
                sequence_sha256=(sequence_hash(CAND),)),))
        notes = guard_sequence_submission(predictor, CAND, ctx, authorised)
        self.assertTrue(any("operator:alice" in n for n in notes))

    def test_offline_local_predictor_needs_no_authorisation(self) -> None:
        ctx = _ctx(self.tmp / "w3")
        notes = guard_sequence_submission(_FakePredictor(), CAND, ctx)
        self.assertTrue(any("locally" in n for n in notes))

    def test_network_disabled_is_a_typed_refusal(self) -> None:
        ctx = _ctx(self.tmp / "w4")
        with self.assertRaises(NetworkDisabledError):
            guard_sequence_submission(_FakePredictor(runs_remotely=True), CAND, ctx)

    # -- inputs -----------------------------------------------------------
    def test_missing_candidates_fails_without_inventing_any(self) -> None:
        ctx = _ctx(self.tmp / "w5")
        result = self.iface.run(ctx, candidates=[], index=StructureIndex())
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("invent", result.message)

    def test_missing_index_names_the_missing_input(self) -> None:
        ctx = _ctx(self.tmp / "w6")
        result = self.iface.run(ctx, candidates=[_candidate()])
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("structure index", result.message)

    def test_index_entry_must_say_which_protein_it_is_of(self) -> None:
        with self.assertRaises(ValueError):
            StructureIndexEntry(structure_id="x", path="x.pdb",
                                source=StructureSource.PDB_APO)

    def test_provenance_records_the_policy_and_the_priority_order(self) -> None:
        entries = [self._entry("holo.pdb", _protein_pdb(ligands=(("NDP", 3),)),
                               structure_id="holo", source="pdb_complex",
                               database="rcsb_pdb", database_version="2024-01-01")]
        _ctx_, result = self._run(entries)
        prov = result.provenance
        self.assertEqual(prov.databases["rcsb_pdb"], "2024-01-01")
        self.assertIn("source_priority", prov.parameters)
        self.assertEqual(prov.parameters["policy"]["pocket_radius_angstrom"], 6.0)
        self.assertIsNotNone(prov.random_seed)


# ---------------------------------------------------------------------------
# PAE
# ---------------------------------------------------------------------------


class TestPae(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pae_is_summarised_and_tied_to_pocket_rows(self) -> None:
        n = len(CAND)
        matrix = [[1.0 for _ in range(n)] for _ in range(n)]
        for i in range(POCKET_CENTRE_INDEX - 2, POCKET_CENTRE_INDEX + 3):
            for j in range(POCKET_CENTRE_INDEX - 2, POCKET_CENTRE_INDEX + 3):
                matrix[i][j] = 9.0
        (self.tmp / "pred.pdb").write_text(
            _protein_pdb(bfactor=90.0, ligands=(("NDP", 3),)), encoding="utf-8")
        (self.tmp / "pred_pae.json").write_text(json.dumps({"pae": matrix}),
                                                encoding="utf-8")
        index_path = self.tmp / "index.json"
        index_path.write_text(json.dumps({"entries": [{
            "structure_id": "pred", "path": "pred.pdb", "candidate_id": "c1",
            "source": "predicted", "pae_json": "pred_pae.json"}]}),
            encoding="utf-8")
        ctx = _ctx(self.tmp / "work")
        result = PrepareStructures().run(
            ctx, candidates=[_candidate()], index_path=index_path,
            catalytic_templates=[_template()])
        conf = json.loads(next((self.tmp / "work" / "prepare_structures"
                                / "confidence_metrics").glob("*.json"))
                          .read_text(encoding="utf-8"))
        self.assertIsNotNone(conf["pae_mean"])
        self.assertAlmostEqual(conf["pae_pocket_mean"], 9.0, places=6)
        self.assertTrue(result.status.usable, result.message)


# ---------------------------------------------------------------------------
# the candidate hand-off
# ---------------------------------------------------------------------------


class TestCandidateHandoff(unittest.TestCase):
    """This step must read what ``annotate_family`` actually publishes.

    ``annotate_family`` writes its candidates into ``result.data`` as JSON,
    because that is what the manifest stores, while this step is typed against
    the model. Wiring the two together the obvious way therefore used to hand
    mappings to code expecting models, and the run died on a missing attribute
    deep inside a structure assessment with nothing naming the step that
    produced the payload.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.iface = PrepareStructures()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _index_path(self) -> Path:
        _write(self.tmp, "holo.pdb", _protein_pdb(ligands=(("NDP", 3),)))
        index_path = self.tmp / "index.json"
        index_path.write_text(json.dumps({"cache_version": "test-1", "entries": [
            {"structure_id": "holo", "path": "holo.pdb", "candidate_id": "c1",
             "source": "pdb_complex"}]}), encoding="utf-8")
        return index_path

    def _run_with(self, candidates: object, workdir: str):
        return self.iface.run(
            _ctx(self.tmp / workdir), candidates=candidates,
            index_path=self._index_path(),
            catalytic_templates=[_template()])

    def test_accepts_the_serialised_payload_annotate_family_publishes(self) -> None:
        # Exactly the expression annotate_family writes into result.data.
        published = {CANDIDATES_KEY: serialise_candidates([_candidate()]),
                     "annotations": [], "per_family_counts": {}}
        from_models = self._run_with([_candidate()], "w_models")
        from_mapping = self._run_with(published, "w_mapping")
        from_rows = self._run_with(published[CANDIDATES_KEY], "w_rows")

        for result in (from_models, from_mapping, from_rows):
            self.assertTrue(result.status.usable, result.message)
            self.assertEqual(result.data["selected"]["c1"]["structure_id"],
                             "holo")
        # and the step republishes the same serialised form, so the next step
        # can be wired from this one's data mapping in turn.
        self.assertEqual(from_mapping.data[CANDIDATES_KEY],
                         serialise_candidates([_candidate()]))

    def test_a_malformed_payload_fails_at_the_boundary(self) -> None:
        # A candidate that lost its sequence record in transit. Scoring it as
        # though the sequence were genuinely absent is the failure mode the
        # boundary exists to stop.
        broken = [{"candidate_id": "c1", "family": {"family_name": "SDR"}}]
        with self.assertRaises(HandoffError) as caught:
            self.iface.execute(_ctx(self.tmp / "w_raise"), candidates=broken,
                               index_path=self._index_path(),
                               catalytic_templates=[_template()])
        self.assertIn("prepare_structures", str(caught.exception))

        result = self._run_with(broken, "w_envelope")
        self.assertIs(result.status, Status.FAILED)
        codes = {f.code for f in result.blockers}
        self.assertIn("handofferror", codes)
        self.assertNotIn("internal_error", codes)
        self.assertNotIn("traceback", result.data)

    def test_a_string_is_not_a_candidate_payload(self) -> None:
        with self.assertRaises(HandoffError):
            self.iface.execute(_ctx(self.tmp / "w_str"),
                               candidates="candidates.json",
                               index_path=self._index_path(),
                               catalytic_templates=[_template()])


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
