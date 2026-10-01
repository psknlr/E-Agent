"""Tests for :mod:`eagent.tools.propose_mutations`.

The cases are the failures the interface exists to make impossible, not its
happy path:

* designing on a parent nobody ever assayed, which is how "we never found the
  right family" gets mistaken for "the scaffold needs engineering";
* a frozen catalytic residue reaching the library, which costs the attribution
  of every other variant on the plate;
* a frozen role that cannot be located, which looks exactly like a safe
  library from the outside;
* a wild-type letter that does not match the parent, which produces a variant
  that expresses, folds and silently answers a different question;
* a mixed numbering convention inside one library;
* a combination variant with no single-mutant controls;
* a design model that edits a position it was told to keep fixed;
* a position supported only by shell membership being ranked as if it had a
  reason.

Everything is built in memory. No file is read, no network is touched.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Severity, Status
from eagent.errors import ToolUnavailableError
from eagent.provenance import RunManifest, sequence_hash
from eagent.schemas import (
    Candidate,
    CatalyticMapping,
    Conditions,
    Detection,
    EngineeringTemplate,
    EvidenceRef,
    EvidenceStrength,
    ExperimentRecord,
    FamilyAnnotation,
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
from eagent.science.numbering import AuthorPosition, ResidueMap
from eagent.tools.propose_mutations import (
    ExperimentalPrecedent,
    FamilySignal,
    LigandMPNNAdapter,
    LigandMPNNRequest,
    LigandMPNNResult,
    MissingLigandMPNN,
    ParentOverride,
    ProposeMutations,
    SitePriority,
    StructuralObservation,
    StructuralRole,
    confirm_parent,
    frozen_indices,
)

#  0         1         2         3
#  0123456789012345678901234567890123456789
SEQ = "MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHT"

FROZEN_TYR = 6            # Y, catalytic
FROZEN_GLY = 9            # G, cofactor-anchoring
CLASH_SITE = 12           # L, steric clash only
SHELL_SITE = 20           # A, shell membership only
PRECEDENT_SITE = 29       # I, experimental + structural
FAMILY_SITE = 33          # Y, family + structural


def make_task(task_id: str = "T1") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        reaction=ReactionSpec(
            substrate=SubstrateSpec(name="acetophenone",
                                    isomeric_smiles="CC(=O)c1ccccc1"),
            product=ProductSpec(name="(R)-1-phenylethanol",
                                isomeric_smiles="C[C@@H](O)c1ccccc1",
                                target_stereochemistry=Stereochemistry.R,
                                creates_new_stereocenter=True),
        ),
        conditions=Conditions(pH=7.0, temperature_C=30.0,
                              expression_host="E. coli BL21(DE3)"),
    )


def make_ctx(tmp: Path, task: TaskSpec | None = None) -> RunContext:
    task = task or make_task()
    return RunContext(
        task=task, workdir=tmp,
        manifest=RunManifest(run_id="R1", task_id=task.task_id),
        policy=ExecutionPolicy(allow_network=False, strict=True),
    )


def make_residue_map(sequence: str = SEQ, *, chain: str = "A",
                     first_author: int = 101,
                     unobserved: tuple[int, ...] = ()) -> ResidueMap:
    """Author numbering offset from the index on purpose: 101, not 1.

    An offset is the normal case and is exactly what an unverified proposal
    gets wrong, so the fixture never lets index and author number coincide.
    """
    rmap = ResidueMap(candidate_sequence=sequence, chain_id=chain)
    for index in range(len(sequence)):
        rmap.index_to_reference[index] = index + 1
        rmap.reference_to_index[index + 1] = index
        if index in unobserved:
            continue
        pos = AuthorPosition(chain, first_author + index, "")
        rmap.index_to_author[index] = pos
        rmap.author_to_index[pos] = index
    return rmap


def make_parent(cid: str = "C1", sequence: str = SEQ, *,
                map_frozen: bool = True) -> Candidate:
    mapping = CatalyticMapping(catalytic_template_id="cat:sdr:kred")
    if map_frozen:
        mapping = CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Tyr": "Y107", "nad_binding_Gly": "G110"},
            role_to_index={"catalytic_Tyr": FROZEN_TYR,
                           "nad_binding_Gly": FROZEN_GLY},
        )
    return Candidate(
        candidate_id=cid,
        sequence_record=SequenceRecord(candidate_id=cid, sequence=sequence),
        family=FamilyAnnotation(family_name="SDR"),
        catalytic_mapping=mapping,
    )


def make_template(**overrides) -> EngineeringTemplate:
    kwargs = dict(
        template_id="eng:sdr:kred",
        family_name="SDR",
        frozen_roles=["catalytic_Tyr", "nad_binding_Gly"],
        mutable_zones=[{
            "name": "first_shell", "role": "substrate_contact",
            "min_angstrom": 0.0, "max_angstrom": 4.5,
            "rationale": "residues whose side chains line the carbonyl pocket",
        }],
        max_simultaneous_mutations_round1=2,
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.MECHANISM_LITERATURE,
            identifiers=["PMID:00000001"]),
    )
    kwargs.update(overrides)
    return EngineeringTemplate(**kwargs)


def confirming_record(sequence: str = SEQ) -> ExperimentRecord:
    return ExperimentRecord(
        record_id="run1:C1:NADPH",
        sequence=sequence,
        outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
        detection=Detection(method="chiral GC-MS",
                            confirms_product_identity=True,
                            authentic_standard=True,
                            chiral_method_validated=True),
        conversion_pct=42.0,
        ee_target_pct=88.0,
        evidence=[EvidenceRef(
            source_type="internal_experiment", identifier="run1",
            strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
            extracted_by="instrument_export")],
    )


def evidence_bundle():
    structural = [
        StructuralObservation(
            candidate_index=CLASH_SITE, role=StructuralRole.STERIC_CLASH,
            detail="hard-sphere overlap of 0.8 A with the substrate aryl ring",
            distance_to_substrate_A=2.9, zone="first_shell", source="1XYZ"),
        StructuralObservation(
            candidate_index=SHELL_SITE,
            role=StructuralRole.POCKET_SHELL_MEMBER,
            detail="an atom lies in the 4-8 A shell",
            distance_to_substrate_A=6.4, zone="outer_shell", source="1XYZ"),
        StructuralObservation(
            candidate_index=PRECEDENT_SITE,
            role=StructuralRole.SUBSTRATE_CONTACT,
            detail="side chain packs against the prochiral face",
            distance_to_substrate_A=3.6, zone="first_shell", source="1XYZ"),
        StructuralObservation(
            candidate_index=FAMILY_SITE, role=StructuralRole.POCKET_ENTRANCE,
            detail="lines the substrate entrance channel",
            distance_to_substrate_A=5.1, zone="entrance", source="1XYZ"),
        StructuralObservation(
            candidate_index=FROZEN_TYR, role=StructuralRole.SUBSTRATE_CONTACT,
            detail="catalytic tyrosine hydroxyl contacts the carbonyl oxygen",
            distance_to_substrate_A=2.7, zone="first_shell", source="1XYZ"),
    ]
    family = [FamilySignal(
        observation=("bulky-substrate subfamily carries F here while the "
                     "small-substrate subfamily carries Y"),
        source="alignment:SDR-KRED-2024",
        reference_position=FAMILY_SITE + 1,
        alternative_residues=["F"],
        covaries_with=[PRECEDENT_SITE + 1])]
    precedents = [ExperimentalPrecedent(
        detail="I->V at this position raised activity on aryl ketones 4-fold",
        source="PMID:00000002", candidate_index=PRECEDENT_SITE,
        wild_type="I", mutant="V", on_this_sequence=False)]
    return structural, family, precedents


def run_interface(ctx: RunContext, **kwargs):
    return ProposeMutations().run(ctx, **kwargs)


def base_kwargs(parent: Candidate, rmap: ResidueMap, template):
    structural, family, precedents = evidence_bundle()
    return dict(
        parents=[parent],
        records=[confirming_record()],
        engineering_template=template,
        residue_maps={parent.candidate_id: rmap},
        structural_observations={parent.candidate_id: structural},
        family_signals={parent.candidate_id: family},
        experimental_precedents={parent.candidate_id: precedents},
    )


class ParentConfirmationTests(unittest.TestCase):
    def test_untested_parent_is_not_confirmed(self):
        verdict = confirm_parent(make_parent(), [])
        self.assertFalse(verdict.confirmed)
        self.assertIn("has not been tested", verdict.reason)

    def test_homolog_evidence_does_not_confirm_this_parent(self):
        record = confirming_record()
        weak = record.model_copy(update={"evidence": [EvidenceRef(
            source_type="publication", identifier="PMID:9",
            strength=EvidenceStrength.HOMOLOG_EXPERIMENTAL)]})
        verdict = confirm_parent(make_parent(), [weak])
        self.assertFalse(verdict.confirmed)
        self.assertIn("homologue", verdict.reason)

    def test_sequence_level_record_confirms(self):
        verdict = confirm_parent(make_parent(), [confirming_record()])
        self.assertTrue(verdict.confirmed)
        self.assertEqual(verdict.supporting_record_ids, ("run1:C1:NADPH",))

    def test_hash_match_not_accession_match(self):
        other = confirming_record("MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHA")
        self.assertNotEqual(other.sequence_sha256, sequence_hash(SEQ))
        self.assertFalse(confirm_parent(make_parent(), [other]).confirmed)


class RefusalTests(unittest.TestCase):
    def test_unconfirmed_parent_produces_no_proposals(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            kwargs = base_kwargs(parent, make_residue_map(), make_template())
            kwargs["records"] = []
            out = run_interface(ctx, **kwargs)
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "unconfirmed_parent"
                                for f in out.blockers))
            self.assertEqual(out.data["n_proposals"], 0)

    def test_override_is_recorded_on_every_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            kwargs = base_kwargs(parent, make_residue_map(), make_template())
            kwargs["records"] = []
            kwargs["overrides"] = [ParentOverride(
                candidate_id=parent.candidate_id,
                reason="the parent assay is queued but the plate ships Friday",
                authorised_by="A. Operator")]
            out = run_interface(ctx, **kwargs)
            self.assertGreater(out.data["n_proposals"], 0)
            self.assertTrue(any(f.code == "unconfirmed_parent_override"
                                and f.severity is Severity.WARN
                                for f in out.qc_flags))
            for proposal in out.data["proposals"]:
                self.assertTrue(
                    any("override recorded by A. Operator" in text
                        for text in proposal["contradicting_evidence"]),
                    proposal["proposal_id"])

    def test_missing_residue_map_blocks_the_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            kwargs = base_kwargs(parent, make_residue_map(), make_template())
            kwargs["residue_maps"] = {}
            out = run_interface(ctx, **kwargs)
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "numbering_unavailable"
                                for f in out.blockers))

    def test_residue_map_for_a_different_sequence_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            wrong = make_residue_map("MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHA")
            kwargs = base_kwargs(parent, wrong, make_template())
            out = run_interface(ctx, **kwargs)
            self.assertTrue(any(f.code == "numbering_unavailable"
                                for f in out.blockers))
            self.assertEqual(out.data["n_proposals"], 0)

    def test_submit_to_is_refused_outright(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            kwargs = base_kwargs(parent, make_residue_map(), make_template())
            out = run_interface(ctx, submit_to="https://design.example.com",
                                **kwargs)
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "external_submission_refused"
                                for f in out.blockers))

    def test_no_parents_is_a_failure_not_an_empty_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = run_interface(make_ctx(Path(tmp)), parents=[])
            self.assertIs(out.status, Status.FAILED)


class FreezePolicyTests(unittest.TestCase):
    def test_frozen_roles_resolve_to_indices(self):
        indices, unresolved = frozen_indices(make_parent(), make_template())
        self.assertEqual(indices, {FROZEN_TYR, FROZEN_GLY})
        self.assertEqual(unresolved, [])

    def test_unmapped_frozen_role_blocks_the_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent(map_frozen=False)
            kwargs = base_kwargs(parent, make_residue_map(), make_template())
            out = run_interface(ctx, **kwargs)
            self.assertTrue(any(f.code == "freeze_unverifiable"
                                for f in out.blockers))
            self.assertEqual(out.data["n_proposals"], 0)

    def test_catalytic_residue_never_appears_in_a_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            out = run_interface(
                ctx, **base_kwargs(parent, make_residue_map(), make_template()))
            self.assertGreater(out.data["n_proposals"], 0)
            touched = {m["position_index"] for p in out.data["proposals"]
                       for m in p["mutations"]}
            self.assertNotIn(FROZEN_TYR, touched)
            self.assertNotIn(FROZEN_GLY, touched)
            reasons = {(e["position_index"], e["reason"])
                       for e in out.data["excluded_sites"]}
            self.assertIn((FROZEN_TYR, "frozen_role"), reasons)

    def test_every_proposal_declares_the_freeze_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run_interface(
                ctx, **base_kwargs(make_parent(), make_residue_map(),
                                   make_template()))
            self.assertTrue(all(p["frozen_roles_respected"]
                                for p in out.data["proposals"]))


class EvidenceAndPriorityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        ctx = make_ctx(Path(self.tmp.name))
        self.out = run_interface(
            ctx, **base_kwargs(make_parent(), make_residue_map(),
                               make_template()))
        self.by_site = {}
        for proposal in self.out.data["proposals"]:
            for mutation in proposal["mutations"]:
                self.by_site.setdefault(mutation["position_index"],
                                        []).append(proposal)

    def tearDown(self):
        self.tmp.cleanup()

    def test_multi_evidence_site_outranks_proximity_only(self):
        best = min(p["experimental_priority"]
                   for p in self.by_site[PRECEDENT_SITE])
        self.assertEqual(best, SitePriority.MULTI_EVIDENCE_WITH_EXPERIMENT.value)
        if SHELL_SITE in self.by_site:
            worst = min(p["experimental_priority"]
                        for p in self.by_site[SHELL_SITE])
            self.assertGreater(worst, best)

    def test_shell_only_site_has_no_sourced_substitution(self):
        reasons = {(e["position_index"], e["reason"])
                   for e in self.out.data["excluded_sites"]}
        self.assertIn((SHELL_SITE, "no_sourced_substitution"), reasons)

    def test_family_signal_supplies_the_alternative_residue(self):
        mutants = {m["mutant"] for p in self.by_site[FAMILY_SITE]
                   for m in p["mutations"]
                   if m["position_index"] == FAMILY_SITE}
        self.assertIn("F", mutants)

    def test_steric_clash_offers_only_smaller_residues(self):
        for proposal in self.by_site[CLASH_SITE]:
            for mutation in proposal["mutations"]:
                if mutation["position_index"] != CLASH_SITE:
                    continue
                self.assertIn(mutation["mutant"], {"A", "S", "V", "T", "C"})
                self.assertTrue(
                    any("volume-reduction design rule" in text
                        for text in proposal["contradicting_evidence"]))

    def test_homolog_precedent_is_flagged_as_untransferred(self):
        proposals = [p for p in self.by_site[PRECEDENT_SITE]
                     if len(p["mutations"]) == 1]
        self.assertTrue(proposals)
        self.assertTrue(any(
            any("measured on a homologue" in text
                for text in p["contradicting_evidence"]) for p in proposals))

    def test_author_numbering_is_offset_and_named(self):
        reference = self.out.data["numbering_reference"]["C1"]
        self.assertIn("author numbering", reference)
        for proposal in self.out.data["proposals"]:
            for mutation in proposal["mutations"]:
                self.assertEqual(mutation["position_author"],
                                 mutation["position_index"] + 101)

    def test_all_three_axes_are_pre_stated(self):
        for proposal in self.out.data["proposals"]:
            axes = {e["axis"] for e in proposal["axis_expectations"]}
            self.assertEqual(
                axes, {"substrate_fit", "catalytic_function",
                       "stability_expression_risk"})

    def test_every_proposal_names_a_property_it_might_damage(self):
        for proposal in self.out.data["proposals"]:
            self.assertTrue(proposal["possible_cost"])

    def test_combinations_carry_their_single_mutant_controls(self):
        ids = {p["proposal_id"] for p in self.out.data["proposals"]}
        combos = [p for p in self.out.data["proposals"]
                  if len(p["mutations"]) > 1]
        self.assertTrue(combos, "expected at least one combination variant")
        for combo in combos:
            self.assertTrue(combo["decomposition_controls"])
            for control in combo["decomposition_controls"]:
                self.assertIn(control, ids)


class NumberingFallbackTests(unittest.TestCase):
    def test_unobserved_site_switches_the_whole_library_to_1_based(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            rmap = make_residue_map(unobserved=(FAMILY_SITE,))
            out = run_interface(
                ctx, **base_kwargs(parent, rmap, make_template()))
            reference = out.data["numbering_reference"]["C1"]
            self.assertIn("1-based", reference)
            for proposal in out.data["proposals"]:
                self.assertEqual(proposal["numbering_reference"], reference)
                for mutation in proposal["mutations"]:
                    self.assertEqual(mutation["position_author"],
                                     mutation["position_index"] + 1)

    def test_structure_only_site_requires_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            rmap = make_residue_map(unobserved=(CLASH_SITE,))
            out = run_interface(
                ctx, **base_kwargs(parent, rmap, make_template()))
            touched = {m["position_index"] for p in out.data["proposals"]
                       for m in p["mutations"]}
            self.assertNotIn(CLASH_SITE, touched)
            reasons = {(e["position_index"], e["reason"])
                       for e in out.data["excluded_sites"]}
            self.assertIn((CLASH_SITE, "wild_type_unverified"), reasons)


class PrecedentIntegrityTests(unittest.TestCase):
    def test_precedent_with_the_wrong_wild_type_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            kwargs = base_kwargs(parent, make_residue_map(), make_template())
            kwargs["experimental_precedents"] = {parent.candidate_id: [
                ExperimentalPrecedent(
                    detail="W->A reported to widen the pocket",
                    source="PMID:00000003", candidate_index=PRECEDENT_SITE,
                    wild_type="W", mutant="A")]}
            out = run_interface(ctx, **kwargs)
            reasons = {(e["position_index"], e["reason"])
                       for e in out.data["excluded_sites"]}
            self.assertIn((PRECEDENT_SITE, "precedent_wild_type_mismatch"),
                          reasons)
            mutants = {m["mutant"] for p in out.data["proposals"]
                       for m in p["mutations"]
                       if m["position_index"] == PRECEDENT_SITE}
            self.assertNotIn("A", mutants)

    def test_unsourced_template_precedent_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            parent = make_parent()
            template = make_template(known_beneficial_mutations=[{
                "candidate_index": CLASH_SITE, "mutant": "A",
                "notes": "everybody knows this one helps"}])
            out = run_interface(
                ctx, **base_kwargs(parent, make_residue_map(), template))
            reasons = {e["reason"] for e in out.data["excluded_sites"]}
            self.assertIn("template_entry_unsourced", reasons)


class _FakeMPNN(LigandMPNNAdapter):
    name = "ligandmpnn"
    reaches_network = False

    def __init__(self, sequences, version: str = "v_32_010"):
        self._sequences = list(sequences)
        self._version = version
        self.last_request: LigandMPNNRequest | None = None

    def available(self) -> bool:
        return True

    def propose(self, request: LigandMPNNRequest) -> LigandMPNNResult:
        self.last_request = request
        return LigandMPNNResult(sequences=self._sequences,
                                model_version=self._version)


class _RemoteMPNN(_FakeMPNN):
    reaches_network = True


def _with_substitution(index: int, letter: str, sequence: str = SEQ) -> str:
    chars = list(sequence)
    chars[index] = letter
    return "".join(chars)


class LigandMPNNTests(unittest.TestCase):
    def test_default_adapter_reports_unavailable(self):
        adapter = MissingLigandMPNN()
        self.assertFalse(adapter.available())
        with self.assertRaises(ToolUnavailableError):
            adapter.propose(LigandMPNNRequest(
                parent_candidate_id="C1", sequence=SEQ))

    def test_required_but_absent_fails_the_step_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run_interface(
                ctx, require_ligandmpnn=True,
                **base_kwargs(make_parent(), make_residue_map(),
                              make_template()))
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "tool_unavailable"
                                for f in out.blockers))

    def test_absent_but_optional_degrades_to_partial_with_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run_interface(
                ctx, ligandmpnn=MissingLigandMPNN(),
                **base_kwargs(make_parent(), make_residue_map(),
                              make_template()))
            self.assertIs(out.status, Status.PARTIAL)
            self.assertTrue(any(f.code == "ligandmpnn_unavailable"
                                for f in out.qc_flags))
            self.assertGreater(out.data["n_proposals"], 0)

    def test_remote_adapter_is_blocked_while_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            adapter = _RemoteMPNN([_with_substitution(CLASH_SITE, "A")])
            out = run_interface(
                ctx, ligandmpnn=adapter,
                **base_kwargs(make_parent(), make_residue_map(),
                              make_template()))
            self.assertTrue(any(f.code == "ligandmpnn_network_blocked"
                                for f in out.qc_flags))
            self.assertIsNone(adapter.last_request)

    def test_remote_adapter_online_still_needs_disclosure_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task()
            ctx = RunContext(
                task=task, workdir=Path(tmp),
                manifest=RunManifest(run_id="R1", task_id=task.task_id),
                policy=ExecutionPolicy(allow_network=True))
            adapter = _RemoteMPNN([_with_substitution(CLASH_SITE, "A")])
            out = run_interface(
                ctx, ligandmpnn=adapter,
                **base_kwargs(make_parent(), make_residue_map(),
                              make_template()))
            self.assertTrue(any(f.code == "ligandmpnn_disclosure_blocked"
                                for f in out.blockers))
            self.assertIsNone(adapter.last_request)

    def test_fixed_positions_are_passed_and_re_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            adapter = _FakeMPNN([_with_substitution(CLASH_SITE, "A")])
            out = run_interface(
                ctx, ligandmpnn=adapter,
                **base_kwargs(make_parent(), make_residue_map(),
                              make_template()))
            self.assertIsNotNone(adapter.last_request)
            self.assertEqual(set(adapter.last_request.fixed_indices),
                             {FROZEN_TYR, FROZEN_GLY})
            designed = [p for p in out.data["proposals"]
                        if (p["generator"] or "").startswith("ligandmpnn")]
            self.assertTrue(designed)
            for proposal in designed:
                self.assertTrue(any(
                    "not a demonstrated improvement" in text
                    for text in proposal["contradicting_evidence"]))

    def test_design_that_edits_a_frozen_residue_is_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            adapter = _FakeMPNN([_with_substitution(FROZEN_TYR, "F")])
            out = run_interface(
                ctx, ligandmpnn=adapter,
                **base_kwargs(make_parent(), make_residue_map(),
                              make_template()))
            self.assertTrue(any(f.code == "ligandmpnn_touched_frozen_residue"
                                for f in out.blockers))
            touched = {m["position_index"] for p in out.data["proposals"]
                       for m in p["mutations"]}
            self.assertNotIn(FROZEN_TYR, touched)

    def test_design_longer_than_the_parent_is_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            adapter = _FakeMPNN([SEQ + "AAA"])
            out = run_interface(
                ctx, ligandmpnn=adapter,
                **base_kwargs(make_parent(), make_residue_map(),
                              make_template()))
            self.assertTrue(any(f.code == "ligandmpnn_length_mismatch"
                                for f in out.blockers))


class ArtifactTests(unittest.TestCase):
    def test_tsv_has_one_row_per_proposal_and_names_the_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run_interface(
                ctx, **base_kwargs(make_parent(), make_residue_map(),
                                   make_template()))
            proposals = out.artifact("mutation_proposals")
            excluded = out.artifact("excluded_sites")
            self.assertIsNotNone(proposals)
            self.assertIsNotNone(excluded)
            with open(proposals.path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            self.assertEqual(len(rows), out.data["n_proposals"])
            self.assertIn("contradicting_evidence", rows[0])
            self.assertIn("axis_substrate_fit", rows[0])
            with open(excluded.path, encoding="utf-8") as fh:
                dropped = list(csv.DictReader(fh, delimiter="\t"))
            self.assertTrue(any(r["reason_excluded"] == "frozen_role"
                                for r in dropped))

    def test_provenance_records_the_ladder_and_the_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run_interface(
                ctx, **base_kwargs(make_parent(), make_residue_map(),
                                   make_template()))
            self.assertIsNotNone(out.provenance)
            self.assertEqual(out.provenance.tool, "propose_mutations")
            self.assertIsNotNone(out.provenance.random_seed)
            self.assertIn("steric_relief_ladder", out.provenance.parameters)
            self.assertFalse(out.provenance.parameters["allow_network"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
