"""Tests for the ``annotate_family`` interface.

The fixtures build three deliberately non-homologous synthetic families -- an
SDR, an MDR/ADH and an AKR -- from disjoint residue alphabets, with their own
motifs, their own catalytic residues and their own cofactor-recognition rule.
That is the only way to test the property that matters here: that one family's
mechanism can never be applied to another's sequence, and that a family is
called from several agreeing signals rather than from a motif hit.

Every position in the fixtures is implanted at a stated coordinate, so each
expected residue token (``Y60``, ``C40``, ``K85``) can be counted by hand.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Status
from eagent.errors import FabricationGuardError, TemplateError
from eagent.provenance import RunManifest
from eagent.schemas import (
    Budget, Candidate, CatalyticTemplate, CofactorState, ConfidenceLevel,
    FamilyTemplate, SequenceRecord, TaskSpec, TemplateProvenance,
    TemplateSourceType,
)
from eagent.tools.handoff import CANDIDATES_KEY, as_candidates
from eagent.tools.annotate_family import (
    AnnotateFamily,
    AnnotationPolicy,
    CofactorRecognitionRule,
    DomainHit,
    FamilyHypothesis,
    HypothesisSet,
    ReferenceSequence,
    build_similarity_network,
    compile_motif,
    map_catalytic_roles,
    map_reference_positions,
    match_motifs,
    prosite_to_regex,
)

LENGTH = 150


def _scaffold(cycle: str, length: int = LENGTH) -> list[str]:
    return [cycle[i % len(cycle)] for i in range(length)]


def _implant(chars: list[str], position_1based: int, text: str) -> None:
    for i, letter in enumerate(text):
        chars[position_1based - 1 + i] = letter


# ---------------------------------------------------------------------------
# SDR: Rossmann fold, Ser-Tyr-Lys-Asn, NADPH selected by a basic residue.
# ---------------------------------------------------------------------------
def _sdr_reference() -> str:
    chars = _scaffold("AVLTPEQ")
    _implant(chars, 10, "GAAAGAG")   # G-x(3)-G-x-G at 10..16
    _implant(chars, 15, "R")         # basic residue: NADPH recognition
    _implant(chars, 30, "N")
    _implant(chars, 45, "S")
    _implant(chars, 60, "YAAAK")     # Y60 ... K64
    return "".join(chars)


# ---------------------------------------------------------------------------
# MDR/ADH: catalytic zinc on Cys-His-Cys, no Rossmann glycine motif.
# ---------------------------------------------------------------------------
def _mdr_reference() -> str:
    chars = _scaffold("LQWFDI")
    _implant(chars, 40, "CGHCE")     # C40 G41 H42 C43 E44
    _implant(chars, 90, "GHEFG")     # second diagnostic motif
    return "".join(chars)


# ---------------------------------------------------------------------------
# AKR: (beta/alpha)8 barrel, Asp-Tyr-Lys-His tetrad, no Rossmann motif.
# ---------------------------------------------------------------------------
def _akr_reference() -> str:
    chars = _scaffold("IVLQAF")
    _implant(chars, 25, "D")
    _implant(chars, 55, "Y")
    _implant(chars, 65, "WPQNF")
    _implant(chars, 85, "K")
    _implant(chars, 100, "RMTEW")
    _implant(chars, 115, "H")
    return "".join(chars)


SDR_REF = _sdr_reference()
MDR_REF = _mdr_reference()
AKR_REF = _akr_reference()


def _mutate(sequence: str, changes: dict[int, str]) -> str:
    """Point-mutate by 1-based position, using letters already in the scaffold."""
    chars = list(sequence)
    for pos, letter in changes.items():
        chars[pos - 1] = letter
    return "".join(chars)


def _provenance(identifier: str) -> TemplateProvenance:
    return TemplateProvenance(
        source_type=TemplateSourceType.MECHANISM_LITERATURE,
        identifiers=[identifier], curated_by="test fixture",
    )


SDR_FAMILY = FamilyTemplate(
    template_id="fam.sdr.v1", family_name="SDR",
    pfam_ids=["PF00106"],
    conserved_motifs=[
        {"name": "rossmann_glycine_rich", "pattern": "G-x(3)-G-x-G",
         "role": "cofactor binding", "evidence": "PMID:00000010"},
        {"name": "sdr_catalytic_yxxxk", "pattern": "Y-x(3)-K",
         "role": "catalytic pair", "evidence": "PMID:00000010"},
    ],
    cofactor_preference={"NADPH": "family survey", "NADH": "family survey"},
    catalytic_template_ids=["cat.sdr.v1"],
    provenance=_provenance("PMID:00000010"),
)

SDR_CATALYTIC = CatalyticTemplate(
    template_id="cat.sdr.v1", family_name="SDR",
    mechanism_summary="Tyr general acid, Lys lowers Tyr pKa, Ser orients substrate",
    catalytic_residues=[
        {"label": "catalytic_asn", "residue_types": ["N"], "role": "proton relay",
         "functional_atoms": ["OD1"], "evidence": "PMID:00000010"},
        {"label": "catalytic_ser", "residue_types": ["S"], "role": "substrate anchor",
         "functional_atoms": ["OG"], "evidence": "PMID:00000010"},
        {"label": "catalytic_tyr", "residue_types": ["TYR"], "role": "general acid",
         "functional_atoms": ["OH"], "evidence": "PMID:00000010"},
        {"label": "catalytic_lys", "residue_types": ["K"], "role": "pKa modulation",
         "functional_atoms": ["NZ"], "evidence": "PMID:00000010"},
    ],
    required_cofactor="NADPH", required_cofactor_state=CofactorState.REDUCED,
    cofactor_ligand_codes=["NDP"],
    provenance=_provenance("PMID:00000010"),
)

SDR_REFERENCE = ReferenceSequence(
    accession="SDR_REF", sequence=SDR_REF, source="test fixture: implanted",
    role_to_number={"catalytic_asn": 30, "catalytic_ser": 45,
                    "catalytic_tyr": 60, "catalytic_lys": 64},
)

SDR_NADPH_RULE = CofactorRecognitionRule(
    cofactor="NADPH", family_template_id="fam.sdr.v1",
    reference_positions=[15], accepted_residues=["R", "K"],
    evidence="PMID:00000011 basic residue in the glycine-rich region selects 2'-phosphate",
    description="SDR-specific; has no counterpart in AKR or MDR",
)

MDR_FAMILY = FamilyTemplate(
    template_id="fam.mdr.v1", family_name="MDR/ADH",
    pfam_ids=["PF08240"],
    conserved_motifs=[
        {"name": "zinc_binding_cgHC", "pattern": "C-G-H-C",
         "role": "catalytic zinc", "evidence": "PMID:00000020"},
        {"name": "mdr_second_site", "pattern": "G-H-E-x-G",
         "role": "structural", "evidence": "PMID:00000020"},
    ],
    cofactor_preference={"NADH": "family survey"},
    catalytic_template_ids=["cat.mdr.v1"],
    provenance=_provenance("PMID:00000020"),
)

MDR_CATALYTIC = CatalyticTemplate(
    template_id="cat.mdr.v1", family_name="MDR/ADH",
    mechanism_summary="Zinc-polarised carbonyl, hydride from NADH",
    catalytic_residues=[
        {"label": "zinc_cys_1", "residue_types": ["C"], "role": "zinc ligand",
         "functional_atoms": ["SG"], "evidence": "PMID:00000020"},
        {"label": "zinc_his", "residue_types": ["H"], "role": "zinc ligand",
         "functional_atoms": ["NE2"], "evidence": "PMID:00000020"},
        {"label": "zinc_cys_2", "residue_types": ["C"], "role": "zinc ligand",
         "functional_atoms": ["SG"], "evidence": "PMID:00000020"},
        {"label": "proton_relay_glu", "residue_types": ["E"], "role": "proton relay",
         "functional_atoms": ["OE1"], "evidence": "PMID:00000020"},
    ],
    required_cofactor="NADH", required_cofactor_state=CofactorState.REDUCED,
    metals=["ZN"],
    provenance=_provenance("PMID:00000020"),
)

MDR_REFERENCE = ReferenceSequence(
    accession="MDR_REF", sequence=MDR_REF, source="test fixture: implanted",
    role_to_number={"zinc_cys_1": 40, "zinc_his": 42, "zinc_cys_2": 43,
                    "proton_relay_glu": 44},
)

MDR_NADH_RULE = CofactorRecognitionRule(
    cofactor="NADH", family_template_id="fam.mdr.v1",
    reference_positions=[44], accepted_residues=["E", "D"],
    evidence="PMID:00000021 acidic residue at the adenine ribose selects NADH",
)

AKR_FAMILY = FamilyTemplate(
    template_id="fam.akr.v1", family_name="AKR",
    pfam_ids=["PF00248"],
    conserved_motifs=[
        {"name": "akr_loop_a", "pattern": "W-P-Q-N-F", "role": "substrate loop",
         "evidence": "PMID:00000030"},
        {"name": "akr_loop_b", "pattern": "R-M-T-E-W", "role": "cofactor loop",
         "evidence": "PMID:00000030"},
    ],
    cofactor_preference={"NADPH": "family survey"},
    catalytic_template_ids=["cat.akr.v1"],
    provenance=_provenance("PMID:00000030"),
)

AKR_CATALYTIC = CatalyticTemplate(
    template_id="cat.akr.v1", family_name="AKR",
    mechanism_summary="Asp-Tyr-Lys-His tetrad, hydride from NADPH 4-pro-R",
    catalytic_residues=[
        {"label": "tetrad_asp", "residue_types": ["D"], "role": "orients Lys",
         "functional_atoms": ["OD1"], "evidence": "PMID:00000030"},
        {"label": "tetrad_tyr", "residue_types": ["Y"], "role": "general acid",
         "functional_atoms": ["OH"], "evidence": "PMID:00000030"},
        {"label": "tetrad_lys", "residue_types": ["K"], "role": "pKa modulation",
         "functional_atoms": ["NZ"], "evidence": "PMID:00000030"},
        {"label": "tetrad_his", "residue_types": ["H"], "role": "substrate binding",
         "functional_atoms": ["NE2"], "evidence": "PMID:00000030"},
    ],
    required_cofactor="NADPH", required_cofactor_state=CofactorState.REDUCED,
    provenance=_provenance("PMID:00000030"),
)

AKR_REFERENCE = ReferenceSequence(
    accession="AKR_REF", sequence=AKR_REF, source="test fixture: implanted",
    role_to_number={"tetrad_asp": 25, "tetrad_tyr": 55, "tetrad_lys": 85,
                    "tetrad_his": 115},
)


def _hypotheses() -> HypothesisSet:
    return HypothesisSet([
        FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC, SDR_REFERENCE, (SDR_NADPH_RULE,)),
        FamilyHypothesis(MDR_FAMILY, MDR_CATALYTIC, MDR_REFERENCE, (MDR_NADH_RULE,)),
        FamilyHypothesis(AKR_FAMILY, AKR_CATALYTIC, AKR_REFERENCE),
    ])


def _record(candidate_id: str, sequence: str) -> SequenceRecord:
    return SequenceRecord(candidate_id=candidate_id, sequence=sequence,
                          accession=candidate_id, source_database="uniprotkb",
                          database_version="2024_01")


#: Candidates: point mutations at non-functional positions, using letters that
#: already occur in the family's own scaffold so no motif can be created.
SDR_CAND_A = _mutate(SDR_REF, {5: "P", 120: "L", 140: "T"})
SDR_CAND_B = _mutate(SDR_REF, {7: "V", 125: "A", 138: "Q"})
SDR_CAND_NADH = _mutate(SDR_CAND_A, {15: "E"})     # acidic -> NADPH rule fails
MDR_CAND = _mutate(MDR_REF, {5: "W", 120: "L", 140: "Q"})
AKR_CAND = _mutate(AKR_REF, {5: "V", 130: "L", 145: "Q"})
NOISE_CAND = "".join(_scaffold("AILVF"))


def _ctx(tmp: Path) -> RunContext:
    task = TaskSpec(task_id="t-annot", budget=Budget(initial_sequence_target=10))
    manifest = RunManifest(run_id="r1", task_id="t-annot", global_seed=11)
    return RunContext(task=task, workdir=tmp, manifest=manifest,
                      policy=ExecutionPolicy())


# ---------------------------------------------------------------------------
# motif patterns
# ---------------------------------------------------------------------------

class TestMotifPatterns(unittest.TestCase):
    def test_prosite_translation(self) -> None:
        self.assertEqual(prosite_to_regex("G-x(3)-G-x-G"), "G.{3}G.G")
        self.assertEqual(prosite_to_regex("[ST]-x(2,4)-{P}-Y"), "[ST].{2,4}[^P]Y")
        self.assertEqual(prosite_to_regex("<M-K>"), "^MK$")
        self.assertEqual(prosite_to_regex("re:Y.{3}K"), "Y.{3}K")

    def test_bad_pattern_raises_instead_of_never_matching(self) -> None:
        for bad in ("G-x(3", "[", "G-Z-G", "{}-Y", "G-1-G"):
            with self.assertRaises(TemplateError, msg=bad):
                compile_motif(bad)

    def test_motif_without_a_pattern_is_an_error(self) -> None:
        with self.assertRaises(TemplateError):
            match_motifs("MKV", [{"name": "nameless"}])

    def test_repeated_occurrence_counts_once(self) -> None:
        found = match_motifs("YAAAKQQQYAAAK",
                             [{"name": "yxxxk", "pattern": "Y-x(3)-K"}])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["start"], 1)

    def test_family_motifs_do_not_cross_match_the_fixtures(self) -> None:
        # The fixtures are only meaningful if each family's motifs are absent
        # from the other two scaffolds.
        pairs = [(SDR_FAMILY, SDR_REF), (MDR_FAMILY, MDR_REF), (AKR_FAMILY, AKR_REF)]
        for family, own in pairs:
            self.assertEqual(len(match_motifs(own, family.conserved_motifs)), 2,
                             f"{family.family_name} should match its own motifs")
            for other_family, other in pairs:
                if other is own:
                    continue
                self.assertEqual(
                    match_motifs(other, family.conserved_motifs), [],
                    f"{family.family_name} motifs must not match "
                    f"{other_family.family_name}",
                )


# ---------------------------------------------------------------------------
# hypothesis construction: the structural guards
# ---------------------------------------------------------------------------

class TestHypothesisGuards(unittest.TestCase):
    def test_valid_hypothesis_builds(self) -> None:
        h = FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC, SDR_REFERENCE)
        self.assertEqual(h.template_id, "fam.sdr.v1")
        self.assertEqual(h.min_independent_signals, 3)
        self.assertIn("catalytic_tyr", h.role_labels)

    def test_cross_family_catalytic_template_is_impossible(self) -> None:
        with self.assertRaises(FabricationGuardError) as cm:
            FamilyHypothesis(SDR_FAMILY, AKR_CATALYTIC, SDR_REFERENCE)
        self.assertIn("AKR", str(cm.exception))

    def test_catalytic_template_not_listed_by_the_family_is_refused(self) -> None:
        other = SDR_FAMILY.model_copy(update={"catalytic_template_ids": ["cat.sdr.v9"]})
        with self.assertRaises(TemplateError):
            FamilyHypothesis(other, SDR_CATALYTIC, SDR_REFERENCE)

    def test_reference_missing_a_role_is_refused(self) -> None:
        ref = ReferenceSequence(
            accession="X", sequence=SDR_REF, source="test",
            role_to_number={"catalytic_tyr": 60},
        )
        with self.assertRaises(TemplateError) as cm:
            FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC, ref)
        self.assertIn("catalytic_ser", str(cm.exception))

    def test_off_by_one_reference_numbering_is_caught_at_load(self) -> None:
        shifted = ReferenceSequence(
            accession="X", sequence=SDR_REF, source="test",
            role_to_number={"catalytic_asn": 31, "catalytic_ser": 46,
                            "catalytic_tyr": 61, "catalytic_lys": 65},
        )
        with self.assertRaises(TemplateError) as cm:
            FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC, shifted)
        self.assertIn("off-by-one", str(cm.exception))

    def test_reference_position_outside_the_sequence_is_refused(self) -> None:
        with self.assertRaises(TemplateError):
            ReferenceSequence(accession="X", sequence="MKV",
                              source="test").letter_at(9)

    def test_reference_without_a_source_is_refused(self) -> None:
        with self.assertRaises(TemplateError):
            ReferenceSequence(accession="X", sequence=SDR_REF, source="")

    def test_cofactor_rule_of_another_family_cannot_be_attached(self) -> None:
        with self.assertRaises(FabricationGuardError) as cm:
            FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC, SDR_REFERENCE,
                             (MDR_NADH_RULE,))
        self.assertIn("fam.mdr.v1", str(cm.exception))

    def test_unsourced_cofactor_rule_is_refused(self) -> None:
        with self.assertRaises(TemplateError):
            CofactorRecognitionRule(cofactor="NADPH", family_template_id="fam.sdr.v1",
                                    reference_positions=[15],
                                    accepted_residues=["R"], evidence="   ")

    def test_cofactor_rule_needs_real_residue_letters(self) -> None:
        with self.assertRaises(TemplateError):
            CofactorRecognitionRule(cofactor="NADPH", family_template_id="fam.sdr.v1",
                                    reference_positions=[15],
                                    accepted_residues=["X"], evidence="PMID:1")

    def test_one_catalytic_template_cannot_serve_two_families(self) -> None:
        twin = SDR_FAMILY.model_copy(update={"template_id": "fam.sdr.v2"})
        a = FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC, SDR_REFERENCE)
        b = FamilyHypothesis(twin, SDR_CATALYTIC, SDR_REFERENCE)
        with self.assertRaises(FabricationGuardError):
            HypothesisSet([a, b])

    def test_hypothesis_set_lookup_refuses_an_unknown_id(self) -> None:
        hset = _hypotheses()
        self.assertEqual(hset.ids(), ["fam.akr.v1", "fam.mdr.v1", "fam.sdr.v1"])
        with self.assertRaises(TemplateError):
            hset.get("fam.unknown.v1")


# ---------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------

class TestMapping(unittest.TestCase):
    def test_reference_positions_survive_an_indel(self) -> None:
        reference = "MKVLAWYTTGGQ"
        candidate = "MKVLAAAWYTTGGQ"          # two residues inserted after 5
        mapping, aln = map_reference_positions(candidate, reference)
        self.assertEqual(mapping[1], 0)
        self.assertEqual(candidate[mapping[6]], reference[5])
        self.assertGreater(aln.identity, 0.8)

    def test_roles_map_onto_the_candidate_with_candidate_numbering(self) -> None:
        h = FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC, SDR_REFERENCE)
        ref_to_index, _ = map_reference_positions(SDR_CAND_A, SDR_REF)
        mapping = map_catalytic_roles(SDR_CAND_A, h, ref_to_index)
        self.assertEqual(mapping.catalytic_template_id, "cat.sdr.v1")
        self.assertEqual(mapping.role_to_residue["catalytic_tyr"], "Y60")
        self.assertEqual(mapping.role_to_residue["catalytic_lys"], "K64")
        self.assertEqual(mapping.role_to_residue["catalytic_ser"], "S45")
        self.assertEqual(mapping.role_to_index["catalytic_tyr"], 59)
        self.assertEqual(mapping.missing_roles, [])
        self.assertTrue(mapping.is_complete)

    def test_conservative_substitution_is_flagged_not_counted_as_present(self) -> None:
        h = FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC, SDR_REFERENCE)
        mutant = _mutate(SDR_CAND_A, {45: "T"})      # Ser -> Thr
        ref_to_index, _ = map_reference_positions(mutant, SDR_REF)
        mapping = map_catalytic_roles(mutant, h, ref_to_index)
        self.assertIn("catalytic_ser", mapping.missing_roles)
        self.assertEqual(mapping.substituted_roles["catalytic_ser"], "T45")
        self.assertFalse(mapping.is_complete)

    def test_wrong_family_mapping_finds_nothing(self) -> None:
        akr = FamilyHypothesis(AKR_FAMILY, AKR_CATALYTIC, AKR_REFERENCE)
        ref_to_index, _ = map_reference_positions(SDR_CAND_A, AKR_REF)
        mapping = map_catalytic_roles(SDR_CAND_A, akr, ref_to_index)
        self.assertEqual(mapping.catalytic_template_id, "cat.akr.v1")
        self.assertTrue(mapping.missing_roles)
        self.assertEqual(mapping.role_to_residue, {})

    def test_three_letter_residue_types_are_accepted(self) -> None:
        # SDR_CATALYTIC declares the Tyr as "TYR"; it must still map.
        h = FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC, SDR_REFERENCE)
        ref_to_index, _ = map_reference_positions(SDR_REF, SDR_REF)
        mapping = map_catalytic_roles(SDR_REF, h, ref_to_index)
        self.assertEqual(mapping.role_to_residue["catalytic_tyr"], "Y60")


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

class TestNetwork(unittest.TestCase):
    def test_edges_need_identity_and_coverage(self) -> None:
        full = SDR_CAND_A
        fragment = SDR_CAND_A[40:90]            # perfect identity, poor coverage
        net = build_similarity_network([("full", full), ("frag", fragment)],
                                       identity_threshold_pct=40.0,
                                       coverage_threshold=0.8)
        self.assertEqual(net.edges, [])
        self.assertNotEqual(net.clusters["full"], net.clusters["frag"])

    def test_related_sequences_join_unrelated_ones_do_not(self) -> None:
        net = build_similarity_network(
            [("sdrA", SDR_CAND_A), ("sdrB", SDR_CAND_B), ("mdr", MDR_CAND),
             ("akr", AKR_CAND)],
            identity_threshold_pct=40.0, coverage_threshold=0.8,
        )
        self.assertEqual(net.clusters["sdrA"], net.clusters["sdrB"])
        self.assertNotEqual(net.clusters["sdrA"], net.clusters["mdr"])
        self.assertNotEqual(net.clusters["sdrA"], net.clusters["akr"])
        self.assertNotEqual(net.clusters["mdr"], net.clusters["akr"])

    def test_budget_exhaustion_is_reported(self) -> None:
        net = build_similarity_network(
            [("a", SDR_CAND_A), ("b", SDR_CAND_B), ("c", MDR_CAND)],
            max_alignments=1, kmer_prefilter=None,
        )
        self.assertTrue(net.budget_exhausted)
        self.assertEqual(len(net.clusters), 3)


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------

class TestAnnotateFamilyInterface(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.iface = AnnotateFamily()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, records, *, domains=True, policy=None):
        ctx = _ctx(self.tmp)
        domain_hits = {}
        if domains:
            domain_hits = {
                "sdrA": [DomainHit(accession="PF00106", source="Pfam 36.0")],
                "sdrB": [DomainHit(accession="PF00106", source="Pfam 36.0")],
                "sdrNADH": [DomainHit(accession="PF00106", source="Pfam 36.0")],
                "mdr": [DomainHit(accession="PF08240", source="Pfam 36.0")],
                "akr": [DomainHit(accession="PF00248", source="Pfam 36.0")],
            }
        result = self.iface.run(ctx, sequences=records, hypotheses=_hypotheses(),
                                domain_hits=domain_hits, policy=policy)
        return ctx, result

    def _three_families(self):
        return [_record("sdrA", SDR_CAND_A), _record("mdr", MDR_CAND),
                _record("akr", AKR_CAND)]

    def test_each_family_is_called_correctly_and_never_cross_applied(self) -> None:
        _, result = self._run(self._three_families())
        self.assertTrue(result.status.usable, result.message)
        rows = {r["candidate_id"]: r for r in result.data["annotations"]}
        self.assertEqual(rows["sdrA"]["family_name"], "SDR")
        self.assertEqual(rows["sdrA"]["family_template_id"], "fam.sdr.v1")
        self.assertEqual(rows["sdrA"]["catalytic_template_id"], "cat.sdr.v1")
        self.assertEqual(rows["mdr"]["catalytic_template_id"], "cat.mdr.v1")
        self.assertEqual(rows["akr"]["catalytic_template_id"], "cat.akr.v1")
        # The catalytic roles are the family's own, at the implanted positions.
        self.assertIn("catalytic_tyr=Y60", rows["sdrA"]["catalytic_roles_mapped"])
        self.assertIn("zinc_cys_1=C40", rows["mdr"]["catalytic_roles_mapped"])
        self.assertIn("tetrad_lys=K85", rows["akr"]["catalytic_roles_mapped"])
        # No SDR role token may appear on the AKR candidate's row and vice versa.
        self.assertNotIn("catalytic_tyr", rows["akr"]["catalytic_roles_mapped"])
        self.assertNotIn("tetrad", rows["sdrA"]["catalytic_roles_mapped"])

    def test_confidence_comes_from_several_independent_signals(self) -> None:
        _, result = self._run(self._three_families())
        rows = {r["candidate_id"]: r for r in result.data["annotations"]}
        for cid in ("sdrA", "mdr", "akr"):
            signals = rows[cid]["signals_supporting"].split(",")
            self.assertIn("domain_architecture", signals)
            self.assertIn("conserved_motifs", signals)
            self.assertIn("catalytic_residue_correspondence", signals)
            self.assertIn("overall_identity_to_reference", signals)
            self.assertGreaterEqual(rows[cid]["n_signals_supporting"], 3)
            self.assertEqual(rows[cid]["confidence"], ConfidenceLevel.STRONG.value)

    def test_one_signal_alone_does_not_make_a_family(self) -> None:
        # Domain evidence withheld and the catalytic residues knocked out: the
        # motifs alone must not produce a family call.
        knocked = _mutate(SDR_CAND_A, {30: "A", 45: "A"})
        _, result = self._run([_record("sdrA", knocked), _record("mdr", MDR_CAND)],
                              domains=False)
        rows = {r["candidate_id"]: r for r in result.data["annotations"]}
        self.assertLess(rows["sdrA"]["n_signals_supporting"], 3)
        self.assertIsNone(rows["sdrA"]["family_template_id"])
        self.assertIn("sdrA", result.data["unassigned"])
        self.assertIs(result.status, Status.PARTIAL)
        self.assertIn("family_unassigned", {f.code for f in result.qc_flags})

    def test_unrelated_sequence_is_left_unassigned(self) -> None:
        records = self._three_families() + [_record("noise", NOISE_CAND)]
        _, result = self._run(records)
        self.assertIn("noise", result.data["unassigned"])
        rows = {r["candidate_id"]: r for r in result.data["annotations"]}
        self.assertIsNone(rows["noise"]["family_name"])
        self.assertEqual(rows["noise"]["catalytic_template_id"], None)

    def test_cofactor_preference_carries_its_evidence(self) -> None:
        _, result = self._run([_record("sdrA", SDR_CAND_A), _record("mdr", MDR_CAND)])
        rows = {r["candidate_id"]: r for r in result.data["annotations"]}
        self.assertEqual(rows["sdrA"]["cofactor_preference"], "NADPH")
        self.assertIn("PMID:00000011", rows["sdrA"]["cofactor_preference_evidence"])
        self.assertIn("fam.sdr.v1", rows["sdrA"]["cofactor_preference_evidence"])
        self.assertEqual(rows["mdr"]["cofactor_preference"], "NADH")
        self.assertIn("PMID:00000021", rows["mdr"]["cofactor_preference_evidence"])

    def test_unsupported_cofactor_preference_stays_none(self) -> None:
        # The basic residue is replaced by an acidic one: the SDR rule fails and
        # the family template declares two preferences, so nothing is claimed.
        _, result = self._run([_record("sdrNADH", SDR_CAND_NADH),
                               _record("mdr", MDR_CAND)])
        rows = {r["candidate_id"]: r for r in result.data["annotations"]}
        self.assertEqual(rows["sdrNADH"]["family_template_id"], "fam.sdr.v1")
        self.assertIsNone(rows["sdrNADH"]["cofactor_preference"])
        self.assertIsNone(rows["sdrNADH"]["cofactor_preference_evidence"])

    def test_identity_signal_is_withheld_without_a_sourced_floor(self) -> None:
        ctx = _ctx(self.tmp)
        single = HypothesisSet([FamilyHypothesis(SDR_FAMILY, SDR_CATALYTIC,
                                                 SDR_REFERENCE, (SDR_NADPH_RULE,))])
        result = self.iface.run(ctx, sequences=[_record("sdrA", SDR_CAND_A)],
                                hypotheses=single)
        row = result.data["annotations"][0]
        self.assertNotIn("overall_identity_to_reference", row["signals_supporting"])
        self.assertIn("identity_signal_withheld", {f.code for f in result.qc_flags})

        result2 = self.iface.run(
            ctx, sequences=[_record("sdrA", SDR_CAND_A)], hypotheses=single,
            policy=AnnotationPolicy(min_identity_for_signal=30.0,
                                    source="operator: curated for this family"),
        )
        self.assertIn("overall_identity_to_reference",
                      result2.data["annotations"][0]["signals_supporting"])

    def test_missing_domain_evidence_is_an_uncertainty_not_an_assumption(self) -> None:
        _, result = self._run(self._three_families(), domains=False)
        codes = {u.code for u in result.uncertainty}
        self.assertIn("domain_evidence_absent", codes)
        for row in result.data["annotations"]:
            self.assertNotIn("domain_architecture", row["signals_supporting"])

    def test_artifacts_include_per_family_fasta_network_and_handoff(self) -> None:
        records = self._three_families() + [_record("sdrB", SDR_CAND_B),
                                            _record("noise", NOISE_CAND)]
        _, result = self._run(records)
        tsv = result.artifact("sequence_annotations")
        self.assertTrue(Path(tsv.path).is_file())
        header = Path(tsv.path).read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("catalytic_template_id", header)
        self.assertIn("numbering_basis", header)

        family_dir = Path(result.artifact("family_analysis_dir").path)
        names = sorted(p.name for p in family_dir.glob("*.fasta"))
        self.assertIn("unassigned.fasta", names)
        self.assertTrue(any(n.startswith("SDR__fam.sdr.v1") for n in names), names)
        self.assertTrue(any("fam.akr.v1" in n for n in names), names)
        # Each family file holds only its own sequences.
        sdr_file = next(p for p in family_dir.glob("SDR__*.fasta"))
        text = sdr_file.read_text(encoding="utf-8")
        self.assertIn(">sdrA", text)
        self.assertIn(">sdrB", text)
        self.assertNotIn(">akr", text)

        edges = Path(result.artifact("network_edges").path)
        clusters = Path(result.artifact("network_clusters").path)
        self.assertIn("percent_identity", edges.read_text(encoding="utf-8"))
        self.assertIn("not a phylogeny", clusters.read_text(encoding="utf-8"))

        handoff = Path(result.artifact("alignment_handoff").path)
        note = handoff.read_text(encoding="utf-8")
        self.assertIn("PER FAMILY", note)
        self.assertIn("mafft", note)
        self.assertIn("iqtree", note)
        self.assertIn("not a clade", note)

    def test_result_states_the_network_and_numbering_caveats(self) -> None:
        _, result = self._run(self._three_families())
        self.assertIn("not a phylogeny", result.data["network_caveat"])
        self.assertIn("candidate-sequence", result.data["numbering_basis"])
        self.assertEqual(result.provenance.parameters["numbering_basis"],
                         "candidate_sequence_1based")
        self.assertIn("separately", result.data["tree_handoff"])
        actions = {a.action for a in result.next_actions}
        self.assertIn("align_and_tree", actions)

    def test_provenance_records_hypotheses_and_thresholds(self) -> None:
        ctx, result = self._run(self._three_families())
        prov = result.provenance
        self.assertEqual(prov.tool, "annotate_family")
        self.assertEqual(sorted(prov.parameters["hypotheses"]),
                         ["fam.akr.v1", "fam.mdr.v1", "fam.sdr.v1"])
        self.assertEqual(prov.parameters["min_independent_signals"]["fam.sdr.v1"], 3)
        self.assertEqual(prov.random_seed, ctx.seed_for("annotate_family"))
        self.assertIn("network", prov.parameters)
        self.assertIn("policy", prov.parameters)

    def test_candidates_are_emitted_in_a_rebuildable_form(self) -> None:
        _, result = self._run(self._three_families())
        from eagent.schemas import Candidate
        rebuilt = [Candidate.model_validate(d) for d in result.data["candidates"]]
        by_id = {c.candidate_id: c for c in rebuilt}
        self.assertEqual(by_id["sdrA"].family.family_name, "SDR")
        self.assertEqual(by_id["sdrA"].catalytic_mapping.catalytic_template_id,
                         "cat.sdr.v1")
        self.assertTrue(by_id["sdrA"].catalytic_mapping.is_complete)

    def test_missing_inputs_fail_naming_what_was_needed(self) -> None:
        ctx = _ctx(self.tmp)
        r1 = self.iface.run(ctx, hypotheses=_hypotheses())
        self.assertIs(r1.status, Status.FAILED)
        self.assertIn("candidate_sequences.fasta", r1.message)

        r2 = self.iface.run(ctx, sequences=[_record("sdrA", SDR_CAND_A)])
        self.assertIs(r2.status, Status.FAILED)
        self.assertEqual(r2.blockers[0].code, "no_hypotheses")

        r3 = self.iface.run(ctx, hypotheses=_hypotheses(),
                            sequences_fasta=str(self.tmp / "nope.fasta"))
        self.assertIs(r3.status, Status.FAILED)
        self.assertEqual(r3.blockers[0].code, "missing_input_artifact")

    def test_sequences_can_be_read_from_the_pool_fasta(self) -> None:
        from eagent.tools.mine_sequences import FastaEntry, write_fasta
        path = self.tmp / "candidate_sequences.fasta"
        write_fasta([
            FastaEntry("sdrA", "accession=P1 db=uniprotkb@2024_01 seed=seedA "
                               "method=blastp", SDR_CAND_A),
            FastaEntry("mdr", "accession=P2 db=uniprotkb@2024_01 seed=seedB "
                              "method=blastp", MDR_CAND),
        ], path)
        ctx = _ctx(self.tmp)
        result = self.iface.run(ctx, sequences_fasta=str(path),
                                hypotheses=_hypotheses())
        self.assertTrue(result.status.usable, result.message)
        rows = {r["candidate_id"]: r for r in result.data["annotations"]}
        self.assertEqual(rows["sdrA"]["accession"], "P1")
        self.assertEqual(rows["sdrA"]["family_name"], "SDR")
        self.assertEqual(rows["mdr"]["family_name"], "MDR/ADH")


# ---------------------------------------------------------------------------
# the step-to-step hand-off
# ---------------------------------------------------------------------------


class TestCandidateHandoff(unittest.TestCase):
    """The payload this step publishes is the one the next step can read.

    ``result.data`` is serialised into the manifest, so the candidates leave
    here as plain mappings while every consumer is typed against the model.
    These tests pin the contract from the producing end: what is published is
    JSON, and ``as_candidates`` turns exactly that back into models without
    losing the family call or the catalytic mapping -- the two fields whose
    silent loss would be scored as "this candidate has no family" rather than
    "the family call did not survive the hand-off".
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.iface = AnnotateFamily()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _result(self):
        ctx = _ctx(self.tmp)
        return self.iface.run(
            ctx,
            sequences=[_record("sdrA", SDR_CAND_A), _record("mdr", MDR_CAND)],
            hypotheses=_hypotheses(),
            domain_hits={"sdrA": [DomainHit(accession="PF00106",
                                            source="Pfam 36.0")]})

    def test_candidates_are_published_as_json_under_the_agreed_key(self) -> None:
        result = self._result()
        published = result.data[CANDIDATES_KEY]
        self.assertEqual(len(published), 2)
        for item in published:
            self.assertIsInstance(item, dict)
            self.assertNotIsInstance(item, Candidate)
        # JSON mode, not python mode: an enum left as an object would not
        # survive the manifest and would come back as something else.
        self.assertIsInstance(published[0]["family"]["confidence"], str)

    def test_the_published_payload_round_trips_into_models(self) -> None:
        result = self._result()
        models = as_candidates(result.data, source="test")
        self.assertEqual([c.candidate_id for c in models], ["sdrA", "mdr"])
        self.assertTrue(all(isinstance(c, Candidate) for c in models))
        by_id = {c.candidate_id: c for c in models}
        self.assertEqual(by_id["sdrA"].family.family_name, "SDR")
        self.assertEqual(by_id["sdrA"].catalytic_mapping.catalytic_template_id,
                         "cat.sdr.v1")
        self.assertEqual(by_id["mdr"].family.family_name, "MDR/ADH")
        # The bare list is the same payload, read the same way.
        self.assertEqual(
            [c.candidate_id
             for c in as_candidates(result.data[CANDIDATES_KEY], source="test")],
            ["sdrA", "mdr"])


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
