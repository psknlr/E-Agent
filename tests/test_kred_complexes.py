"""From 19 audited entries to reference complexes -- and mostly to reasons.

What these tests pin, on real coordinates (small excerpts of the RCSB files
committed under ``tests/fixtures/kred``, each carrying the sha256 of the file it
was cut from):

* **a binding is derived by a stated rule, not by nearness.** The ketone carbon
  is the one carbon with exactly one terminal oxygen and two other heavy
  neighbours; a product (an alcohol) has none, and the rule says so instead of
  picking something. Protein roles are never bound;
* **a conformer is chosen, not inherited.** A ligand with alternate locations
  is measured per location, and the first one listed in the file is never the
  quiet default;
* **eligibility is an audit.** Every check reports pass, fail or unknown, and
  unknown fails closed. Under the template as shipped exactly one of the 19
  entries is eligible, and the reasons for the other 18 are named;
* **the calibration run is in memory and honest.** Whatever is relaxed, no
  scenario meets any policy for any constraint; the single real
  pre-reaction complex measures an approach angle outside the shipped advisory
  window; and nothing here edits a template or writes a calibration record.
"""

from __future__ import annotations

import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path

from eagent.eval.kred_complexes import (
    AUDIT_POLICIES, SCENARIOS, SDR_TEMPLATE_ID, BindingRecord, audit_entry,
    audit_report, build_excerpt, derive_binding, find_carbonyl, load_bindings,
    render_report, representatives, run_scenario, select_conformer, write_excerpt,
)
from eagent.eval.kred_coordinates import (
    default_cache_dir, lineage_counts, load_coordinates_manifest,
)
from eagent.eval.kred_reference import default_reference_dir, load_reference_set
from eagent.harness.templates import TemplateLibrary
from eagent.science.calibration import minimum_actives
from eagent.science.structure_io import Atom, Residue, read_mmcif

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kred"
AUDIT_JSON = (Path(__file__).resolve().parents[1] / "docs" / "results"
              / "kred_reference_audit.json")


def excerpt(pid: str):
    return read_mmcif(FIXTURES / f"{pid}_excerpt.cif", structure_id=pid)


def atom(name: str, element: str, x: float, y: float, z: float, *, altloc: str = "") -> Atom:
    return Atom(serial=1, name=name, element=element, resname="LIG", chain="A",
                resseq=1, icode="", altloc=altloc, x=x, y=y, z=z, occupancy=1.0,
                bfactor_or_plddt=20.0, is_hetatm=True)


def residue(*atoms: Atom) -> Residue:
    return Residue(chain="A", resname="LIG", resseq=1, icode="", atoms=list(atoms),
                   is_hetatm=True)


def ketone_atoms(altloc: str = "") -> list[Atom]:
    """Acetone, ideal geometry: C2=O at 1.22 A, two methyls at 1.43 A."""
    return [atom("C1", "C", -0.7, 1.25, 0, altloc=altloc),
            atom("C2", "C", 0, 0, 0, altloc=altloc),
            atom("O", "O", 1.22, 0, 0, altloc=altloc),
            atom("C3", "C", -0.7, -1.25, 0, altloc=altloc)]


class _Shared(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rs = load_reference_set()
        cls.pins = load_coordinates_manifest(default_reference_dir())
        cls.bindings = load_bindings(default_reference_dir())
        cls.templates = TemplateLibrary.load().catalytic_templates
        cls.template = cls.templates[SDR_TEMPLATE_ID]
        cls.structures = {pid: excerpt(pid) for pid in ("1IPF", "2AE2", "6ZZO", "6ZZP", "1ZK4")}


# ==========================================================================
class TheFixturesAreCutFromThePinnedFiles(_Shared):
    def test_each_excerpt_names_the_hash_of_the_file_it_came_from(self) -> None:
        for pid in self.structures:
            head = (FIXTURES / f"{pid}_excerpt.cif").read_text(encoding="utf-8").splitlines()
            line = next(l for l in head if l.startswith("# source sha256 "))
            self.assertEqual(line.split()[-1], self.pins["files"][pid]["sha256"], pid)

    @unittest.skipUnless(len(list(default_cache_dir().glob("*.cif"))) == 19,
                         "the 19 coordinate files have not been fetched")
    def test_every_excerpt_atom_is_the_full_files_atom_unchanged(self) -> None:
        for pid, struct in self.structures.items():
            full = read_mmcif(default_cache_dir() / f"{pid}.cif", structure_id=pid)
            by_serial = {a.serial: a for a in full.atoms()}
            for a in struct.atoms():
                b = by_serial[a.serial]
                self.assertEqual((a.name, a.resname, a.chain, a.resseq, a.altloc, a.x, a.y, a.z),
                                 (b.name, b.resname, b.chain, b.resseq, b.altloc, b.x, b.y, b.z),
                                 f"{pid} atom {a.serial}")

    @unittest.skipUnless((default_cache_dir() / "6ZZO.cif").is_file(),
                         "the coordinate files have not been fetched")
    def test_the_excerpt_builder_reproduces_the_committed_excerpt(self) -> None:
        full = read_mmcif(default_cache_dir() / "6ZZO.cif", structure_id="6ZZO")
        atoms = build_excerpt(full, self.bindings["6ZZO"], radius=6.0)
        with tempfile.TemporaryDirectory() as tmp:
            head = (FIXTURES / "6ZZO_excerpt.cif").read_text(encoding="utf-8").splitlines()
            prov = [l[2:] for l in head if l.startswith("# ")]
            path = write_excerpt(atoms, Path(tmp) / "x.cif", pdb_id="6ZZO", provenance=prov)
            self.assertEqual(path.read_text(encoding="utf-8"),
                             (FIXTURES / "6ZZO_excerpt.cif").read_text(encoding="utf-8"))


# ==========================================================================
class TheKetoneIsFoundByRuleNotByNearness(_Shared):
    def residue(self, pid: str, chain: str, resseq: int) -> Residue:
        return next(r for r in self.structures[pid].residues()
                    if r.chain == chain and r.resseq == resseq and r.is_hetatm)

    def test_tropinone_c3_is_the_electrophile(self) -> None:
        self.assertEqual(find_carbonyl(self.residue("1IPF", "A", 262)), ("C3", "O3"))

    def test_acetoacetate_picks_the_ketone_not_the_carboxylate(self) -> None:
        res = self.residue("6ZZO", "B", 302)
        self.assertEqual(find_carbonyl(res, "A"), ("C5", "O8"))
        self.assertEqual(find_carbonyl(res, "B"), ("C5", "O8"))

    def test_a_residue_with_alternate_locations_and_no_conformer_is_refused(self) -> None:
        carbon, why = find_carbonyl(self.residue("6ZZO", "B", 302), None)
        self.assertIsNone(carbon)
        self.assertIn("alternate locations", why)

    def test_a_product_alcohol_has_no_ketone_and_the_rule_says_so(self) -> None:
        carbon, why = find_carbonyl(self.residue("2AE2", "A", 262))
        self.assertIsNone(carbon)
        self.assertIn("a product, not a substrate", why)

    def test_acetophenone_is_bound(self) -> None:
        self.assertEqual(find_carbonyl(self.residue("1ZK4", "A", 1260)), ("C7", "O1"))

    def test_the_rule_on_ideal_geometries(self) -> None:
        self.assertEqual(find_carbonyl(residue(*ketone_atoms())), ("C2", "O"))
        carboxylate = [atom("C1", "C", 0, 0, 0), atom("C2", "C", -0.7, 1.4, 0),
                       atom("O1", "O", 0.6, -1.1, 0), atom("O2", "O", 0.6, 1.1, 0)]
        # two terminal oxygens on the carbon: a carboxylate, not a ketone
        carboxylate[2] = atom("O1", "O", 1.1, -0.6, 0)
        carboxylate[3] = atom("O2", "O", -1.1, -0.6, 0)
        self.assertIsNone(find_carbonyl(residue(*carboxylate))[0])
        alcohol = [atom("C1", "C", 0, 0, 0), atom("C2", "C", -0.7, 1.3, 0),
                   atom("C3", "C", -0.7, -1.3, 0), atom("O", "O", 1.43, 0, 0)]
        self.assertIsNone(find_carbonyl(residue(*alcohol))[0])
        aldehyde = [atom("C1", "C", 0, 0, 0), atom("C2", "C", -1.2, 0.8, 0),
                    atom("O", "O", 1.2, 0, 0)]
        self.assertIsNone(find_carbonyl(residue(*aldehyde))[0])

    def test_a_diketone_is_refused_rather_than_resolved(self) -> None:
        diketone = [atom("C1", "C", -1.7, 1.0, 0), atom("C2", "C", -0.75, 0, 0),
                    atom("O2", "O", -1.2, -1.1, 0), atom("C3", "C", 0.75, 0, 0),
                    atom("O3", "O", 1.2, 1.1, 0), atom("C4", "C", 1.7, -1.0, 0)]
        carbon, why = find_carbonyl(residue(*diketone))
        self.assertIsNone(carbon)
        self.assertIn("2 carbonyl carbons", why)
        self.assertIn("without choosing", why)


# ==========================================================================
class AConformerIsChosenNotInherited(_Shared):
    def test_selecting_a_conformer_keeps_only_that_location_across_the_whole_model(self) -> None:
        full = self.structures["6ZZO"]
        a = select_conformer(full, "A")
        self.assertTrue(all(x.altloc in ("", "A") for x in a.atoms()))
        self.assertLess(a.n_atoms(), full.n_atoms())
        self.assertTrue(any("alternate location 'A' selected" in n for n in a.parse_notes))
        # the original is untouched
        self.assertTrue(any(x.altloc == "B" for x in full.atoms()))

    def test_no_conformer_leaves_the_structure_as_it_is(self) -> None:
        self.assertIs(select_conformer(self.structures["1IPF"], None), self.structures["1IPF"])

    def test_the_two_conformers_of_6zzo_give_different_geometry(self) -> None:
        def c4n_to_c5(conf: str) -> float:
            s = select_conformer(self.structures["6ZZO"], conf)
            nad = next(r for r in s.residues() if r.chain == "B" and r.resname == "NAD")
            aae = next(r for r in s.residues() if r.chain == "B" and r.resname == "AAE")
            return math.dist(nad.atom("C4N").coords, aae.atom("C5").coords)
        self.assertAlmostEqual(c4n_to_c5("A"), 3.4487, places=3)
        self.assertNotAlmostEqual(c4n_to_c5("A"), c4n_to_c5("B"), places=1)

    def test_the_first_location_listed_is_not_a_quiet_default(self) -> None:
        """Without a selection, ``Residue.atom`` returns whichever is listed first."""
        aae = next(r for r in self.structures["6ZZO"].residues()
                   if r.chain == "B" and r.resname == "AAE")
        first = aae.atom("C5")
        self.assertEqual(first.altloc, "A")        # the trap: it looks like a choice
        b = select_conformer(self.structures["6ZZO"], "B")
        aae_b = next(r for r in b.residues() if r.chain == "B" and r.resname == "AAE")
        self.assertEqual(aae_b.atom("C5").altloc, "B")


# ==========================================================================
class BindingsAreDerivedAndWrittenDown(_Shared):
    def test_the_committed_bindings_are_what_the_rule_gives_on_the_real_atoms(self) -> None:
        for pid, struct in self.structures.items():
            entry = self.rs.structure(pid)
            derived = derive_binding(entry, struct, self.rs.validation_rows(pid))
            self.assertEqual(derived.to_dict(), self.bindings[pid].to_dict(), pid)

    def test_exactly_these_entries_are_completely_bound(self) -> None:
        complete = sorted(pid for pid, b in self.bindings.items() if b.complete)
        self.assertEqual(complete, ["1IPF", "1ZK1", "1ZK4", "6ZZO", "6ZZP", "6ZZQ", "6ZZS"])

    def test_products_are_recognised_as_products_by_the_rule(self) -> None:
        for pid in ("2AE2", "1ZJY", "1ZJZ", "1ZK0", "7UUT"):
            b = self.bindings[pid]
            self.assertFalse(b.complete, pid)
            self.assertTrue(any("product, not a substrate" in f for f in b.failures), pid)

    def test_a_racemic_site_is_not_bound(self) -> None:
        b = self.bindings["6XEW"]
        self.assertFalse(b.complete)
        self.assertIn("racemic or mixed site", b.failures[0])

    def test_the_site_the_workbook_scored_is_the_one_bound(self) -> None:
        # 6ZZQ has two acetoacetate sites; the summary RSCC singles out A302
        self.assertEqual(self.bindings["6ZZQ"].substrate["resseq"], 302)
        self.assertEqual(self.bindings["6ZZO"].chain, "B")
        self.assertEqual(self.bindings["6ZZO"].conformers, ("A",))
        self.assertEqual(self.bindings["6ZZP"].conformers, ("A", "B"))

    def test_every_binding_says_it_is_rule_derived_and_unreviewed(self) -> None:
        for b in self.bindings.values():
            self.assertEqual(b.review_status, "rule_derived_unreviewed")
            self.assertIn("find_carbonyl", b.derivation)

    def test_protein_roles_are_never_bound(self) -> None:
        for b in self.bindings.values():
            self.assertEqual(b.pose_binding().protein_atoms, {})

    def test_the_hydride_donor_is_the_ccd_name_for_the_nicotinamide_c4(self) -> None:
        for b in self.bindings.values():
            if b.cofactor:
                self.assertEqual(b.cofactor_atoms, {"hydride_donor_C4": "C4N"})

    def test_a_round_trip_through_json_keeps_everything(self) -> None:
        b = self.bindings["6ZZP"]
        self.assertEqual(BindingRecord.from_dict(json.loads(json.dumps(b.to_dict()))), b)


# ==========================================================================
class EligibilityIsAnAuditNotAGuess(_Shared):
    def audit(self, pid: str, policy: str = "strict", **replace):
        entry = self.rs.structure(pid)
        if replace:
            entry = dataclasses.replace(entry, **replace)
        return audit_entry(self.rs, entry, self.template, manifest=self.pins,
                           binding=self.bindings.get(pid), policy=AUDIT_POLICIES[policy])

    def names(self, audit) -> set[str]:
        return {c.name for c in audit.checks if c.status != "pass"}

    def test_under_the_template_as_shipped_one_entry_of_nineteen_is_eligible(self) -> None:
        eligible = [s.pdb_id for s in self.rs.structures if self.audit(s.pdb_id).eligible]
        self.assertEqual(eligible, ["1IPF"])

    def test_every_ineligible_entry_says_why(self) -> None:
        for s in self.rs.structures:
            a = self.audit(s.pdb_id)
            if not a.eligible:
                self.assertTrue(a.reasons, s.pdb_id)
                for reason in a.reasons:
                    self.assertRegex(reason, r"^[a-z_]+: ")

    def test_6zzo_fails_on_the_cofactor_alone(self) -> None:
        """The best-supported HBDH complex has NAD+, and the template wants NADPH."""
        a = self.audit("6ZZO")
        self.assertEqual(self.names(a), {"cofactor"})
        self.assertIn("NAD is an oxidized nicotinamide cofactor", a.reasons[0])

    def test_the_product_complex_fails_five_ways(self) -> None:
        self.assertEqual(self.names(self.audit("2AE2")),
                         {"workbook_grade", "cofactor", "substrate_state",
                          "activity_evidence", "binding"})

    def test_the_other_families_are_named_as_other_families(self) -> None:
        self.assertIn("family", self.names(self.audit("1Y1P")))
        self.assertIn("extended_sdr_epimerase_dehydratase",
                      " ".join(self.audit("1Y1P").reasons))
        self.assertIn("zinc_adh_mdr", " ".join(self.audit("7UUT").reasons))

    def test_a_poorly_resolved_ligand_fails_on_density_whatever_the_resolution(self) -> None:
        a = self.audit("1ZK4")
        self.assertIn("density", self.names(a))
        self.assertIn("0.154", " ".join(a.reasons))

    def test_missing_evidence_fails_closed(self) -> None:
        a = self.audit("1IPF", ligand_rscc=(), cofactor_rscc=None)
        self.assertFalse(a.eligible)
        self.assertEqual([c.status for c in a.checks if c.name == "density"], ["unknown"])

    def test_a_weak_density_value_disqualifies(self) -> None:
        self.assertIn("density", self.names(self.audit("1IPF", ligand_rscc=(0.5,))))

    def test_without_a_binding_nothing_is_eligible(self) -> None:
        entry = self.rs.structure("1IPF")
        a = audit_entry(self.rs, entry, self.template, manifest=self.pins, binding=None,
                        policy=AUDIT_POLICIES["strict"])
        self.assertFalse(a.eligible)
        self.assertEqual({c.status for c in a.checks if c.name in ("binding", "cofactor")},
                         {"unknown"})

    def test_activity_must_be_for_this_variant_and_this_substrate(self) -> None:
        self.assertIn("activity_evidence", self.names(self.audit("1IPF", variant="G37D")))
        self.assertIn("activity_evidence",
                      self.names(self.audit("1IPF", bound_reaction_ligand="acetophenone")))

    def test_the_conformer_must_be_single_under_strict_and_may_be_two_when_relaxed(self) -> None:
        self.assertIn("conformer", self.names(self.audit("6ZZP", "strict")))
        self.assertNotIn("conformer", self.names(self.audit("6ZZP", "every_graded_pose")))

    def test_relaxing_the_oxidation_state_alone_admits_nothing_new(self) -> None:
        eligible = [s.pdb_id for s in self.rs.structures
                    if self.audit(s.pdb_id, "oxidised_same_cofactor").eligible]
        self.assertEqual(eligible, ["1IPF"])

    def test_dropping_the_cofactor_requirement_admits_one_more_lineage(self) -> None:
        eligible = [s.pdb_id for s in self.rs.structures
                    if self.audit(s.pdb_id, "any_nicotinamide_cofactor").eligible]
        self.assertEqual(eligible, ["1IPF", "6ZZO"])

    def test_every_graded_pose_adds_a_second_hbdh_entry_and_no_third_lineage(self) -> None:
        eligible = [s.pdb_id for s in self.rs.structures
                    if self.audit(s.pdb_id, "every_graded_pose").eligible]
        self.assertEqual(eligible, ["1IPF", "6ZZO", "6ZZP"])
        self.assertEqual({self.rs.structure(p).lineage for p in eligible}, {"TR-II", "HBDH"})

    def test_an_accepted_stand_in_is_labelled_as_one(self) -> None:
        a = self.audit("6ZZO", "any_nicotinamide_cofactor")
        text = " ".join(c.detail for c in a.checks if c.name == "cofactor")
        self.assertIn("ACCEPTED ONLY UNDER THIS POLICY", text)

    def test_the_activity_evidence_names_the_tier_of_the_label_it_rests_on(self) -> None:
        text = " ".join(c.detail for c in self.audit("1IPF").checks
                        if c.name == "activity_evidence")
        self.assertIn("DSTRII_TROP_NADPH_2003 [secondary", text)

    def test_one_representative_per_lineage_so_n_counts_lineages_not_files(self) -> None:
        entries = [self.rs.structure(p) for p in ("1IPF", "2AE2", "6ZZO", "6ZZP")]
        self.assertEqual([e.pdb_id for e in representatives(entries)], ["1IPF", "6ZZO"])
        lacto = [self.rs.structure(p) for p in ("1ZK4", "4RF2", "4RF3")]
        self.assertEqual(len(representatives(lacto)), 1)


# ==========================================================================
class TheCalibrationRunIsInMemoryAndHonest(_Shared):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.runs = {sc.name: run_scenario(cls.rs, sc, cls.template, manifest=cls.pins,
                                          bindings=cls.bindings, structures=cls.structures)
                    for sc in SCENARIOS}

    def test_the_scenarios_are_the_four_that_relax_one_thing_at_a_time(self) -> None:
        self.assertEqual([s.name for s in SCENARIOS],
                         ["as_shipped", "oxidised_same_cofactor",
                          "any_nicotinamide_cofactor", "every_graded_pose"])

    def test_as_shipped_there_is_one_active_and_no_inactive(self) -> None:
        run = self.runs["as_shipped"]
        self.assertEqual(run["eligible_entries"], ["1IPF"])
        self.assertEqual((run["n_independent_actives"], run["n_known_inactives"]), (1, 0))

    def test_no_scenario_meets_any_policy_for_any_constraint(self) -> None:
        for name, run in self.runs.items():
            for cal in run["calibrations"]:
                for constraint, rec in cal["records"].items():
                    self.assertFalse(rec["meets_policy"], f"{name}/{constraint}")

    def test_no_scenario_has_a_known_inactive_because_the_set_has_none(self) -> None:
        for run in self.runs.values():
            self.assertEqual(run["n_known_inactives"], 0)
            self.assertIn("ND is not a zero", run["known_inactives_note"])

    def test_the_one_real_prereaction_complex_measures_outside_the_advisory_angle_window(self) -> None:
        m = self.runs["as_shipped"]["measurements"][0]
        self.assertEqual(m["reference_id"], "1IPF:A:-")
        self.assertAlmostEqual(m["measurements"]["hydride_donor_to_carbonyl_carbon"], 4.059, places=2)
        self.assertAlmostEqual(m["measurements"]["hydride_approach_angle_burgi_dunitz"], 72.61, places=2)
        self.assertTrue(m["inside_shipped_window"]["hydride_donor_to_carbonyl_carbon"])
        self.assertFalse(m["inside_shipped_window"]["hydride_approach_angle_burgi_dunitz"])

    def test_protein_constraints_are_unmeasured_not_failed(self) -> None:
        m = self.runs["as_shipped"]["measurements"][0]
        for name in ("carbonyl_oxygen_to_catalytic_tyr_OH",
                     "carbonyl_oxygen_to_catalytic_ser_OG",
                     "catalytic_lys_NZ_to_catalytic_tyr_OH"):
            self.assertIsNone(m["measurements"][name])
            self.assertIsNone(m["inside_shipped_window"][name])

    def test_two_lineages_propose_a_window_and_it_still_does_not_meet_the_policy(self) -> None:
        run = self.runs["any_nicotinamide_cofactor"]
        self.assertEqual(run["eligible_lineages"], ["HBDH", "TR-II"])
        rec = run["calibrations"][0]["records"]["hydride_donor_to_carbonyl_carbon"]
        self.assertEqual(rec["n_active"], 2)
        self.assertAlmostEqual(rec["window"][0], 3.4487, places=3)
        self.assertAlmostEqual(rec["window"][1], 4.0590, places=3)
        self.assertFalse(rec["meets_policy"])
        self.assertEqual(rec["n_actives_needed"], 14)
        self.assertTrue(any("2 active reference" in r for r in rec["reasons"]))
        self.assertTrue(any("0 known-inactive" in r for r in rec["reasons"]))

    def test_all_three_measured_hbdh_and_tr2_poses_have_angles_below_the_advisory_band(self) -> None:
        for m in self.runs["every_graded_pose"]["measurements"]:
            angle = m["measurements"]["hydride_approach_angle_burgi_dunitz"]
            self.assertLess(angle, 90.0, m["reference_id"])
            self.assertFalse(m["inside_shipped_window"]["hydride_approach_angle_burgi_dunitz"])
            self.assertTrue(m["inside_shipped_window"]["hydride_donor_to_carbonyl_carbon"])

    def test_a_second_conformer_is_measured_and_shown_but_not_counted(self) -> None:
        run = self.runs["every_graded_pose"]
        six_p = [m for m in run["measurements"] if m["reference_id"].startswith("6ZZP")]
        self.assertEqual({m["conformer"] for m in six_p}, {"A", "B"})
        self.assertEqual([m["in_calibration"] for m in six_p], [False, False])
        self.assertFalse(any(m["representative"] for m in six_p))
        self.assertEqual(run["n_independent_actives"], 2)

    def test_the_conformers_of_one_ligand_do_not_count_as_two_samples(self) -> None:
        refs = [m["reference_id"] for m in self.runs["every_graded_pose"]["measurements"]
                if m["in_calibration"]]
        self.assertEqual(refs, ["1IPF:A:-", "6ZZO:B:A"])

    def test_policy_numbers_are_what_the_calibration_module_says(self) -> None:
        recs = self.runs["as_shipped"]["calibrations"]
        self.assertEqual([r["records"]["hydride_donor_to_carbonyl_carbon"]["n_actives_needed"]
                          for r in recs], [minimum_actives(0.8, 0.8), minimum_actives(0.9, 0.9)])
        self.assertEqual([minimum_actives(0.8, 0.8), minimum_actives(0.9, 0.9)], [14, 38])

    def test_the_policy_says_where_its_numbers_came_from(self) -> None:
        for cal in self.runs["as_shipped"]["calibrations"]:
            self.assertIn("not by a campaign", cal["policy"]["source"])

    def test_nothing_in_the_run_produces_a_citation(self) -> None:
        text = json.dumps(self.runs)
        self.assertNotIn("calibration:", text)
        self.assertNotIn("calibrated_on_entry", text)


class NothingShippedWasEdited(_Shared):
    def test_the_sdr_template_is_still_uncalibrated_with_no_reference_structures(self) -> None:
        for c in self.template.geometry_constraints:
            self.assertEqual(list(c.calibrated_on), [], c.name)
        self.assertEqual(list(self.template.reference_structures), [])

    def test_no_shipped_catalytic_template_cites_a_calibration(self) -> None:
        for tid, tpl in self.templates.items():
            for c in tpl.geometry_constraints:
                self.assertEqual(list(c.calibrated_on), [], f"{tid}/{c.name}")

    def test_the_other_two_templates_have_no_entry_that_matches_family_and_cofactor(self) -> None:
        report = audit_report(self.rs, self.templates, manifest=self.pins,
                              bindings=self.bindings, structures=self.structures)
        got = {tid: v["entries_matching_family_and_cofactor"]
               for tid, v in report["template_applicability"].items()}
        self.assertEqual(got, {"cat.akr.nadph_carbonyl_reduction.v1": [],
                               "cat.mdr_adh.zn_nadh_carbonyl_reduction.v1": [],
                               "cat.sdr.nadph_carbonyl_reduction.v1": ["1IPF"]})


# ==========================================================================
class TheCommittedAuditIsCurrent(_Shared):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.committed = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        cls.fresh = audit_report(cls.rs, cls.templates, manifest=cls.pins,
                                 bindings=cls.bindings, structures=cls.structures,
                                 identity_counts=lineage_counts(cls.rs, cls.pins))

    def test_it_was_written_for_the_data_that_is_committed(self) -> None:
        manifest = json.loads((default_reference_dir() / "MANIFEST.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(self.committed["reference_set"]["data_digest"],
                         manifest["data_digest"],
                         "the audit result is stale: re-run "
                         "`eagent reference audit --out docs/results`")

    def test_a_fresh_run_reproduces_its_verdicts_and_measurements(self) -> None:
        for old, new in zip(self.committed["scenarios"], self.fresh["scenarios"]):
            self.assertEqual(old["scenario"], new["scenario"])
            self.assertEqual(old["eligible_entries"], new["eligible_entries"])
            self.assertEqual(old["representatives"], new["representatives"])
            self.assertEqual([m["reference_id"] for m in old["measurements"]],
                             [m["reference_id"] for m in new["measurements"]])
            for a, b in zip(old["measurements"], new["measurements"]):
                for name in a["measurements"]:
                    if a["measurements"][name] is None:
                        self.assertIsNone(b["measurements"][name])
                    else:
                        self.assertAlmostEqual(a["measurements"][name],
                                               b["measurements"][name], places=9)
            self.assertEqual(old["calibrations"], new["calibrations"])

    def test_the_text_report_is_the_rendering_of_the_json(self) -> None:
        text = (AUDIT_JSON.with_suffix(".txt")).read_text(encoding="utf-8")
        self.assertEqual(text, render_report(self.committed))

    def test_the_committed_lineage_counts_are_those_of_the_pinned_sequences(self) -> None:
        self.assertEqual(self.committed["lineage_counts_by_identity_threshold"],
                         lineage_counts(self.rs, self.pins))


if __name__ == "__main__":
    unittest.main()
