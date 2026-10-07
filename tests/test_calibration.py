"""Tests for window calibration: a window may reject only on a record that opens.

``calibrated_on`` was a free-text list, so ``calibrated_on: ["trust me"]`` gave
a geometric window the power to reject an enzyme. These tests pin the
replacement: a ``calibration:<digest>`` citation resolves to a stored record,
the record's digest is recomputed, its verdict is re-run rather than read, and
anything that cannot be shown fails closed to *uncalibrated*.

EVERY REFERENCE COMPLEX IN THIS FILE IS SYNTHETIC. The coordinates are placed
by hand so the expected distances can be read off the constructors. That is the
right input for testing the *machinery* (the statistics, the exclusions, the
tamper checks) and it is the wrong input for a real window: nothing here is a
claim about where a ketoreductase's hydride donor sits. No real reference set
exists in the repository, which is why the shipped windows remain uncalibrated.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from eagent.schemas import (
    CatalyticTemplate,
    CofactorState,
    GeometryConstraint,
    TemplateProvenance,
    TemplateSourceType,
)
from eagent.science.calibration import (
    CALIBRATION_ALGORITHM,
    CALIBRATION_PREFIX,
    CalibrationContext,
    CalibrationError,
    CalibrationPolicy,
    CalibrationStore,
    ReferenceObservation,
    calibrate,
    coverage_at_confidence,
    minimum_actives,
    wilks_two_sided_confidence,
)
from eagent.science.structure_io import Chain, Structure
from eagent.tools.calibrate_windows import (
    ReferenceComplex,
    calibrate_template,
    calibrated_fields,
    measure_references,
)
from eagent.tools.evaluate_catalysis import (
    PoseOutcome,
    WindowAuthority,
    constraint_authority,
    template_authority,
)

from test_evaluate_catalysis import (
    CARBONYL_C,
    CARBONYL_O,
    LYS_NZ,
    SUBST_ARYL,
    SUBST_METHYL,
    TYR_OH,
    _Harness,
    _atom,
    _residue,
    make_binding,
    make_candidate,
    make_pose,
)

#: Small enough to build a test set by hand, and still a real Wilks bound:
#: 14 actives support 80% coverage at 80% confidence.
POLICY = CalibrationPolicy(
    coverage=0.8, confidence=0.8, min_inactives=5,
    max_inactive_inside_upper=0.5,
    source="test-suite: a policy chosen so that a hand-built set can pass it; "
           "not a recommendation for a real campaign")


def structure_with_donor_at(distance: float, structure_id: str = "ref") -> Structure:
    """The test active site, with the hydride donor ``distance`` A from C1.

    The donor stays at the 107 degree Burgi-Dunitz direction so the angle is
    constant and only the distance varies.
    """
    angle = math.radians(107.0)
    donor = (distance * math.cos(angle), 0.0, distance * math.sin(angle))
    tyr = _residue("TYR", "A", 155, [_atom(1, "OH", "O", "TYR", "A", 155, TYR_OH)])
    lys = _residue("LYS", "A", 159, [_atom(2, "NZ", "N", "LYS", "A", 159, LYS_NZ)])
    cof = _residue("NDP", "A", 301,
                   [_atom(3, "C4N", "C", "NDP", "A", 301, donor, hetatm=True)],
                   hetatm=True)
    lig = _residue("LIG", "A", 401, [
        _atom(4, "C1", "C", "LIG", "A", 401, CARBONYL_C, hetatm=True),
        _atom(5, "O1", "O", "LIG", "A", 401, CARBONYL_O, hetatm=True),
        _atom(6, "C2", "C", "LIG", "A", 401, SUBST_METHYL, hetatm=True),
        _atom(7, "C3", "C", "LIG", "A", 401, SUBST_ARYL, hetatm=True),
    ], hetatm=True)
    return Structure(structure_id=structure_id,
                     chains=[Chain(chain_id="A", residues=[tyr, lys, cof, lig])],
                     source_format="mmcif", models_present=[1], model_selected=1)


def uncalibrated_template(**overrides) -> CatalyticTemplate:
    """One distance window, deliberately wide, no calibration cited."""
    constraint = dict(
        name="hydride_transfer_distance", kind="distance",
        atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
        min_value=2.0, max_value=9.0, severity="gating", source="synthetic")
    constraint.update(overrides)
    return CatalyticTemplate(
        template_id="ct_sdr_ketoreductase_v1", family_name="SDR",
        mechanism_summary="synthetic, for the calibration tests",
        catalytic_residues=[
            {"label": "catalytic_Tyr", "residue_types": ["TYR"],
             "role": "proton donor", "functional_atoms": ["OH"],
             "evidence": "synthetic"},
            {"label": "catalytic_Lys", "residue_types": ["LYS"],
             "role": "anchors the cofactor", "functional_atoms": ["NZ"],
             "evidence": "synthetic"},
        ],
        required_cofactor="NADPH", required_cofactor_state=CofactorState.REDUCED,
        cofactor_ligand_codes=["NDP"],
        geometry_constraints=[GeometryConstraint(**constraint)],
        reference_structures=["synthetic"],
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.EXPERIMENTAL_STRUCTURE,
            identifiers=["synthetic"], curated_by="test"))


def references(actives, inactives, *, evidence="synthetic: placed by hand"):
    out = []
    for i, d in enumerate(actives):
        out.append(ReferenceComplex(
            reference_id=f"act{i:02d}", label="active", evidence=evidence,
            binding=make_binding(pose_id=f"act{i:02d}"),
            structure=structure_with_donor_at(d, f"act{i:02d}")))
    for i, d in enumerate(inactives):
        out.append(ReferenceComplex(
            reference_id=f"ina{i:02d}", label="inactive", evidence=evidence,
            binding=make_binding(pose_id=f"ina{i:02d}"),
            structure=structure_with_donor_at(d, f"ina{i:02d}")))
    return out


ACTIVES = [3.2 + 0.05 * i for i in range(14)]         # 3.20 .. 3.85
INACTIVES = [5.0, 5.4, 5.8, 6.2, 6.6, 7.0]


# ==========================================================================
# the statistics
# ==========================================================================

class TestWilks(unittest.TestCase):

    def test_two_samples_have_the_closed_form(self) -> None:
        # n=2: 1 - 2p + p^2 = (1 - p)^2
        for p in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(wilks_two_sided_confidence(2, p), (1 - p) ** 2)

    def test_known_sample_sizes(self) -> None:
        self.assertEqual(minimum_actives(0.9, 0.9), 38)
        self.assertEqual(minimum_actives(0.9, 0.95), 46)
        self.assertEqual(minimum_actives(0.8, 0.8), 14)

    def test_seven_actives_do_not_support_ninety_percent_coverage(self) -> None:
        self.assertLess(wilks_two_sided_confidence(7, 0.9), 0.2)

    def test_fewer_than_two_samples_support_nothing(self) -> None:
        self.assertEqual(wilks_two_sided_confidence(1, 0.5), 0.0)
        self.assertEqual(wilks_two_sided_confidence(0, 0.5), 0.0)
        self.assertEqual(coverage_at_confidence(1, 0.9), 0.0)

    def test_confidence_falls_as_coverage_rises(self) -> None:
        values = [wilks_two_sided_confidence(20, p) for p in (0.5, 0.7, 0.9, 0.99)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_coverage_at_confidence_inverts_confidence(self) -> None:
        for n in (5, 14, 38):
            p = coverage_at_confidence(n, 0.9)
            self.assertAlmostEqual(wilks_two_sided_confidence(n, p), 0.9, places=6)

    def test_out_of_range_arguments_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            wilks_two_sided_confidence(10, 1.0)
        with self.assertRaises(ValueError):
            coverage_at_confidence(10, 0.0)


# ==========================================================================
# inputs
# ==========================================================================

class TestInputsAreEvidenced(unittest.TestCase):

    def test_a_policy_must_say_where_its_numbers_came_from(self) -> None:
        with self.assertRaises(CalibrationError):
            CalibrationPolicy(coverage=0.9, confidence=0.9, min_inactives=5,
                              max_inactive_inside_upper=0.4, source="  ")

    def test_a_policy_rejects_out_of_range_numbers(self) -> None:
        with self.assertRaises(CalibrationError):
            CalibrationPolicy(coverage=1.2, confidence=0.9, min_inactives=5,
                              max_inactive_inside_upper=0.4, source="x")
        with self.assertRaises(CalibrationError):
            CalibrationPolicy(coverage=0.9, confidence=0.9, min_inactives=0,
                              max_inactive_inside_upper=0.4, source="x")

    def test_a_reference_with_no_evidence_is_refused_for_either_label(self) -> None:
        for label in ("active", "inactive"):
            with self.assertRaises(CalibrationError):
                ReferenceObservation("r", label, 3.4, evidence="")

    def test_an_unknown_label_is_refused(self) -> None:
        with self.assertRaises(CalibrationError):
            ReferenceObservation("r", "maybe", 3.4, evidence="x")

    def test_a_non_finite_value_is_refused(self) -> None:
        with self.assertRaises(CalibrationError):
            ReferenceObservation("r", "active", float("nan"), evidence="x")


# ==========================================================================
# the calibration
# ==========================================================================

def obs(rid, label, value, **kw):
    return ReferenceObservation(rid, label, value, evidence="synthetic", **kw)


def good_observations():
    return ([obs(f"a{i:02d}", "active", d) for i, d in enumerate(ACTIVES)]
            + [obs(f"i{i:02d}", "inactive", d) for i, d in enumerate(INACTIVES)])


class TestCalibrate(unittest.TestCase):

    def test_the_window_is_the_range_of_the_actives(self) -> None:
        record = calibrate("hydride_transfer_distance", good_observations(), POLICY)
        self.assertAlmostEqual(record.window[0], 3.2)
        self.assertAlmostEqual(record.window[1], 3.85)

    def test_enough_actives_and_a_discriminating_window_meets_the_policy(self) -> None:
        record = calibrate("hydride_transfer_distance", good_observations(), POLICY)
        self.assertTrue(record.verdict.meets_policy, record.verdict.reasons)
        self.assertEqual(record.verdict.inactive_inside, 0)
        self.assertRegex(record.calibrated_on_entry,
                         r"^calibration:[0-9a-f]{16}$")

    def test_too_few_actives_state_what_they_support_and_yield_no_citation(self) -> None:
        few = [obs(f"a{i}", "active", d) for i, d in enumerate(ACTIVES[:7])]
        few += [obs(f"i{i}", "inactive", d) for i, d in enumerate(INACTIVES)]
        record = calibrate("hydride_transfer_distance", few, POLICY)
        self.assertFalse(record.verdict.meets_policy)
        self.assertIsNone(record.calibrated_on_entry)
        self.assertEqual(record.verdict.n_actives_needed, 14)
        self.assertLess(record.verdict.achieved_confidence, 0.8)
        self.assertIn("needs 14", " ".join(record.verdict.reasons))
        # the window is still proposed: it is information, only not authority
        self.assertIsNotNone(record.window)

    def test_one_active_proposes_no_window(self) -> None:
        record = calibrate("c", [obs("a0", "active", 3.4)] +
                           [obs(f"i{i}", "inactive", d)
                            for i, d in enumerate(INACTIVES)], POLICY)
        self.assertIsNone(record.window)
        self.assertFalse(record.verdict.meets_policy)

    def test_a_window_that_does_not_exclude_the_inactives_fails(self) -> None:
        inside = [obs(f"i{i}", "inactive", 3.3 + 0.1 * i) for i in range(6)]
        record = calibrate("c", [obs(f"a{i:02d}", "active", d)
                                 for i, d in enumerate(ACTIVES)] + inside, POLICY)
        self.assertFalse(record.verdict.meets_policy)
        self.assertEqual(record.verdict.inactive_inside, 6)
        self.assertIn("accept what it exists to reject",
                      " ".join(record.verdict.reasons))

    def test_too_few_inactives_leave_discrimination_unshown(self) -> None:
        record = calibrate("c", [obs(f"a{i:02d}", "active", d)
                                 for i, d in enumerate(ACTIVES)]
                           + [obs("i0", "inactive", 7.0)], POLICY)
        self.assertFalse(record.verdict.meets_policy)
        self.assertIn("nothing to show the window discriminates",
                      " ".join(record.verdict.reasons))

    def test_modelled_references_never_contribute(self) -> None:
        modelled = [obs(f"m{i}", "active", 5.0 + i, source_type="modelled")
                    for i in range(3)]
        record = calibrate("c", good_observations() + modelled, POLICY)
        self.assertAlmostEqual(record.window[1], 3.85)
        self.assertEqual({e.reference_id for e in record.exclusions},
                         {"m0", "m1", "m2"})
        self.assertIn("belongs to the model", record.exclusions[0].reason)

    def test_a_reference_restrained_on_this_constraint_is_excluded(self) -> None:
        restrained = obs("r0", "active", 3.4, restrained=("c",))
        elsewhere = obs("r1", "active", 3.5, restrained=("other",))
        record = calibrate("c", good_observations() + [restrained, elsewhere], POLICY)
        excluded = {e.reference_id for e in record.exclusions}
        self.assertIn("r0", excluded)
        self.assertNotIn("r1", excluded)

    def test_an_unmeasured_reference_is_listed_not_dropped(self) -> None:
        record = calibrate("c", good_observations() + [obs("u0", "active", None)],
                           POLICY)
        self.assertEqual([e.reference_id for e in record.exclusions], ["u0"])
        self.assertEqual(record.verdict.n_active, len(ACTIVES))

    def test_one_complex_counted_twice_is_refused(self) -> None:
        with self.assertRaises(CalibrationError) as caught:
            calibrate("c", good_observations() + [obs("a00", "active", 3.2)], POLICY)
        self.assertIn("twice", str(caught.exception))

    def test_the_digest_does_not_depend_on_input_order(self) -> None:
        a = calibrate("c", good_observations(), POLICY)
        b = calibrate("c", list(reversed(good_observations())), POLICY)
        self.assertEqual(a.digest, b.digest)

    def test_a_different_policy_is_a_different_record(self) -> None:
        stricter = CalibrationPolicy(
            coverage=0.8, confidence=0.8, min_inactives=6,
            max_inactive_inside_upper=0.5, source="test-suite")
        self.assertNotEqual(calibrate("c", good_observations(), POLICY).digest,
                            calibrate("c", good_observations(), stricter).digest)


# ==========================================================================
# the store: a citation is checkable or it is nothing
# ==========================================================================

class _StoreCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = CalibrationStore(Path(self._tmp.name) / "calibrations")
        self.record = calibrate("hydride_transfer_distance",
                                good_observations(), POLICY)
        self.path = self.store.write(self.record)
        self.entry = self.record.calibrated_on_entry
        self.window = self.record.window

    def edit(self, mutate) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        mutate(raw)
        self.path.write_text(json.dumps(raw), encoding="utf-8")


class TestStore(_StoreCase):

    def test_a_written_record_verifies(self) -> None:
        loaded = self.store.verify(self.entry, "hydride_transfer_distance",
                                   self.window)
        self.assertEqual(loaded.digest, self.record.digest)

    def test_the_record_is_stored_under_its_digest(self) -> None:
        self.assertEqual(self.path.stem,
                         self.entry.removeprefix(CALIBRATION_PREFIX))

    def test_editing_an_observation_after_the_fact_is_caught(self) -> None:
        def widen(raw):
            raw["observations"][0]["value"] = 9.9
        self.edit(widen)
        with self.assertRaises(CalibrationError) as caught:
            self.store.verify(self.entry, "hydride_transfer_distance", self.window)
        self.assertIn("edited after it was written", str(caught.exception))

    def test_editing_the_stored_window_is_caught(self) -> None:
        def move(raw):
            raw["window"] = [1.0, 9.0]
        self.edit(move)
        with self.assertRaises(CalibrationError):
            self.store.verify(self.entry, "hydride_transfer_distance", (1.0, 9.0))

    def test_editing_the_policy_is_caught(self) -> None:
        def loosen(raw):
            raw["policy"]["confidence"] = 0.2
        self.edit(loosen)
        with self.assertRaises(CalibrationError):
            self.store.verify(self.entry, "hydride_transfer_distance", self.window)

    def test_editing_the_stored_verdict_changes_nothing_because_it_is_re_run(self) -> None:
        # Not part of the digest, by design: it is derived. Setting it to true on
        # a record that fails must not make the record pass.
        few = calibrate("hydride_transfer_distance",
                        [obs(f"a{i}", "active", d) for i, d in enumerate(ACTIVES[:5])]
                        + [obs(f"i{i}", "inactive", d)
                           for i, d in enumerate(INACTIVES)], POLICY)
        path = self.store.write(few)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["verdict"]["meets_policy"] = True
        raw["calibrated_on_entry"] = f"{CALIBRATION_PREFIX}{path.stem}"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(CalibrationError) as caught:
            self.store.verify(f"{CALIBRATION_PREFIX}{path.stem}",
                              "hydride_transfer_distance", few.window)
        self.assertIn("does not meet its own policy", str(caught.exception))

    def test_a_failed_record_is_kept_but_has_nothing_to_cite(self) -> None:
        few = calibrate("hydride_transfer_distance",
                        [obs(f"a{i}", "active", d) for i, d in enumerate(ACTIVES[:5])],
                        POLICY)
        path = self.store.write(few)
        self.assertTrue(path.is_file())
        self.assertIsNone(few.calibrated_on_entry)

    def test_a_widened_template_window_is_no_longer_calibrated(self) -> None:
        lo, hi = self.window
        with self.assertRaises(CalibrationError) as caught:
            self.store.verify(self.entry, "hydride_transfer_distance",
                              (lo - 0.5, hi))
        self.assertIn("different, uncalibrated window", str(caught.exception))

    def test_a_calibration_of_one_constraint_cannot_vouch_for_another(self) -> None:
        with self.assertRaises(CalibrationError) as caught:
            self.store.verify(self.entry, "oxyanion_Tyr_OH", self.window)
        self.assertIn("calibrates 'hydride_transfer_distance'", str(caught.exception))

    def test_a_one_sided_window_cannot_match_a_record(self) -> None:
        with self.assertRaises(CalibrationError):
            self.store.verify(self.entry, "hydride_transfer_distance", None)

    def test_a_missing_record_is_a_claim_nobody_can_open(self) -> None:
        with self.assertRaises(CalibrationError) as caught:
            self.store.verify(f"{CALIBRATION_PREFIX}{'0' * 16}",
                              "hydride_transfer_distance", self.window)
        self.assertIn("not in", str(caught.exception))

    def test_a_non_hex_digest_is_refused(self) -> None:
        with self.assertRaises(CalibrationError):
            self.store.load(f"{CALIBRATION_PREFIX}../../etc/passwd")

    def test_free_text_is_not_a_calibration_citation(self) -> None:
        with self.assertRaises(CalibrationError):
            self.store.load("1E3W")

    def test_a_record_from_another_algorithm_is_not_judged_by_this_one(self) -> None:
        # The algorithm is in the digest, so changing it also breaks the digest;
        # either way the citation does not survive.
        def relabel(raw):
            raw["algorithm"] = "wilks-range-v0"
        self.edit(relabel)
        with self.assertRaises(CalibrationError):
            self.store.verify(self.entry, "hydride_transfer_distance", self.window)


# ==========================================================================
# the context: what "calibrated" means to the evaluator
# ==========================================================================

def constraint_citing(*entries, window=(3.2, 3.85)) -> GeometryConstraint:
    return GeometryConstraint(
        name="hydride_transfer_distance", kind="distance",
        atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
        min_value=window[0], max_value=window[1], severity="gating",
        calibrated_on=list(entries), source="synthetic")


class TestContext(_StoreCase):

    def test_nothing_cited_is_not_calibrated(self) -> None:
        status = CalibrationContext(self.store).status(constraint_citing())
        self.assertFalse(status.calibrated)
        self.assertEqual(status.kind, "none")

    def test_a_verified_citation_is_calibrated(self) -> None:
        status = CalibrationContext(self.store).status(
            constraint_citing(self.entry, window=self.window))
        self.assertTrue(status.calibrated)
        self.assertEqual(status.kind, "verified")

    def test_prose_is_accepted_but_named_when_not_strict(self) -> None:
        status = CalibrationContext(self.store).status(
            constraint_citing("trust me"))
        self.assertTrue(status.calibrated)
        self.assertEqual(status.kind, "free_text")
        self.assertTrue(status.is_unverified_claim)

    def test_prose_is_refused_when_strict(self) -> None:
        status = CalibrationContext(self.store, strict=True).status(
            constraint_citing("trust me"))
        self.assertFalse(status.calibrated)
        self.assertEqual(status.kind, "free_text")

    def test_a_citation_that_cannot_be_shown_fails_closed_even_when_not_strict(self) -> None:
        status = CalibrationContext(self.store).status(
            constraint_citing(f"{CALIBRATION_PREFIX}{'0' * 16}", window=self.window))
        self.assertFalse(status.calibrated)
        self.assertEqual(status.kind, "unverifiable")
        self.assertTrue(status.problems)

    def test_a_citation_with_no_store_fails_closed(self) -> None:
        status = CalibrationContext(None).status(
            constraint_citing(self.entry, window=self.window))
        self.assertFalse(status.calibrated)
        self.assertIn("no calibration store", status.problems[0])

    def test_prose_beside_a_failing_citation_does_not_rescue_it(self) -> None:
        status = CalibrationContext(self.store).status(constraint_citing(
            f"{CALIBRATION_PREFIX}{'0' * 16}", "1E3W", window=self.window))
        self.assertFalse(status.calibrated)

    def test_authority_follows_the_context(self) -> None:
        cited = constraint_citing(self.entry, window=self.window)
        template = uncalibrated_template(
            min_value=self.window[0], max_value=self.window[1],
            calibrated_on=[self.entry])
        self.assertIs(constraint_authority(cited, template,
                                           CalibrationContext(self.store)),
                      WindowAuthority.CALIBRATED)
        self.assertIs(constraint_authority(cited, template,
                                           CalibrationContext(None)),
                      WindowAuthority.UNCALIBRATED)
        self.assertIs(template_authority(template, CalibrationContext(self.store)),
                      WindowAuthority.CALIBRATED)
        self.assertIs(template_authority(template), WindowAuthority.UNCALIBRATED)


# ==========================================================================
# measuring the references with the evaluator's own code
# ==========================================================================

class TestMeasureReferences(unittest.TestCase):

    def test_values_are_the_coordinates_distances(self) -> None:
        template = uncalibrated_template()
        out = measure_references(references([3.3, 3.6], [6.0]), template)
        values = {o.reference_id: o.value
                  for o in out["hydride_transfer_distance"]}
        self.assertAlmostEqual(values["act00"], 3.3, places=6)
        self.assertAlmostEqual(values["act01"], 3.6, places=6)
        self.assertAlmostEqual(values["ina00"], 6.0, places=6)

    def test_an_unreadable_reference_is_kept_as_unmeasured(self) -> None:
        template = uncalibrated_template()
        ghost = ReferenceComplex(
            reference_id="ghost", label="active", evidence="x",
            binding=make_binding(), path="/nonexistent/ghost.cif")
        out = measure_references([ghost], template)
        self.assertIsNone(out["hydride_transfer_distance"][0].value)

    def test_a_binding_that_names_no_such_atom_is_unmeasured_not_zero(self) -> None:
        template = uncalibrated_template()
        binding = make_binding()
        binding.cofactor_atoms["hydride_donor_C4"] = "NOPE"
        ref = ReferenceComplex("r", "active", "x", binding,
                               structure=structure_with_donor_at(3.4))
        out = measure_references([ref], template)
        self.assertIsNone(out["hydride_transfer_distance"][0].value)


class TestCalibrateTemplate(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = CalibrationStore(Path(self._tmp.name) / "cal")

    def test_end_to_end_calibration_licenses_exactly_the_measured_window(self) -> None:
        template = uncalibrated_template()
        report = calibrate_template(references(ACTIVES, INACTIVES), template,
                                    POLICY, store=self.store)
        self.assertEqual(report.calibrated, ["hydride_transfer_distance"])
        record = report.records["hydride_transfer_distance"]
        fields = calibrated_fields(record)
        self.assertAlmostEqual(fields["min_value"], ACTIVES[0], places=6)
        self.assertAlmostEqual(fields["max_value"], ACTIVES[-1], places=6)
        # the template a curator would write from those fields is verified
        calibrated = uncalibrated_template(**fields)
        self.assertTrue(CalibrationContext(self.store).status(
            calibrated.geometry_constraints[0]).calibrated)

    def test_records_are_written_whether_or_not_they_pass(self) -> None:
        report = calibrate_template(references(ACTIVES[:4], INACTIVES),
                                    uncalibrated_template(), POLICY,
                                    store=self.store)
        self.assertEqual(report.not_calibrated, ["hydride_transfer_distance"])
        self.assertTrue(Path(report.written["hydride_transfer_distance"]).is_file())

    def test_a_failing_record_yields_no_fields_to_cite(self) -> None:
        report = calibrate_template(references(ACTIVES[:4], INACTIVES),
                                    uncalibrated_template(), POLICY)
        with self.assertRaises(ValueError):
            calibrated_fields(report.records["hydride_transfer_distance"])

    def test_the_report_names_what_fell_short(self) -> None:
        report = calibrate_template(references(ACTIVES[:4], INACTIVES),
                                    uncalibrated_template(), POLICY)
        text = report.render()
        self.assertIn("not calibrated", text)
        self.assertIn("needs 14", text)

    def test_a_constraint_no_reference_could_measure_is_reported(self) -> None:
        template = uncalibrated_template(atom_a="cofactor.no_such_role")
        report = calibrate_template(references(ACTIVES, INACTIVES), template, POLICY)
        self.assertIn("hydride_transfer_distance", report.unmeasurable)
        self.assertFalse(report.records["hydride_transfer_distance"]
                         .verdict.meets_policy)

    def test_references_modelled_by_the_tool_under_test_are_excluded(self) -> None:
        refs = references(ACTIVES, INACTIVES)
        refs.append(ReferenceComplex(
            "docked", "active", "x", make_binding(),
            structure=structure_with_donor_at(5.5), source_type="modelled"))
        report = calibrate_template(refs, uncalibrated_template(), POLICY)
        record = report.records["hydride_transfer_distance"]
        self.assertAlmostEqual(record.window[1], ACTIVES[-1], places=6)
        self.assertEqual([e.reference_id for e in record.exclusions], ["docked"])


# ==========================================================================
# the evaluator no longer takes a string's word for it
# ==========================================================================

class TestEvaluatorUsesTheStore(_Harness):

    def setUp(self) -> None:
        super().setUp()
        self.store = CalibrationStore(self.workdir / "cal")
        report = calibrate_template(references(ACTIVES, INACTIVES),
                                    uncalibrated_template(), POLICY,
                                    store=self.store)
        self.record = report.records["hydride_transfer_distance"]
        self.fields = calibrated_fields(self.record)

    def evaluate(self, distance: float, template: CatalyticTemplate, **kw):
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        _, result = self.run_step(
            [candidate], [make_binding()],
            {"p1": structure_with_donor_at(distance, "p1")},
            template=template, **kw)
        return result

    def test_a_verified_window_rejects_a_pose_outside_it(self) -> None:
        template = uncalibrated_template(**self.fields)
        result = self.evaluate(4.6, template, calibration_store=self.store)
        self.assertEqual(self.pose_eval(result)["outcome"],
                         PoseOutcome.MECHANISM_VIOLATED.value)
        self.assertNotIn("calibration_unverified", self.codes(result))

    def test_a_verified_window_accepts_a_pose_inside_it(self) -> None:
        template = uncalibrated_template(**self.fields)
        result = self.evaluate(3.5, template, calibration_store=self.store)
        self.assertNotEqual(self.pose_eval(result)["outcome"],
                            PoseOutcome.MECHANISM_VIOLATED.value)

    def test_the_same_window_with_no_store_cannot_reject(self) -> None:
        template = uncalibrated_template(**self.fields)
        result = self.evaluate(4.6, template)
        self.assertEqual(self.pose_eval(result)["outcome"],
                         PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW.value)
        self.assertIn("calibration_unverified", self.codes(result))

    def test_a_window_widened_after_calibration_cannot_reject(self) -> None:
        widened = dict(self.fields, max_value=self.fields["max_value"] + 0.5)
        template = uncalibrated_template(**widened)
        result = self.evaluate(5.0, template, calibration_store=self.store)
        self.assertEqual(self.pose_eval(result)["outcome"],
                         PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW.value)

    def test_prose_still_works_by_default_and_is_named(self) -> None:
        template = uncalibrated_template(
            min_value=3.2, max_value=3.85, calibrated_on=["trust me"])
        result = self.evaluate(4.6, template)
        self.assertEqual(self.pose_eval(result)["outcome"],
                         PoseOutcome.MECHANISM_VIOLATED.value)
        flags = [f for f in result.qc_flags if f.code == "calibration_unverified"]
        self.assertTrue(flags)
        self.assertIn("not strict", flags[0].message)

    def test_prose_cannot_reject_under_strict_calibration(self) -> None:
        template = uncalibrated_template(
            min_value=3.2, max_value=3.85, calibrated_on=["trust me"])
        result = self.evaluate(4.6, template, strict_calibration=True)
        self.assertEqual(self.pose_eval(result)["outcome"],
                         PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW.value)
        # the template as a whole is uncalibrated too, not only the one pose
        self.assertIn("template_window_provisional", self.codes(result))


if __name__ == "__main__":
    unittest.main()
