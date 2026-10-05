"""Tests for the geometry layer.

Every numeric expectation here is hand-computable from the coordinates in the
fixture -- a 3-4-5 triangle, axis-aligned right angles, and torsions built on
a unit frame so the dihedral sign can be checked rather than assumed. Testing
geometry against values produced by the same code it is testing proves
nothing; these are the values a reviewer can rederive on paper.

The sign of :func:`dihedral` gets its own test because +90 and -90 are the
two prochiral faces of the substrate carbonyl, which is the difference
between the (R) and the (S) alcohol.
"""

from __future__ import annotations

import math
import unittest

from eagent.errors import TemplateError
from eagent.schemas.templates import (
    GeometryConstraint,
    TemplateProvenance,
    TemplateSourceType,
)
from eagent.science.geometry import (
    BURGI_DUNITZ_LITERATURE_DEG,
    DEFAULT_VDW_OVERLAP_TOLERANCE_A,
    GeometryError,
    RoleResolver,
    angle,
    burgi_dunitz_angle,
    centroid,
    clash_count,
    clash_pairs,
    dihedral,
    distance,
    hydride_transfer_distance,
    measure_all,
    measure_constraint,
    min_distance,
    pocket_shell,
    resolve_role,
)
from eagent.science.structure_io import Atom, read_pdb

# TRP15 sits 1.0-1.4 A from the zinc, TYR16 2.0-3.0 A, MET10/VAL12 far away.
PDB_TEXT = """\
ATOM      1  N   MET A  10      11.104   6.134  -6.504  1.00 20.00           N
ATOM      2  CA  MET A  10      11.639   6.071  -5.147  1.00 20.00           C
ATOM      9  N   VAL A  12      15.900   6.500  -4.000  1.00 21.00           N
ATOM     10  CA  VAL A  12      17.350   6.600  -3.950  1.00 21.00           C
ATOM     11  N   TRP A  15       4.000   5.000   5.000  1.00 19.00           N
ATOM     12  CA  TRP A  15       4.500   5.500   6.200  1.00 19.00           C
ATOM     13  N   TYR A  16       7.000   5.000   5.000  1.00 19.00           N
ATOM     14  CA  TYR A  16       7.800   5.600   6.000  1.00 19.00           C
HETATM  900 ZN    ZN A 400       5.000   5.000   5.000  1.00 15.00          ZN
HETATM  901  C4  NAP A 401       3.000   4.000   5.000  1.00 18.00           C
HETATM  950  O   HOH A 500       1.000   2.000   3.000  1.00 30.00           O
END
"""

#: Iron has no Bondi van der Waals radius and is deliberately absent from the
#: table, so it must be reported as unscreened rather than given a made-up one.
PDB_WITH_IRON = """\
ATOM      1  N   HIS A   1       0.000   0.000   0.000  1.00 20.00           N
HETATM    2 FE    FE A 100       1.500   0.000   0.000  1.00 20.00          FE
END
"""


def _atom(x: float, y: float, z: float, element: str = "C",
          name: str = "C1", serial: int = 1) -> Atom:
    return Atom(serial=serial, name=name, element=element, resname="LIG",
                chain="A", resseq=1, icode="", altloc="", x=x, y=y, z=z,
                occupancy=1.0, bfactor_or_plddt=0.0, is_hetatm=True)


def _prov() -> TemplateProvenance:
    return TemplateProvenance(
        source_type=TemplateSourceType.EXPERIMENTAL_STRUCTURE,
        identifiers=["PDB 0TST"],
    )


class TestPrimitives(unittest.TestCase):
    def test_distance_is_a_3_4_5_triangle(self) -> None:
        self.assertAlmostEqual(distance((0, 0, 0), (3, 4, 0)), 5.0)
        self.assertAlmostEqual(distance(_atom(0, 0, 0), _atom(0, 0, 2.5)), 2.5)
        # Atom and tuple are interchangeable.
        self.assertAlmostEqual(distance(_atom(1, 1, 1), (1, 1, 4)), 3.0)

    def test_angle_right_and_straight(self) -> None:
        self.assertAlmostEqual(angle((1, 0, 0), (0, 0, 0), (0, 1, 0)), 90.0)
        self.assertAlmostEqual(angle((1, 0, 0), (0, 0, 0), (-1, 0, 0)), 180.0)
        self.assertAlmostEqual(angle((2, 0, 0), (0, 0, 0), (1, 1, 0)), 45.0)

    def test_angle_does_not_blow_up_on_exact_collinearity(self) -> None:
        # Without clamping the cosine this is where acos() raises.
        self.assertAlmostEqual(angle((5, 0, 0), (0, 0, 0), (-7, 0, 0)), 180.0)
        self.assertAlmostEqual(angle((5, 0, 0), (0, 0, 0), (3, 0, 0)), 0.0)

    def test_angle_undefined_on_coincident_points(self) -> None:
        with self.assertRaises(GeometryError):
            angle((0, 0, 0), (0, 0, 0), (1, 0, 0))

    def test_dihedral_values_and_sign(self) -> None:
        # Frame: a on +y, b at origin, c on +x. Placing d at
        # (1, cos t, sin t) gives a torsion of exactly -t degrees.
        a, b, c = (0, 1, 0), (0, 0, 0), (1, 0, 0)
        self.assertAlmostEqual(dihedral(a, b, c, (1, 1, 0)), 0.0)          # syn
        self.assertAlmostEqual(abs(dihedral(a, b, c, (1, -1, 0))), 180.0)  # anti
        self.assertAlmostEqual(dihedral(a, b, c, (1, 0, 1)), -90.0)
        self.assertAlmostEqual(dihedral(a, b, c, (1, 0, -1)), 90.0)
        # Mirroring the last atom flips the sign: this is the face swap that
        # distinguishes the two product enantiomers.
        self.assertAlmostEqual(
            dihedral(a, b, c, (1, 0.5, 0.5)),
            -dihedral(a, b, c, (1, 0.5, -0.5)),
        )

    def test_dihedral_undefined_when_collinear(self) -> None:
        with self.assertRaises(GeometryError):
            dihedral((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))

    def test_centroid_and_min_distance(self) -> None:
        pts = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)]
        cx, cy, cz = centroid(pts)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(cz, 0.5)
        self.assertAlmostEqual(
            min_distance([(0, 0, 0), (10, 0, 0)], [(3, 4, 0), (0, 0, 7)]), 5.0
        )

    def test_empty_sets_raise_instead_of_returning_infinity(self) -> None:
        # inf would silently pass a "no clash" or "outside the shell" test.
        with self.assertRaises(GeometryError):
            min_distance([], [(0, 0, 0)])
        with self.assertRaises(GeometryError):
            centroid([])

    def test_bad_point_raises(self) -> None:
        with self.assertRaises(GeometryError):
            distance(None, (0, 0, 0))
        with self.assertRaises(GeometryError):
            distance("origin", (0, 0, 0))


class TestClashScreen(unittest.TestCase):
    def setUp(self) -> None:
        self.s = read_pdb(PDB_TEXT)
        self.zn = self.s.select(resname="ZN")
        self.shell = self.s.select(resname=["TRP", "TYR"])

    def test_counts_hard_sphere_overlaps(self) -> None:
        # Zn 1.39 + N 1.55 = 2.94, minus 0.5 tolerance -> 2.44 A limit.
        #   TRP15 N  at 1.000 -> clash
        #   TRP15 CA at 1.393 -> clash (Zn 1.39 + C 1.70 - 0.5 = 2.59)
        #   TYR16 N  at 2.000 -> clash
        #   TYR16 CA at 3.033 -> no
        n = clash_count(self.shell, self.zn, DEFAULT_VDW_OVERLAP_TOLERANCE_A)
        self.assertEqual(n, 3)

    def test_tolerance_changes_the_answer_so_it_must_be_recorded(self) -> None:
        self.assertGreater(
            clash_count(self.shell, self.zn, 0.0),
            clash_count(self.shell, self.zn, 2.0),
        )

    def test_pair_filter_excludes_coordination_bonds(self) -> None:
        # A metal and its ligating atoms always "clash" under a vdW screen.
        no_metal = clash_count(
            self.shell, self.zn, DEFAULT_VDW_OVERLAP_TOLERANCE_A,
            pair_filter=lambda p, l: l.element != "ZN",
        )
        self.assertEqual(no_metal, 0)

    def test_untabulated_element_is_reported_not_silently_skipped(self) -> None:
        s = read_pdb(PDB_WITH_IRON)
        fe = s.select(resname="FE")
        protein = s.select(resname="HIS")
        with self.assertRaises(GeometryError) as cm:
            clash_count(protein, fe)
        self.assertIn("FE", str(cm.exception))
        # Opting in returns a count *and* the list of what was not screened.
        self.assertEqual(clash_count(protein, fe, allow_unscreened=True), 0)
        pairs, unscreened = clash_pairs(protein, fe)
        self.assertEqual(pairs, [])
        self.assertEqual([a.element for a in unscreened], ["FE"])

    def test_self_pairs_are_not_counted(self) -> None:
        self.assertEqual(clash_count(self.zn, self.zn), 0)

    def test_negative_tolerance_rejected(self) -> None:
        with self.assertRaises(GeometryError):
            clash_count(self.shell, self.zn, -0.1)

    def test_clash_pair_reports_its_numbers(self) -> None:
        pairs, _ = clash_pairs(self.shell, self.zn)
        worst = max(pairs, key=lambda p: p.overlap_A)
        self.assertAlmostEqual(worst.distance_A, 1.0, places=3)
        self.assertIn("overlap", worst.describe())


class TestPocketShell(unittest.TestCase):
    def setUp(self) -> None:
        self.s = read_pdb(PDB_TEXT)
        self.zn = self.s.select(resname="ZN")

    def test_shell_is_a_scope_with_an_inner_cutoff(self) -> None:
        # TRP15's atoms are all within 1.4 A, i.e. entirely inside the inner
        # cutoff, so it is NOT in the 2-4 A shell. That is the documented
        # behaviour: min_angstrom carves out the frozen first shell.
        names = [r.resname for r in pocket_shell(self.s, self.zn, 2.0, 4.0)]
        self.assertEqual(names, ["TYR"])

    def test_zero_inner_cutoff_returns_everything_within_max(self) -> None:
        names = sorted(r.resname for r in pocket_shell(self.s, self.zn, 0.0, 4.0))
        self.assertEqual(names, ["TRP", "TYR"])

    def test_waters_and_other_hetatms_are_excluded_by_default(self) -> None:
        got = pocket_shell(self.s, self.zn, 0.0, 30.0)
        self.assertNotIn("HOH", [r.resname for r in got])
        self.assertNotIn("NAP", [r.resname for r in got])
        self.assertNotIn("ZN", [r.resname for r in got])
        with_het = pocket_shell(self.s, self.zn, 0.0, 30.0, include_hetatm=True)
        self.assertIn("NAP", [r.resname for r in with_het])

    def test_results_are_sorted_and_unique(self) -> None:
        got = pocket_shell(self.s, self.zn, 0.0, 30.0)
        keys = [(r.chain, r.resseq, r.icode) for r in got]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(set(keys)))

    def test_empty_ligand_selection_raises(self) -> None:
        with self.assertRaises(GeometryError):
            pocket_shell(self.s, [], 0.0, 5.0)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(GeometryError):
            pocket_shell(self.s, self.zn, 5.0, 5.0)


class TestRoleResolution(unittest.TestCase):
    def setUp(self) -> None:
        self.c4 = _atom(0.0, 0.0, 0.0, "C", "C4", serial=1)
        self.carbonyl_c = _atom(3.4, 0.0, 0.0, "C", "C1", serial=2)
        self.carbonyl_o = _atom(4.2, 1.1, 0.0, "O", "O1", serial=3)
        self.context = {
            "cofactor": {"hydride_donor_C4": self.c4},
            "substrate": {
                "electrophile": self.carbonyl_c,
                "carbonyl_O": self.carbonyl_o,
            },
        }

    def test_resolves_a_role_token(self) -> None:
        hit = resolve_role("cofactor.hydride_donor_C4", self.context)
        self.assertTrue(hit.found)
        self.assertIs(hit.atom, self.c4)
        self.assertEqual(hit.namespace, "cofactor")
        self.assertEqual(hit.key, "hydride_donor_C4")

    def test_missing_namespace_and_missing_key_give_reasons(self) -> None:
        miss = resolve_role("protein.catalytic_Tyr_OH", self.context)
        self.assertFalse(miss.found)
        self.assertIn("namespace 'protein'", miss.reason)

        miss2 = resolve_role("cofactor.nicotinamide_N1", self.context)
        self.assertFalse(miss2.found)
        self.assertIn("no atom for role", miss2.reason)

    def test_malformed_token_raises_because_it_is_a_template_bug(self) -> None:
        with self.assertRaises(TemplateError):
            resolve_role("hydride_donor_C4", self.context)
        with self.assertRaises(TemplateError):
            resolve_role("cofactor.", self.context)

    def test_dotted_keys_split_on_the_first_dot_only(self) -> None:
        ctx = {"protein": {"A.155.OH": self.carbonyl_o}}
        self.assertTrue(resolve_role("protein.A.155.OH", ctx).found)

    def test_callable_namespace_and_bare_points(self) -> None:
        ctx = {"probe": lambda key: (1.0, 2.0, 3.0) if key == "p1" else None}
        hit = resolve_role("probe.p1", ctx)
        self.assertTrue(hit.found)
        self.assertIsNone(hit.atom)          # a point, not an Atom
        self.assertEqual(hit.coords, (1.0, 2.0, 3.0))
        self.assertFalse(resolve_role("probe.p2", ctx).found)

    def test_resolver_remembers_what_it_could_not_find(self) -> None:
        r = RoleResolver(self.context)
        r.resolve("cofactor.hydride_donor_C4")
        r.resolve("protein.catalytic_Tyr_OH")
        self.assertEqual(r.unresolved, ["protein.catalytic_Tyr_OH"])
        self.assertIn("protein.catalytic_Tyr_OH", r.reasons())

    def test_require_raises_with_the_reason(self) -> None:
        with self.assertRaises(GeometryError):
            resolve_role("protein.x", self.context).require()


class TestConstraintMeasurement(unittest.TestCase):
    def setUp(self) -> None:
        self.c4 = _atom(0.0, 0.0, 0.0, "C", "C4", serial=1)
        self.carbonyl_c = _atom(3.4, 0.0, 0.0, "C", "C1", serial=2)
        self.carbonyl_o = _atom(3.4, 1.2, 0.0, "O", "O1", serial=3)
        self.tyr_oh = _atom(3.4, 1.2, 2.0, "O", "OH", serial=4)
        self.full = RoleResolver({
            "cofactor": {"hydride_donor_C4": self.c4},
            "substrate": {"electrophile": self.carbonyl_c,
                          "carbonyl_O": self.carbonyl_o},
            "protein": {"catalytic_Tyr_OH": self.tyr_oh},
        })
        self.without_cofactor = RoleResolver({
            "substrate": {"electrophile": self.carbonyl_c,
                          "carbonyl_O": self.carbonyl_o},
        })

        self.dist = GeometryConstraint(
            name="hydride_donor_to_electrophile", kind="distance",
            atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
            target=3.5, tolerance=0.6, severity="gating",
            calibrated_on=["PDB 0TST"], source="unit test",
        )
        self.ang = GeometryConstraint(
            name="burgi_dunitz", kind="angle",
            atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
            atom_c="substrate.carbonyl_O",
            min_value=80.0, max_value=130.0, severity="scoring",
        )
        self.tors = GeometryConstraint(
            name="face_selection", kind="dihedral",
            atom_a="protein.catalytic_Tyr_OH",
            atom_b="substrate.carbonyl_O",
            atom_c="substrate.electrophile",
            atom_d="cofactor.hydride_donor_C4",
            min_value=-180.0, max_value=180.0, severity="advisory",
        )

    def test_distance_constraint_measured(self) -> None:
        v = measure_constraint(self.dist, self.full)
        assert v is not None
        self.assertAlmostEqual(v, 3.4)
        self.assertIs(self.dist.satisfied_by(v), True)

    def test_angle_constraint_measured(self) -> None:
        # C4 at origin, carbonyl C on +x, carbonyl O straight up from C:
        # the C4-C=O angle is exactly 90 degrees.
        v = measure_constraint(self.ang, self.full)
        assert v is not None
        self.assertAlmostEqual(v, 90.0)

    def test_dihedral_constraint_measured(self) -> None:
        v = measure_constraint(self.tors, self.full)
        self.assertIsNotNone(v)

    def test_unresolvable_role_gives_none_not_a_guess(self) -> None:
        # This is the behaviour the whole module is built around: an
        # unmeasurable constraint is UNEVALUATED, which is not the same thing
        # as a failed one.
        v = measure_constraint(self.dist, self.without_cofactor)
        self.assertIsNone(v)
        self.assertIsNone(self.dist.satisfied_by(v))
        self.assertEqual(self.without_cofactor.unresolved,
                         ["cofactor.hydride_donor_C4"])

    def test_degenerate_geometry_is_unmeasurable_not_failed(self) -> None:
        coincident = RoleResolver({
            "cofactor": {"hydride_donor_C4": self.c4},
            "substrate": {"electrophile": self.c4,
                          "carbonyl_O": self.carbonyl_o},
        })
        self.assertIsNone(measure_constraint(self.ang, coincident))

    def test_unknown_kind_raises_as_a_template_error(self) -> None:
        bad = GeometryConstraint(
            name="weird", kind="volume", atom_a="substrate.electrophile",
            atom_b="cofactor.hydride_donor_C4", target=1.0,
        )
        with self.assertRaises(TemplateError):
            measure_constraint(bad, self.full)

    def test_unit_dimension_mismatch_raises(self) -> None:
        bad = GeometryConstraint(
            name="mixed_units", kind="distance",
            atom_a="cofactor.hydride_donor_C4", atom_b="substrate.electrophile",
            target=3.5, tolerance=0.5, unit="degree",
        )
        with self.assertRaises(TemplateError) as cm:
            measure_constraint(bad, self.full)
        self.assertIn("unit", str(cm.exception))

    def test_measure_all_keys_by_name_and_keeps_none(self) -> None:
        got = measure_all([self.dist, self.ang], self.full)
        self.assertEqual(sorted(got), ["burgi_dunitz",
                                       "hydride_donor_to_electrophile"])
        self.assertAlmostEqual(got["hydride_donor_to_electrophile"], 3.4)
        self.assertAlmostEqual(got["burgi_dunitz"], 90.0)

        # Both constraints name the cofactor, so without it both entries are
        # present and both are None. A missing key would be indistinguishable
        # from a constraint nobody tried to evaluate.
        partial = measure_all([self.dist, self.ang], self.without_cofactor)
        self.assertEqual(set(partial), set(got))
        self.assertEqual(list(partial.values()), [None, None])

    def test_measure_all_rejects_duplicate_names(self) -> None:
        with self.assertRaises(TemplateError):
            measure_all([self.dist, self.dist], self.full)

    def test_a_raw_context_mapping_works_as_a_resolver(self) -> None:
        ctx = {"cofactor": {"hydride_donor_C4": self.c4},
               "substrate": {"electrophile": self.carbonyl_c}}
        self.assertAlmostEqual(measure_constraint(self.dist, ctx), 3.4)


class TestNamedMechanisticMeasurements(unittest.TestCase):
    def test_hydride_transfer_distance_is_heavy_atom_to_heavy_atom(self) -> None:
        c4 = _atom(0.0, 0.0, 0.0, "C", "C4")
        carbonyl_c = _atom(3.5, 0.0, 0.0, "C", "C1")
        self.assertAlmostEqual(hydride_transfer_distance(c4, carbonyl_c), 3.5)
        # Identical to a plain distance -- the wrapper is documentation, not
        # a different calculation, and must not secretly subtract a C-H bond.
        self.assertAlmostEqual(hydride_transfer_distance(c4, carbonyl_c),
                               distance(c4, carbonyl_c))

    def test_burgi_dunitz_angle_is_the_donor_C_O_angle(self) -> None:
        donor = _atom(0.0, 1.0, 0.0, "C", "C4")
        c = _atom(0.0, 0.0, 0.0, "C", "C1")
        o = _atom(1.0, 0.0, 0.0, "O", "O1")
        self.assertAlmostEqual(burgi_dunitz_angle(donor, c, o), 90.0)

        # A near-ideal Burgi-Dunitz trajectory, built by construction.
        t = math.radians(107.0)
        donor2 = _atom(math.cos(t), math.sin(t), 0.0, "C", "C4")
        self.assertAlmostEqual(burgi_dunitz_angle(donor2, c, o), 107.0, places=6)

    def test_literature_range_is_not_applied_as_a_cutoff(self) -> None:
        # A wildly non-ideal angle must still be returned as a measurement.
        donor = _atom(-1.0, 0.0, 0.0, "C", "C4")
        c = _atom(0.0, 0.0, 0.0, "C", "C1")
        o = _atom(1.0, 0.0, 0.0, "O", "O1")
        self.assertAlmostEqual(burgi_dunitz_angle(donor, c, o), 180.0)
        lo, hi = BURGI_DUNITZ_LITERATURE_DEG
        self.assertLess(lo, hi)


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
