"""Tests for the dependency-free structure reader/writer.

All fixtures are inline text. Nothing here touches the network or a file
outside the repository, because a parser test that depends on a downloaded
entry stops being a test of the parser the first time the entry is
re-released.

The emphasis is on the failure modes that matter scientifically: a HETATM
that must not disappear, an element that must not be guessed wrong, an
occupancy that must not be invented, and a malformed file that must raise
instead of returning a shorter atom list.
"""

from __future__ import annotations

import unittest

from eagent.science.structure_io import (
    Atom,
    Structure,
    StructureParseError,
    StructureWriteError,
    read_mmcif,
    read_pdb,
    read_structure,
    write_pdb,
)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

#: Two models; chain A holds five polymer residues (10-12, 15-16 -- note the
#: deliberate 13/14 gap), a zinc with NO element columns, a two-atom NAP
#: fragment, and one water whose occupancy column is blank.
PDB_TEXT = """\
HEADER    SYNTHETIC TEST                          01-JAN-00   TEST
MODEL        1
ATOM      1  N   MET A  10      11.104   6.134  -6.504  1.00 20.00           N
ATOM      2  CA  MET A  10      11.639   6.071  -5.147  1.00 20.00           C
ATOM      3  C   MET A  10      13.149   6.228  -5.180  1.00 20.00           C
ATOM      4  O   MET A  10      13.700   7.093  -5.858  0.50 25.00           O
ATOM      5  N  ALYS A  11      13.800   5.400  -4.400  0.60 22.00           N
ATOM      6  N  BLYS A  11      13.850   5.500  -4.300  0.40 24.00           N
ATOM      7  CA  LYS A  11      15.250   5.450  -4.300  1.00 22.00           C
ATOM      8  CB  LYS A  11      15.900   4.100  -4.050  1.00 22.00           C
ATOM      9  N   VAL A  12      15.900   6.500  -4.000  1.00 21.00           N
ATOM     10  CA  VAL A  12      17.350   6.600  -3.950  1.00 21.00           C
ATOM     11  N   TRP A  15       4.000   5.000   5.000  1.00 19.00           N
ATOM     12  CA  TRP A  15       4.500   5.500   6.200  1.00 19.00           C
ATOM     13  N   TYR A  16       7.000   5.000   5.000  1.00 19.00           N
ATOM     14  CA  TYR A  16       7.800   5.600   6.000  1.00 19.00           C
HETATM  900 ZN    ZN A 400       5.000   5.000   5.000  1.00 15.00
HETATM  901  C4  NAP A 401       3.000   4.000   5.000  1.00 18.00           C
HETATM  902  O7  NAP A 401       3.400   4.900   5.600  1.00 18.00           O
HETATM  950  O   HOH A 500       1.000   2.000   3.000       30.00           O
TER
ENDMDL
MODEL        2
ATOM      1  N   MET A  10      11.204   6.234  -6.604  1.00 20.00           N
ENDMDL
END
"""

#: Minimal but realistic mmCIF: an _atom_site loop with both label_* and
#: auth_* columns, a '.' null alt id, and two differently quoted atom names.
MMCIF_TEXT = """\
data_TEST
#
_entry.id   TEST
#
_struct.title
;A synthetic test structure
with a title that spans lines
;
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N  N    . MET A 1 1 ? 11.104 6.134 -6.504 1.00 20.00 10  MET A N     1
ATOM   2 C  CA   . MET A 1 1 ? 11.639 6.071 -5.147 1.00 20.00 10  MET A CA    1
ATOM   3 C  CB   . TYR A 1 2 ? 7.800  5.600 6.000  1.00 19.00 16  TYR A CB    1
HETATM 4 ZN ZN   . ZN  B 1 . ? 5.000  5.000 5.000  1.00 15.00 400 ZN  A ZN    1
HETATM 5 C  'C4'' . NAP C 1 . ? 3.000  4.000 5.000  0.70 18.00 401 NAP A "C4'" 1
HETATM 6 O  O    . HOH D 1 . ? 1.000  2.000 3.000  1.00 30.00 500 HOH A O     1
#
"""


def _cif_without(tag: str) -> str:
    """Drop one tag line and the corresponding column from every row."""
    lines = MMCIF_TEXT.splitlines()
    tags = [i for i, line in enumerate(lines)
            if line.startswith("_atom_site.")]
    target = None
    for pos, i in enumerate(tags):
        if lines[i].strip().lower() == f"_atom_site.{tag}".lower():
            target = (pos, i)
            break
    assert target is not None, f"fixture has no {tag}"
    pos, i = target
    out = []
    for n, line in enumerate(lines):
        if n == i:
            continue
        if line.startswith(("ATOM ", "HETATM")):
            toks = line.split()
            del toks[pos]
            out.append(" ".join(toks))
        else:
            out.append(line)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------


class TestPdbReading(unittest.TestCase):
    def setUp(self) -> None:
        self.s = read_pdb(PDB_TEXT, structure_id="test")

    def test_reads_only_the_first_model_and_says_so(self) -> None:
        self.assertEqual(self.s.models_present, [1, 2])
        self.assertEqual(self.s.model_selected, 1)
        self.assertEqual(self.s.n_atoms(), 18)
        self.assertTrue(any("models:" in n for n in self.s.quality_notes()))

    def test_explicit_model_selection(self) -> None:
        s2 = read_pdb(PDB_TEXT, model=2)
        self.assertEqual(s2.n_atoms(), 1)
        with self.assertRaises(StructureParseError):
            read_pdb(PDB_TEXT, model=7)

    def test_hetatm_groups_are_not_dropped(self) -> None:
        # The entire point of the reader: a cofactor must survive parsing.
        names = sorted(r.resname for r in self.s.ligands())
        self.assertEqual(names, ["NAP", "ZN"])
        self.assertNotIn("HOH", names)
        self.assertEqual(len(self.s.waters()), 1)

    def test_element_inferred_from_name_padding(self) -> None:
        # "ZN  " starts in column 13 -> zinc. " CA " does not -> carbon.
        zn = self.s.select(resname="ZN")
        self.assertEqual(len(zn), 1)
        self.assertEqual(zn[0].element, "ZN")
        self.assertTrue(zn[0].element_inferred)
        ca = self.s.select(resname="MET", name="CA")
        self.assertEqual(ca[0].element, "C")
        self.assertFalse(ca[0].element_inferred)
        self.assertTrue(any("elements:" in n for n in self.s.quality_notes()))

    def test_missing_occupancy_is_none_not_one(self) -> None:
        water = self.s.select(resname="HOH")
        self.assertEqual(len(water), 1)
        self.assertIsNone(water[0].occupancy)
        self.assertEqual(water[0].bfactor_or_plddt, 30.0)

    def test_select_filters(self) -> None:
        self.assertEqual(len(self.s.select(chain="A", resseq=11)), 4)
        self.assertEqual(len(self.s.select(element=["N", "O"])), 9)
        self.assertEqual(len(self.s.select(is_hetatm=True)), 4)
        self.assertEqual(len(self.s.select(resname="LYS", altloc="A")), 1)
        self.assertEqual(self.s.select(resname="NOPE"), [])

    def test_residue_and_chain_access(self) -> None:
        chain = self.s.chain("A")
        self.assertIsNotNone(chain)
        assert chain is not None
        self.assertEqual(len(chain.polymer_residues()), 5)
        lys = chain.residue(11)
        assert lys is not None
        self.assertTrue(lys.has_altloc())
        self.assertEqual(str(lys), "A/LYS11")
        self.assertIsNone(chain.residue(13))

    def test_quality_notes_report_altloc_and_partial_occupancy(self) -> None:
        notes = " ".join(self.s.quality_notes())
        self.assertIn("altloc", notes)
        self.assertIn("A/LYS11", notes)
        self.assertIn("occupancy", notes)
        self.assertIn("0.40", notes)

    def test_apo_structure_is_flagged(self) -> None:
        apo = read_pdb(
            "ATOM      1  N   MET A  10      11.104   6.134  -6.504  "
            "1.00 20.00           N\n"
        )
        self.assertTrue(
            any("no non-water HETATM" in n for n in apo.quality_notes())
        )


class TestPdbErrors(unittest.TestCase):
    def test_bad_coordinate_raises(self) -> None:
        bad = ("ATOM      1  N   MET A  10      ABCDEFGH   6.134  -6.504  "
               "1.00 20.00           N\n")
        with self.assertRaises(StructureParseError) as cm:
            read_pdb(bad)
        self.assertIn("x coordinate", str(cm.exception))

    def test_truncated_record_raises_instead_of_being_skipped(self) -> None:
        with self.assertRaises(StructureParseError) as cm:
            read_pdb("ATOM      1  N   MET A  10      11.104\n")
        self.assertIn("54", str(cm.exception))

    def test_bad_residue_number_raises(self) -> None:
        bad = ("ATOM      1  N   MET A  xx      11.104   6.134  -6.504  "
               "1.00 20.00           N\n")
        with self.assertRaises(StructureParseError):
            read_pdb(bad)

    def test_no_atoms_raises(self) -> None:
        with self.assertRaises(StructureParseError):
            read_pdb("HEADER    NOTHING HERE\nEND\n")

    def test_single_line_non_path_raises_clearly(self) -> None:
        with self.assertRaises(StructureParseError) as cm:
            read_pdb("/no/such/file.pdb")
        self.assertIn("neither an existing file", str(cm.exception))


class TestMmcifReading(unittest.TestCase):
    def setUp(self) -> None:
        self.s = read_mmcif(MMCIF_TEXT, structure_id="cif")

    def test_all_six_atoms_parsed(self) -> None:
        self.assertEqual(self.s.n_atoms(), 6)
        self.assertEqual(self.s.source_format, "mmcif")

    def test_author_numbering_is_preferred(self) -> None:
        # label_seq_id is 1,1,2 ...; auth_seq_id is 10,10,16. Using the label
        # column would renumber every residue in the structure.
        met = self.s.select(resname="MET")
        self.assertEqual({a.resseq for a in met}, {10})
        tyr = self.s.select(resname="TYR")
        self.assertEqual(tyr[0].resseq, 16)
        # auth_asym_id puts everything on chain A even though label_asym_id
        # splits the ligands onto B/C/D.
        self.assertEqual({a.chain for a in self.s.atoms()}, {"A"})

    def test_quoted_atom_names(self) -> None:
        nap = self.s.select(resname="NAP")
        self.assertEqual(len(nap), 1)
        self.assertEqual(nap[0].name, "C4'")   # from the "C4'" auth column
        self.assertEqual(nap[0].occupancy, 0.70)

    def test_ligands_and_waters(self) -> None:
        self.assertEqual(sorted(r.resname for r in self.s.ligands()),
                         ["NAP", "ZN"])
        self.assertEqual(len(self.s.waters()), 1)

    def test_type_symbol_is_used_verbatim(self) -> None:
        zn = self.s.select(resname="ZN")[0]
        self.assertEqual(zn.element, "ZN")
        self.assertFalse(zn.element_inferred)


class TestMmcifErrors(unittest.TestCase):
    def test_missing_coordinate_tag_raises(self) -> None:
        with self.assertRaises(StructureParseError) as cm:
            read_mmcif(_cif_without("Cartn_y"))
        self.assertIn("Cartn_y", str(cm.exception))

    def test_missing_type_symbol_raises_rather_than_guessing(self) -> None:
        # An unpadded mmCIF atom name cannot distinguish calcium from C-alpha.
        with self.assertRaises(StructureParseError) as cm:
            read_mmcif(_cif_without("type_symbol"))
        self.assertIn("type_symbol", str(cm.exception))

    def test_ragged_loop_raises_instead_of_truncating(self) -> None:
        # Drop one value from the last (ligand) row. A reader that tolerated
        # this would silently lose the cofactor at the end of the loop.
        lines = MMCIF_TEXT.rstrip().splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("HETATM"):
                lines[i] = " ".join(lines[i].split()[:-1])
                break
        with self.assertRaises(StructureParseError) as cm:
            read_mmcif("\n".join(lines) + "\n")
        self.assertIn("whole number of rows", str(cm.exception))

    def test_no_atom_site_loop_raises(self) -> None:
        with self.assertRaises(StructureParseError) as cm:
            read_mmcif("data_X\n_entry.id X\n#\n")
        self.assertIn("_atom_site", str(cm.exception))

    def test_unterminated_quote_raises(self) -> None:
        broken = MMCIF_TEXT.replace('"C4\'"', '"C4\'')
        with self.assertRaises(StructureParseError):
            read_mmcif(broken)

    def test_non_numeric_coordinate_raises(self) -> None:
        broken = MMCIF_TEXT.replace("11.104", "eleven")
        with self.assertRaises(StructureParseError) as cm:
            read_mmcif(broken)
        self.assertIn("Cartn_x", str(cm.exception))


class TestPdbWriting(unittest.TestCase):
    def test_round_trip_preserves_every_field(self) -> None:
        original = read_pdb(PDB_TEXT)
        text = write_pdb(original)
        again = read_pdb(text)
        self.assertEqual(original.n_atoms(), again.n_atoms())
        for a, b in zip(original.atoms(), again.atoms()):
            self.assertEqual(
                (a.name, a.element, a.resname, a.chain, a.resseq, a.icode,
                 a.altloc, a.is_hetatm),
                (b.name, b.element, b.resname, b.chain, b.resseq, b.icode,
                 b.altloc, b.is_hetatm),
            )
            self.assertAlmostEqual(a.x, b.x, places=3)
            self.assertAlmostEqual(a.y, b.y, places=3)
            self.assertAlmostEqual(a.z, b.z, places=3)
            self.assertEqual(a.occupancy, b.occupancy)
            self.assertEqual(a.bfactor_or_plddt, b.bfactor_or_plddt)

    def test_unknown_occupancy_round_trips_as_unknown(self) -> None:
        # A written "1.00" would be a fabricated measurement.
        s = read_pdb(PDB_TEXT)
        again = read_pdb(write_pdb(s))
        self.assertIsNone(again.select(resname="HOH")[0].occupancy)

    def test_writing_a_bare_selection(self) -> None:
        s = read_pdb(PDB_TEXT)
        pocket = s.select(resname=["TRP", "TYR", "ZN"])
        text = write_pdb(pocket)
        self.assertEqual(read_pdb(text).n_atoms(), 5)

    def test_oversized_fields_raise_rather_than_truncate(self) -> None:
        bad = Atom(serial=1, name="C1", element="C", resname="NAPX", chain="A",
                   resseq=1, icode="", altloc="", x=0.0, y=0.0, z=0.0,
                   is_hetatm=True)
        with self.assertRaises(StructureWriteError) as cm:
            write_pdb([bad])
        self.assertIn("NAPX", str(cm.exception))

        long_chain = Atom(serial=1, name="C1", element="C", resname="NAP",
                          chain="AAA", resseq=1, icode="", altloc="",
                          x=0.0, y=0.0, z=0.0, is_hetatm=True)
        with self.assertRaises(StructureWriteError):
            write_pdb([long_chain])

    def test_empty_selection_raises(self) -> None:
        with self.assertRaises(StructureWriteError):
            write_pdb([])


class TestDispatch(unittest.TestCase):
    def test_unknown_extension_raises(self) -> None:
        with self.assertRaises(StructureParseError):
            read_structure("model.xyz")

    def test_reads_from_disk_both_ways(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.pdb"
            p.write_text(PDB_TEXT, encoding="utf-8")
            s = read_structure(p)
            self.assertEqual(s.structure_id, "x")
            self.assertEqual(s.source_path, str(p))

            c = Path(d) / "y.cif"
            c.write_text(MMCIF_TEXT, encoding="utf-8")
            self.assertEqual(read_structure(c).n_atoms(), 6)


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
