"""Tests for :mod:`eagent.harness.templates`.

The cases are the ways a template library stops being trustworthy between the
YAML on disk and the geometry check that uses it:

* an unsourced template loading anyway, so a window with no provenance ends up
  deciding which candidates survive;
* a load failure that does not name the file, which sends a curator hunting
  through eleven of them;
* two templates quietly sharing an id, or two family templates sharing a
  family, so a lookup returns one of them and the manifest cannot say which;
* a reaction-plus-family pair resolving to a mechanism nobody declared;
* uncalibrated windows disappearing into the aggregate, so a report cannot say
  how much of its geometry was never fitted;
* a theoretical-model template reading as "sourced" and therefore as
  trustworthy as an experimental structure.

The shipped ``configs/templates`` tree is loaded as well, because the duck-typed
attribute names on the library are a contract with the ten interfaces and a
rename here would only show up as "no catalytic template" at run time.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from eagent.errors import TemplateError
from eagent.harness.templates import (
    TEMPLATE_KINDS,
    WINDOW_AUTHORITIES,
    TemplateLibrary,
    default_template_dir,
    normalise_family,
)
from eagent.schemas.templates import TemplateSourceType

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED = REPO_ROOT / "configs" / "templates"

PROVENANCE = {
    "source_type": "mechanism_literature",
    "identifiers": ["textbook:test_mechanism"],
    "notes": "written for a unit test; needs_curation: true",
}


def reaction_doc(template_id: str = "rxn.test.v1",
                 reaction_class: str = "ketone_to_secondary_alcohol",
                 **over) -> dict:
    doc = {"template_id": template_id, "reaction_class": reaction_class,
           "provenance": dict(PROVENANCE)}
    doc.update(over)
    return doc


def family_doc(template_id: str = "fam.test.v1", family_name: str = "TEST",
               catalytic_template_ids=("cat.test.v1",), **over) -> dict:
    doc = {"template_id": template_id, "family_name": family_name,
           "catalytic_template_ids": list(catalytic_template_ids),
           "provenance": dict(PROVENANCE)}
    doc.update(over)
    return doc


def catalytic_doc(template_id: str = "cat.test.v1", family_name: str = "TEST",
                  *, calibrated: bool = False, severity: str = "scoring",
                  source_type: str = "mechanism_literature", **over) -> dict:
    doc = {
        "template_id": template_id,
        "family_name": family_name,
        "required_cofactor": "NADPH",
        "required_cofactor_state": "reduced",
        "cofactor_ligand_codes": ["NDP"],
        "geometry_constraints": [{
            "name": "donor_to_electrophile",
            "kind": "distance",
            "atom_a": "cofactor.hydride_donor_C4",
            "atom_b": "substrate.electrophile",
            "min_value": 2.9,
            "max_value": 3.9,
            "severity": severity,
            "calibrated_on": ["internal:test_complex_set"] if calibrated else [],
            "source": "written for a unit test",
        }],
        "provenance": dict(PROVENANCE, source_type=source_type,
                           identifiers=["textbook:test_mechanism"]),
    }
    doc.update(over)
    return doc


def write_tree(root: Path, **by_kind) -> Path:
    """Write ``{kind: [doc, ...]}`` out as a template directory."""
    for kind, docs in by_kind.items():
        directory = root / kind
        directory.mkdir(parents=True, exist_ok=True)
        for i, doc in enumerate(docs):
            name = doc.get("template_id", f"t{i}").replace("/", "_")
            (directory / f"{name}.yaml").write_text(
                yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return root


class LoadingRefusalTests(unittest.TestCase):
    """What the loader refuses, and whether it says which file."""

    def test_unsourced_template_is_rejected_and_the_file_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = dict(reaction_doc(),
                       provenance={"source_type": "unsourced",
                                   "identifiers": ["something"]})
            write_tree(root, reaction=[bad])
            with self.assertRaises(TemplateError) as caught:
                TemplateLibrary.load(root)
            message = str(caught.exception)
            self.assertIn("unsourced", message)
            self.assertIn("rxn.test.v1.yaml", message,
                          "the error must name the file a curator has to open")

    def test_template_with_no_identifiers_is_rejected_with_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = dict(reaction_doc(),
                       provenance={"source_type": "curated_database",
                                   "identifiers": []})
            write_tree(root, reaction=[bad])
            with self.assertRaises(TemplateError) as caught:
                TemplateLibrary.load(root)
            self.assertIn("carries no identifiers", str(caught.exception))
            self.assertIn("rxn.test.v1.yaml", str(caught.exception))

    def test_missing_provenance_block_is_rejected_with_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = reaction_doc()
            bad.pop("provenance")
            write_tree(root, reaction=[bad])
            with self.assertRaises(TemplateError) as caught:
                TemplateLibrary.load(root)
            self.assertIn("does not validate", str(caught.exception))
            self.assertIn("rxn.test.v1.yaml", str(caught.exception))

    def test_a_constraint_with_no_window_is_rejected_with_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = catalytic_doc()
            doc["geometry_constraints"][0].pop("min_value")
            doc["geometry_constraints"][0].pop("max_value")
            write_tree(root, catalytic=[doc])
            with self.assertRaises(TemplateError) as caught:
                TemplateLibrary.load(root)
            self.assertIn("defines no window", str(caught.exception))
            self.assertIn("cat.test.v1.yaml", str(caught.exception))

    def test_non_mapping_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reaction").mkdir(parents=True)
            (root / "reaction" / "oops.yaml").write_text("- a\n- b\n",
                                                         encoding="utf-8")
            with self.assertRaises(TemplateError) as caught:
                TemplateLibrary.load(root)
            self.assertIn("oops.yaml", str(caught.exception))

    def test_duplicate_template_id_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, reaction=[reaction_doc()],
                       family=[family_doc(template_id="rxn.test.v1")])
            with self.assertRaises(TemplateError) as caught:
                TemplateLibrary.load(root)
            self.assertIn("already defined by", str(caught.exception))

    def test_two_family_templates_for_one_family_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, family=[
                family_doc(template_id="fam.a.v1", family_name="SDR"),
                family_doc(template_id="fam.b.v1", family_name="sdr"),
            ])
            with self.assertRaises(TemplateError) as caught:
                TemplateLibrary.load(root)
            self.assertIn("mechanistic hypotheses", str(caught.exception))

    def test_empty_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TemplateError):
                TemplateLibrary.load(Path(tmp))

    def test_missing_directory_is_refused(self):
        with self.assertRaises(TemplateError):
            TemplateLibrary.load(Path("/nonexistent/templates/for/this/test"))


class IndexAndResolutionTests(unittest.TestCase):
    """Indexing by id, by family and by reaction class."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = write_tree(
            Path(self._tmp.name),
            reaction=[reaction_doc()],
            family=[family_doc()],
            catalytic=[catalytic_doc()],
        )
        self.lib = TemplateLibrary.load(root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_indexed_by_id(self):
        self.assertEqual(sorted(self.lib.ids()),
                         ["cat.test.v1", "fam.test.v1", "rxn.test.v1"])
        self.assertIn("rxn.test.v1", self.lib)
        self.assertIsNotNone(self.lib.get("cat.test.v1"))
        self.assertIsNone(self.lib.get("nope"))

    def test_indexed_by_family_name(self):
        self.assertEqual(self.lib.family_names(), ["TEST"])
        self.assertIsNotNone(self.lib.family("test"))
        self.assertEqual([t.template_id
                          for t in self.lib.catalytic_for_family("TEST")],
                         ["cat.test.v1"])

    def test_indexed_by_reaction_class(self):
        self.assertEqual(self.lib.reaction_classes(),
                         ["ketone_to_secondary_alcohol"])
        found = self.lib.reaction_class("ketone_to_secondary_alcohol")
        self.assertEqual([t.template_id for t in found], ["rxn.test.v1"])

    def test_resolve_reaction_plus_family_to_catalytic_template(self):
        tpl = self.lib.resolve_catalytic("ketone_to_secondary_alcohol", "TEST")
        self.assertEqual(tpl.template_id, "cat.test.v1")

    def test_unknown_reaction_class_refuses_rather_than_defaults(self):
        with self.assertRaises(TemplateError) as caught:
            self.lib.resolve_catalytic("halogenation", "TEST")
        self.assertIn("known classes", str(caught.exception))

    def test_unknown_family_refuses(self):
        with self.assertRaises(TemplateError):
            self.lib.resolve_catalytic("ketone_to_secondary_alcohol", "AKR")

    def test_path_of_names_the_source_file(self):
        self.assertTrue(str(self.lib.path_of("cat.test.v1")).endswith(
            "cat.test.v1.yaml"))
        self.assertEqual(self.lib.kind_of("cat.test.v1"), "catalytic")

    def test_require_raises_for_an_unknown_id(self):
        with self.assertRaises(TemplateError):
            self.lib.require("cat.absent.v1")


class AmbiguousResolutionTests(unittest.TestCase):
    """Two catalytic templates for one family is a question, not a choice."""

    def test_ambiguous_family_refuses_to_pick(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tree(
                Path(tmp),
                reaction=[reaction_doc()],
                family=[family_doc(catalytic_template_ids=(
                    "cat.test.v1", "cat.test.v2"))],
                catalytic=[catalytic_doc(),
                           catalytic_doc(template_id="cat.test.v2")],
            )
            lib = TemplateLibrary.load(root)
            with self.assertRaises(TemplateError) as caught:
                lib.resolve_catalytic("ketone_to_secondary_alcohol", "TEST")
            self.assertIn("curator must declare", str(caught.exception))

    def test_family_naming_an_absent_catalytic_template_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tree(Path(tmp), reaction=[reaction_doc()],
                              family=[family_doc()])
            lib = TemplateLibrary.load(root)
            with self.assertRaises(TemplateError) as caught:
                lib.resolve_catalytic("ketone_to_secondary_alcohol", "TEST")
            self.assertIn("not loaded", str(caught.exception))
            self.assertIn("cat.test.v1", str(caught.exception))

    def test_family_with_no_catalytic_template_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tree(Path(tmp), reaction=[reaction_doc()],
                              family=[family_doc(catalytic_template_ids=())])
            lib = TemplateLibrary.load(root)
            with self.assertRaises(TemplateError) as caught:
                lib.resolve_catalytic("ketone_to_secondary_alcohol", "TEST")
            self.assertIn("no mechanism", str(caught.exception))


class CalibrationReportingTests(unittest.TestCase):
    """How much of the geometry was never fitted, stated in countable form."""

    def test_uncalibrated_constraints_are_listed_not_summarised_away(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tree(Path(tmp), catalytic=[catalytic_doc()])
            lib = TemplateLibrary.load(root)
            uncal = lib.uncalibrated_constraints()
            self.assertEqual([c.name for c in uncal], ["donor_to_electrophile"])
            self.assertEqual(uncal[0].authority, "uncalibrated")
            self.assertFalse(uncal[0].may_reject)

    def test_calibrated_constraint_may_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tree(Path(tmp),
                              catalytic=[catalytic_doc(calibrated=True)])
            lib = TemplateLibrary.load(root)
            self.assertEqual(lib.uncalibrated_constraints(), [])
            record = lib.constraints()[0]
            self.assertEqual(record.authority, "calibrated")
            self.assertTrue(record.may_reject)

    def test_report_counts_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tree(Path(tmp), catalytic=[
                catalytic_doc(),
                catalytic_doc(template_id="cat.test.v2", calibrated=True),
            ])
            lib = TemplateLibrary.load(root)
            report = lib.calibration_report()
            self.assertEqual(report.total, 2)
            self.assertEqual(report.calibrated, 1)
            self.assertEqual(report.uncalibrated_count, 1)
            self.assertFalse(report.fully_uncalibrated)
            self.assertIn("1 of 2", report.summary())
            payload = report.to_dict()
            self.assertEqual(payload["uncalibrated"][0]["template_id"],
                             "cat.test.v1")

    def test_a_gating_uncalibrated_window_is_an_integrity_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tree(Path(tmp), catalytic=[
                catalytic_doc(severity="gating")])
            lib = TemplateLibrary.load(root)
            problems = lib.integrity_problems()
            self.assertTrue(any("gating" in p and "uncalibrated" in p
                                for p in problems), problems)


class TheoreticalModelTests(unittest.TestCase):
    """A theozyme is admissible and is not an observation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = write_tree(
            Path(self._tmp.name),
            catalytic=[catalytic_doc(template_id="cat.theo.v1",
                                     calibrated=True,
                                     source_type="theoretical_model")])
        self.lib = TemplateLibrary.load(root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_flagged_distinctly_from_other_sourced_templates(self):
        self.assertEqual(self.lib.theoretical_template_ids(), ["cat.theo.v1"])
        self.assertTrue(self.lib.is_theoretical("cat.theo.v1"))
        tpl = self.lib.require("cat.theo.v1")
        self.assertIs(tpl.provenance.source_type,
                      TemplateSourceType.THEORETICAL_MODEL)
        self.assertTrue(tpl.provenance.source_type.admissible,
                        "a labelled theoretical model is admissible, just not "
                        "an observation")

    def test_a_calibrated_window_on_a_theoretical_template_may_not_reject(self):
        authority = self.lib.window_authority("cat.theo.v1",
                                              "donor_to_electrophile")
        self.assertEqual(authority, "theoretical_model")
        record = self.lib.constraints()[0]
        self.assertFalse(record.may_reject,
                         "tuning a window around computed coordinates does not "
                         "make the coordinates an observation")

    def test_caveats_say_so_in_words_not_as_a_penalty(self):
        caveats = self.lib.confidence_caveats("cat.theo.v1")
        self.assertTrue(any("theoretical model" in c for c in caveats), caveats)

    def test_window_authority_rejects_an_unknown_constraint(self):
        with self.assertRaises(TemplateError):
            self.lib.window_authority("cat.theo.v1", "no_such_window")


class VocabularyTests(unittest.TestCase):
    """The authority tokens must stay the same words as the evaluation layer's."""

    def test_window_authority_vocabulary_matches_evaluate_catalysis(self):
        from eagent.tools.evaluate_catalysis import WindowAuthority

        self.assertEqual(set(WINDOW_AUTHORITIES),
                         {a.value for a in WindowAuthority},
                         "templates.py repeats these tokens rather than "
                         "importing the evaluation stack; if they drift, a "
                         "report and the evaluator disagree about which "
                         "windows may reject a candidate")

    def test_kinds_cover_the_five_template_types(self):
        self.assertEqual(sorted(TEMPLATE_KINDS),
                         ["assay", "catalytic", "engineering", "family",
                          "reaction"])

    def test_family_normalisation_collapses_punctuation(self):
        self.assertEqual(normalise_family("MDR/ADH"), "MDRADH")
        self.assertEqual(normalise_family("mdr_adh"), "MDRADH")
        self.assertEqual(normalise_family(None), "")


class ShippedLibraryTests(unittest.TestCase):
    """The real ``configs/templates`` tree, loaded the way a run loads it."""

    @classmethod
    def setUpClass(cls):
        cls.lib = TemplateLibrary.load(SHIPPED)

    def test_default_dir_points_at_the_shipped_tree(self):
        self.assertEqual(default_template_dir().resolve(), SHIPPED.resolve())

    def test_all_eleven_shipped_templates_load(self):
        self.assertEqual(len(self.lib), 11)

    def test_the_three_families_stay_three_hypotheses(self):
        self.assertEqual(self.lib.family_names(), ["AKR", "MDR/ADH", "SDR"])
        for family in self.lib.family_names():
            tpl = self.lib.resolve_catalytic("ketone_to_secondary_alcohol",
                                             family)
            self.assertEqual(tpl.family_name, family)

    def test_the_shipped_geometry_is_entirely_uncalibrated(self):
        report = self.lib.calibration_report()
        self.assertTrue(report.fully_uncalibrated,
                        "the shipped templates say so in their own comments; "
                        "if one becomes calibrated this test should be updated "
                        "with the complexes it was fitted on")
        self.assertEqual(report.gating_uncalibrated, (),
                         "an unfitted window may not be gating")

    def test_no_shipped_template_claims_to_be_a_theoretical_model(self):
        self.assertEqual(self.lib.theoretical_template_ids(), [])

    def test_integrity_problems_are_reported_not_raised(self):
        problems = self.lib.integrity_problems()
        self.assertTrue(any("engineering template" in p for p in problems),
                        "AKR and MDR/ADH ship without one; the library must "
                        "say so rather than fail at variant-proposal time")

    def test_duck_typed_surface_the_interfaces_reach_for(self):
        from eagent.science.scorecard import _resolve_catalytic_template
        from eagent.tools.propose_mutations import ProposeMutations

        found = _resolve_catalytic_template(
            self.lib, "cat.sdr.nadph_carbonyl_reduction.v1")
        self.assertIsNotNone(found)
        self.assertEqual(found.family_name, "SDR")
        self.assertIsNotNone(ProposeMutations)
        self.assertIn("SDR", self.lib.engineering_templates,
                      "ProposeMutations looks the engineering template up by "
                      "family name, not by template id")

    def test_retrieve_evidence_can_read_the_family_names_off_the_library(self):
        from eagent.context import RunContext
        from eagent.provenance import RunManifest
        from eagent.schemas import TaskSpec
        from eagent.tools.retrieve_evidence import _families_from_context

        with tempfile.TemporaryDirectory() as tmp:
            task = TaskSpec(task_id="T")
            ctx = RunContext(task=task, workdir=Path(tmp),
                             manifest=RunManifest(run_id="R", task_id="T"),
                             templates=self.lib)
            self.assertEqual(set(_families_from_context(ctx, None)),
                             {"SDR", "AKR", "MDR/ADH"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
