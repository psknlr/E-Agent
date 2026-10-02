"""Tests for the shipped configuration: templates, the pilot task, the tool registry.

These files are data, and data that validates is not the same as data that is
honest. The tests are therefore written against the claims the configuration
makes, not against its current wording:

* every template file loads into its pydantic model, so a template can never
  reach a run as a dict that happens to have the right keys;
* every catalytic template declares an explicit cofactor oxidation state, and
  its PDB ligand codes agree with that state -- a crystallographic NAD(P)+ is
  not a hydride donor, and a template that mixed the two would score an
  oxidised complex as a competent reducing one;
* no constraint whose ``calibrated_on`` is empty may be ``gating``: an
  uncalibrated window may lower confidence, never reject a candidate;
* the three ketoreductase families stay three separate mechanistic
  hypotheses, with different catalytic sets, cofactors and metals;
* no identifier anywhere is shaped like a PDB entry code, because a plausible
  fabricated PDB id is the single most dangerous value these files could
  carry -- downstream code would fetch it and trust what came back;
* the pilot task keeps its nulls and reports exactly the fields the
  reaction_spec_confirmed gate requires, so the gate refuses to open;
* every registered tool carries all four licence facets -- code, model
  weights, input database and output -- because a single licence line hides
  the conflict the registry exists to catch, and an unrecorded permission
  blocks a commercial run rather than silently allowing one.

The templates are loaded directly through the schema models rather than
through a TemplateLibrary loader, so this file keeps its teeth independently
of how the harness chooses to assemble them.

Runs under pytest, or standalone with ``python3 tests/test_configs_load.py``.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Any

import yaml

from eagent.errors import UnresolvedFieldError
from eagent.schemas.chem import CofactorState, Stereochemistry, cofactor_state_from_ligand_code
from eagent.schemas.reaction import GATE_REQUIREMENTS, ReactionClass, TaskMode, TaskSpec
from eagent.schemas.templates import (
    AssayTemplate,
    CatalyticTemplate,
    EngineeringTemplate,
    FamilyTemplate,
    ReactionTemplate,
    TemplateSourceType,
)
from eagent.science import geometry as geom
from eagent.tools.annotate_family import compile_motif
from eagent.tools.ingest_results import RECOGNISED_CRITERION_KEYS, PositiveCriterion
from eagent.tools.model_complexes import ToolKind, ToolRegistry
from eagent.tools.propose_mutations import StructuralRole
from eagent.tools.select_batch import ASSAY_RESULT_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
TEMPLATE_DIR = CONFIG_DIR / "templates"
TASK_DIR = CONFIG_DIR / "tasks"
TOOL_REGISTRY_PATH = CONFIG_DIR / "tool_registry.yaml"

#: The four facets every external tool must be registered under. Code, weights,
#: input data and outputs carry different terms from different parties.
LICENCE_FACETS: frozenset[str] = frozenset(
    {"code", "model_weights", "input_database", "output"})

#: Tools the pipeline may invoke, each of which must be registered.
REQUIRED_TOOLS: tuple[str, ...] = (
    "blast_plus", "mmseqs2", "hmmer", "mafft", "iqtree", "efi_est",
    "alphafold3", "boltz", "ligandmpnn", "rfdiffusion2",
)

#: Tools whose weights are a separately distributed artefact. Their
#: model_weights facet must be a real entry awaiting review, never a
#: "not applicable" placeholder.
TOOLS_WITH_REAL_WEIGHTS: tuple[str, ...] = (
    "alphafold3", "boltz", "ligandmpnn", "rfdiffusion2",
)

#: A wwPDB entry code: one digit then three alphanumerics. Any identifier of
#: this shape in a template would be fetched and believed.
PDB_CODE_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")

#: Role-token namespaces the geometry resolver understands.
ROLE_NAMESPACES: frozenset[str] = frozenset({"substrate", "cofactor", "protein"})

#: Marker a facet uses to say the artefact does not exist, as opposed to
#: "nobody has looked".
NOT_APPLICABLE = "NOT APPLICABLE"


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse one config file, failing loudly rather than returning ``None``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"{path} did not parse into a mapping")
    return data


def _load_all(subdir: str, model: type) -> dict[Path, Any]:
    paths = sorted((TEMPLATE_DIR / subdir).glob("*.yaml"))
    if not paths:
        raise AssertionError(f"no template files in {TEMPLATE_DIR / subdir}")
    return {p: model(**load_yaml(p)) for p in paths}


REACTION_TEMPLATES: dict[Path, ReactionTemplate] = _load_all("reaction", ReactionTemplate)
FAMILY_TEMPLATES: dict[Path, FamilyTemplate] = _load_all("family", FamilyTemplate)
CATALYTIC_TEMPLATES: dict[Path, CatalyticTemplate] = _load_all("catalytic", CatalyticTemplate)
ENGINEERING_TEMPLATES: dict[Path, EngineeringTemplate] = _load_all(
    "engineering", EngineeringTemplate)
ASSAY_TEMPLATES: dict[Path, AssayTemplate] = _load_all("assay", AssayTemplate)

ALL_TEMPLATES: dict[Path, Any] = {
    **REACTION_TEMPLATES, **FAMILY_TEMPLATES, **CATALYTIC_TEMPLATES,
    **ENGINEERING_TEMPLATES, **ASSAY_TEMPLATES,
}

TASK_PATH = TASK_DIR / "KRED_PILOT_001.yaml"


# ---------------------------------------------------------------------------
# Every file loads into its model
# ---------------------------------------------------------------------------

class TestTemplatesLoad(unittest.TestCase):
    """Loading through the model is the point: a dict is not a template."""

    def test_expected_files_are_present(self) -> None:
        expected = {
            "reaction/ketone_to_secondary_alcohol.yaml",
            "family/sdr.yaml", "family/akr.yaml", "family/mdr_adh.yaml",
            "catalytic/sdr_nadph_carbonyl_reduction.yaml",
            "catalytic/akr_nadph_carbonyl_reduction.yaml",
            "catalytic/mdr_zn_nadh_carbonyl_reduction.yaml",
            "engineering/sdr_substrate_pocket.yaml",
            "assay/tier1_expression_screen.yaml",
            "assay/tier2_product_confirmation.yaml",
            "assay/tier3_quantitative_characterisation.yaml",
        }
        present = {str(p.relative_to(TEMPLATE_DIR)) for p in ALL_TEMPLATES}
        self.assertEqual(expected - present, set(),
                         "a shipped template file has gone missing")

    def test_every_file_validates(self) -> None:
        # Construction happened at import time; this asserts the inventory is
        # non-trivial so an empty directory cannot pass the suite silently.
        # Counted from disk, not hard-coded: a fixed number here means
        # adding a template breaks an unrelated test, and the natural fix is
        # to bump the number, which is how an inventory check stops checking
        # anything.
        on_disk = sorted(TEMPLATE_DIR.rglob("*.yaml"))
        self.assertEqual(len(ALL_TEMPLATES), len(on_disk),
                         f"every YAML under {TEMPLATE_DIR} must load")

    def test_template_ids_are_unique(self) -> None:
        ids = [t.template_id for t in ALL_TEMPLATES.values()]
        self.assertEqual(len(ids), len(set(ids)),
                         "two templates share a template_id; a lookup would be "
                         "ambiguous and would silently return one of them")


# ---------------------------------------------------------------------------
# Provenance honesty
# ---------------------------------------------------------------------------

class TestProvenanceHonesty(unittest.TestCase):
    """What a template claims about where it came from."""

    def test_every_provenance_is_sourced_and_identified(self) -> None:
        for path, tpl in ALL_TEMPLATES.items():
            with self.subTest(path=path.name):
                prov = tpl.provenance
                self.assertTrue(prov.source_type.admissible)
                self.assertNotEqual(prov.source_type, TemplateSourceType.UNSOURCED)
                self.assertTrue(prov.identifiers)
                self.assertTrue(prov.notes.strip(),
                                "provenance notes are where the gaps are "
                                "recorded; an empty one hides them")

    def test_no_identifier_is_shaped_like_a_pdb_entry(self) -> None:
        """A fabricated PDB code is fetched and believed; an absent one is not."""
        for path, tpl in ALL_TEMPLATES.items():
            candidates = list(tpl.provenance.identifiers)
            candidates += list(getattr(tpl, "reference_structures", []) or [])
            candidates += list(getattr(tpl, "seed_accessions", []) or [])
            for value in candidates:
                with self.subTest(path=path.name, value=value):
                    self.assertIsNone(
                        PDB_CODE_RE.match(str(value)),
                        f"{path.name} carries {value!r}, which has the shape of "
                        f"a PDB entry code. No structure was verified when these "
                        f"templates were written; a curator must add one.")

    def test_a_template_without_a_structure_admits_the_gap(self) -> None:
        for path, tpl in CATALYTIC_TEMPLATES.items():
            with self.subTest(path=path.name):
                if tpl.reference_structures:
                    continue
                self.assertIn("needs_curation: true", tpl.provenance.notes.lower(),
                              "a catalytic template with no reference structure "
                              "must say so, and say what a curator must supply")
                self.assertIn("curator must", tpl.provenance.notes.lower())

    def test_theoretical_models_are_labelled(self) -> None:
        """No template may present modelled coordinates as mechanism evidence."""
        for path, tpl in ALL_TEMPLATES.items():
            with self.subTest(path=path.name):
                if tpl.provenance.source_type is TemplateSourceType.THEORETICAL_MODEL:
                    self.assertTrue(tpl.provenance.is_theoretical)


# ---------------------------------------------------------------------------
# Catalytic templates
# ---------------------------------------------------------------------------

class TestCatalyticTemplates(unittest.TestCase):

    def test_cofactor_oxidation_state_is_explicit(self) -> None:
        """NAD(P)+ is not a hydride donor, so the state is never left to default."""
        for path, tpl in CATALYTIC_TEMPLATES.items():
            with self.subTest(path=path.name):
                raw = load_yaml(path)
                self.assertIn("required_cofactor_state", raw,
                              "the state must be written in the file, not "
                              "inherited from a model default")
                self.assertIsNotNone(tpl.required_cofactor)
                self.assertNotEqual(tpl.required_cofactor_state,
                                    CofactorState.UNKNOWN)
                self.assertEqual(tpl.required_cofactor_state, CofactorState.REDUCED,
                                 "a reduction template needs a reduced cofactor")

    def test_ligand_codes_agree_with_the_declared_state(self) -> None:
        for path, tpl in CATALYTIC_TEMPLATES.items():
            for code in tpl.cofactor_ligand_codes:
                with self.subTest(path=path.name, code=code):
                    self.assertEqual(
                        cofactor_state_from_ligand_code(code),
                        tpl.required_cofactor_state,
                        f"{path.name} declares {tpl.required_cofactor_state.value} "
                        f"but lists ligand code {code!r}")

    def test_no_uncalibrated_constraint_gates(self) -> None:
        """An uncalibrated window may lower confidence; it may never reject."""
        for path, tpl in CATALYTIC_TEMPLATES.items():
            for constraint in tpl.geometry_constraints:
                with self.subTest(path=path.name, constraint=constraint.name):
                    if constraint.is_calibrated:
                        continue
                    self.assertIn(constraint.severity, ("scoring", "advisory"),
                                  f"{constraint.name} has calibrated_on=[] and "
                                  f"severity={constraint.severity!r}")

    def test_nothing_shipped_is_gating_yet(self) -> None:
        for path, tpl in CATALYTIC_TEMPLATES.items():
            with self.subTest(path=path.name):
                self.assertEqual(
                    tpl.gating_constraints(), [],
                    "no window in these templates has been calibrated on "
                    "experimental complexes, so none may gate")

    def test_every_window_says_where_it_came_from(self) -> None:
        for path, tpl in CATALYTIC_TEMPLATES.items():
            for constraint in tpl.geometry_constraints:
                with self.subTest(path=path.name, constraint=constraint.name):
                    self.assertTrue(
                        constraint.source.strip(),
                        "a window with no attribution is a number nobody can "
                        "defend or recalibrate")
                    lo, hi = constraint.window()
                    self.assertLess(lo, hi)

    def test_role_tokens_are_well_formed(self) -> None:
        """A malformed token is a template bug; resolve_role raises on one."""
        for path, tpl in CATALYTIC_TEMPLATES.items():
            for constraint in tpl.geometry_constraints:
                tokens = [t for t in (constraint.atom_a, constraint.atom_b,
                                      constraint.atom_c, constraint.atom_d) if t]
                for token in tokens:
                    with self.subTest(path=path.name, token=token):
                        resolved = geom.resolve_role(token, {})
                        self.assertIn(resolved.namespace, ROLE_NAMESPACES)
                        self.assertTrue(resolved.key)

    def test_angles_name_three_atoms_and_use_degrees(self) -> None:
        for path, tpl in CATALYTIC_TEMPLATES.items():
            for constraint in tpl.geometry_constraints:
                if constraint.kind != "angle":
                    continue
                with self.subTest(path=path.name, constraint=constraint.name):
                    self.assertIsNotNone(constraint.atom_c)
                    self.assertEqual(constraint.unit, "degree")

    def test_catalytic_residues_are_fully_specified(self) -> None:
        for path, tpl in CATALYTIC_TEMPLATES.items():
            self.assertTrue(tpl.catalytic_residues, f"{path.name} names no residues")
            labels = [entry.get("label") for entry in tpl.catalytic_residues]
            self.assertEqual(len(labels), len(set(labels)),
                             f"{path.name} repeats a catalytic role label")
            for entry in tpl.catalytic_residues:
                with self.subTest(path=path.name, label=entry.get("label")):
                    self.assertTrue(entry.get("label"))
                    self.assertTrue(entry.get("residue_types"))
                    self.assertTrue(entry.get("functional_atoms"))
                    self.assertTrue(str(entry.get("role", "")).strip())
                    self.assertTrue(str(entry.get("evidence", "")).strip(),
                                    "a catalytic role with no evidence string is "
                                    "an assertion with no author")

    def test_the_three_families_are_separate_hypotheses(self) -> None:
        """Different fold, different catalytic set, different metal."""
        by_family = {t.family_name: t for t in CATALYTIC_TEMPLATES.values()}
        self.assertEqual(set(by_family), {"SDR", "AKR", "MDR/ADH"})

        label_sets = {name: frozenset(e["label"] for e in t.catalytic_residues)
                      for name, t in by_family.items()}
        for a, b in (("SDR", "AKR"), ("SDR", "MDR/ADH"), ("AKR", "MDR/ADH")):
            with self.subTest(pair=(a, b)):
                self.assertEqual(
                    label_sets[a] & label_sets[b], frozenset(),
                    f"{a} and {b} share a catalytic role label; a mapping or a "
                    f"calibration would then silently transfer between two "
                    f"unrelated mechanisms")

        self.assertEqual(by_family["MDR/ADH"].metals, ["ZN"],
                         "the medium-chain family's catalytic zinc is what makes "
                         "it a different hypothesis")
        self.assertEqual(by_family["SDR"].metals, [])
        self.assertEqual(by_family["AKR"].metals, [])
        self.assertEqual(by_family["MDR/ADH"].required_cofactor, "NADH")
        self.assertEqual(by_family["SDR"].required_cofactor, "NADPH")
        self.assertEqual(by_family["AKR"].required_cofactor, "NADPH")

    def test_published_restraints_are_not_presented_as_cutoffs(self) -> None:
        """Each file must say that one study's restraint is not a universal bar."""
        for path in CATALYTIC_TEMPLATES:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8").lower()
                self.assertIn("universal", text)
                self.assertIn("modelling", text)


# ---------------------------------------------------------------------------
# Family templates
# ---------------------------------------------------------------------------

class TestFamilyTemplates(unittest.TestCase):

    def test_motifs_compile_and_carry_evidence(self) -> None:
        """A motif that cannot compile would read as 'the family is absent'."""
        for path, tpl in FAMILY_TEMPLATES.items():
            for motif in tpl.conserved_motifs:
                with self.subTest(path=path.name, motif=motif.get("name")):
                    self.assertTrue(motif.get("name"))
                    self.assertTrue(motif.get("pattern"))
                    compile_motif(str(motif["pattern"]))
                    self.assertTrue(str(motif.get("role", "")).strip())
                    self.assertTrue(str(motif.get("evidence", "")).strip())

    def test_pfam_accessions_have_the_right_shape(self) -> None:
        for path, tpl in FAMILY_TEMPLATES.items():
            self.assertTrue(tpl.pfam_ids, f"{path.name} records no Pfam accession")
            for pfam in tpl.pfam_ids:
                with self.subTest(path=path.name, pfam=pfam):
                    self.assertRegex(pfam, r"^PF\d{5}$")

    def test_cofactor_preference_carries_evidence_per_cofactor(self) -> None:
        for path, tpl in FAMILY_TEMPLATES.items():
            self.assertTrue(tpl.cofactor_preference,
                            f"{path.name} states no cofactor preference")
            for cofactor, evidence in tpl.cofactor_preference.items():
                with self.subTest(path=path.name, cofactor=cofactor):
                    self.assertTrue(evidence.strip(),
                                    "a preference with no evidence string would "
                                    "be inherited by every candidate")

    def test_each_family_points_at_a_shipped_catalytic_template(self) -> None:
        by_id = {t.template_id: t for t in CATALYTIC_TEMPLATES.values()}
        for path, tpl in FAMILY_TEMPLATES.items():
            self.assertTrue(tpl.catalytic_template_ids, f"{path.name} has none")
            for tid in tpl.catalytic_template_ids:
                with self.subTest(path=path.name, tid=tid):
                    self.assertIn(tid, by_id)
                    self.assertEqual(by_id[tid].family_name, tpl.family_name,
                                     "a family template pointing at another "
                                     "family's mechanism")

    def test_family_names_are_distinct(self) -> None:
        names = [t.family_name for t in FAMILY_TEMPLATES.values()]
        self.assertEqual(sorted(names), ["AKR", "MDR/ADH", "SDR"])

    def test_folds_are_distinct(self) -> None:
        folds = {t.family_name: (t.typical_fold or "").lower()
                 for t in FAMILY_TEMPLATES.values()}
        self.assertIn("rossmann", folds["SDR"])
        self.assertIn("barrel", folds["AKR"])
        self.assertTrue(folds["MDR/ADH"])

    def test_a_family_with_no_motifs_says_why(self) -> None:
        for path, tpl in FAMILY_TEMPLATES.items():
            with self.subTest(path=path.name):
                if tpl.conserved_motifs:
                    continue
                self.assertTrue(tpl.caveats.strip())
                self.assertIn("needs_curation: true",
                              tpl.provenance.notes.lower())


# ---------------------------------------------------------------------------
# Reaction template
# ---------------------------------------------------------------------------

class TestReactionTemplate(unittest.TestCase):

    @property
    def template(self) -> ReactionTemplate:
        return next(iter(REACTION_TEMPLATES.values()))

    def test_reaction_class_is_registered(self) -> None:
        ReactionClass(self.template.reaction_class)

    def test_stereocentre_is_left_to_the_substrate(self) -> None:
        """The honesty invariant of this file.

        normalize_reaction copies a non-null ``creates_stereocenter`` into the
        task spec as a template-sourced authority for whatever substrate is
        loaded. An aldehyde and a symmetric ketone both give an achiral
        alcohol, so a ``true`` here would be wrong for them and unfalsifiable.
        """
        self.assertIsNone(self.template.creates_stereocenter)
        self.assertEqual(self.template.stereo_requirement,
                         Stereochemistry.UNSPECIFIED)

    def test_the_class_boundary_is_stated(self) -> None:
        notes = self.template.chemoselectivity_notes.lower()
        for phrase in ("prochiral", "symmetric ketone", "aldehyde", "achiral"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, notes)

    def test_motifs_are_present_and_exclude_aldehydes(self) -> None:
        tpl = self.template
        self.assertTrue(tpl.required_substrate_motif)
        self.assertTrue(tpl.product_motif)
        self.assertTrue(tpl.forbidden_substrate_motif)
        self.assertTrue(any("CX3H1" in m for m in tpl.forbidden_substrate_motif),
                        "an aldehyde in the substrate makes the target "
                        "transformation unidentifiable and must be refused")


# ---------------------------------------------------------------------------
# Engineering template
# ---------------------------------------------------------------------------

class TestEngineeringTemplate(unittest.TestCase):

    @property
    def template(self) -> EngineeringTemplate:
        return next(iter(ENGINEERING_TEMPLATES.values()))

    def test_frozen_roles_exist_in_the_matching_catalytic_template(self) -> None:
        """A frozen role that no mapping knows protects nothing."""
        tpl = self.template
        catalytic = [t for t in CATALYTIC_TEMPLATES.values()
                     if t.family_name == tpl.family_name]
        self.assertEqual(len(catalytic), 1)
        labels = {e["label"] for e in catalytic[0].catalytic_residues}
        self.assertTrue(tpl.frozen_roles)
        self.assertEqual(set(tpl.frozen_roles) - labels, set())

    def test_zones_declare_a_recognised_structural_role(self) -> None:
        for zone in self.template.mutable_zones:
            with self.subTest(zone=zone.get("name")):
                self.assertTrue(zone.get("name"))
                self.assertTrue(str(zone.get("rationale", "")).strip())
                StructuralRole(str(zone["role"]))
                lo = zone.get("min_angstrom")
                hi = zone.get("max_angstrom")
                if lo is not None and hi is not None:
                    self.assertLess(float(lo), float(hi))

    def test_shell_bounds_are_ordered(self) -> None:
        tpl = self.template
        self.assertLess(tpl.default_shell_min_angstrom,
                        tpl.default_shell_max_angstrom)

    def test_every_precedent_carries_a_citation(self) -> None:
        """An unsourced precedent is not evidence; propose_mutations drops it."""
        for entry in self.template.known_beneficial_mutations:
            with self.subTest(entry=entry):
                self.assertTrue(str(entry.get("evidence", "")).strip())

    def test_failure_modes_are_recorded(self) -> None:
        self.assertTrue(self.template.known_failure_modes)
        for item in self.template.known_failure_modes:
            self.assertTrue(item.strip())

    def test_round_one_limits_simultaneous_substitutions(self) -> None:
        self.assertGreaterEqual(self.template.max_simultaneous_mutations_round1, 1)


# ---------------------------------------------------------------------------
# Assay templates
# ---------------------------------------------------------------------------

class TestAssayTemplates(unittest.TestCase):

    @property
    def by_tier(self) -> dict[int, AssayTemplate]:
        return {t.tier: t for t in ASSAY_TEMPLATES.values()}

    def test_three_tiers_are_shipped(self) -> None:
        self.assertEqual(sorted(self.by_tier), [1, 2, 3])

    def test_tier_one_claims_nothing_about_the_product(self) -> None:
        tier1 = self.by_tier[1]
        self.assertFalse(tier1.confirms_product_identity,
                         "a cofactor-depletion readout does not see the product")
        self.assertFalse(tier1.chiral_capable)

    def test_tiers_two_and_three_identify_the_product(self) -> None:
        for tier in (2, 3):
            with self.subTest(tier=tier):
                self.assertTrue(self.by_tier[tier].confirms_product_identity)
                self.assertTrue(self.by_tier[tier].requires_authentic_standard)

    def test_only_tier_three_claims_chiral_capability(self) -> None:
        self.assertTrue(self.by_tier[3].chiral_capable)
        self.assertFalse(self.by_tier[2].chiral_capable)
        self.assertTrue(
            self.by_tier[3].positive_criteria.get("requires_chiral_method_validated"),
            "an ee from an unvalidated separation is not a measurement")

    def test_criteria_keys_are_ones_the_ingest_step_honours(self) -> None:
        """An unrecognised key is a criterion that silently passes everything."""
        for path, tpl in ASSAY_TEMPLATES.items():
            with self.subTest(path=path.name):
                unknown = set(tpl.positive_criteria) - set(RECOGNISED_CRITERION_KEYS)
                self.assertEqual(unknown, set())
                PositiveCriterion.from_template(tpl)

    def test_an_unset_numeric_bar_is_declared_as_pending(self) -> None:
        """Pre-registration is an operator act; an invented bar is not one."""
        numeric = ("min_conversion_pct", "min_measurement_value",
                   "min_ee_target_pct", "min_fold_over_empty_vector")
        for path, tpl in ASSAY_TEMPLATES.items():
            with self.subTest(path=path.name):
                if any(tpl.positive_criteria.get(k) is not None for k in numeric):
                    continue
                self.assertIn("needs_curation: true",
                              tpl.provenance.notes.lower(),
                              "with no numeric bar the ingest step can only "
                              "report wells as undecided; the file must say so")

    def test_controls_are_named(self) -> None:
        for path, tpl in ASSAY_TEMPLATES.items():
            with self.subTest(path=path.name):
                self.assertTrue(tpl.controls_required)
                joined = " ".join(tpl.controls_required).lower()
                self.assertIn("no_enzyme", joined)
                self.assertIn("empty_vector", joined)

    def test_readout_fields_match_the_returned_plate(self) -> None:
        for path, tpl in ASSAY_TEMPLATES.items():
            with self.subTest(path=path.name):
                self.assertTrue(tpl.readout_fields)
                self.assertEqual(
                    set(tpl.readout_fields) - set(ASSAY_RESULT_COLUMNS), set(),
                    "a readout field with no column in the returned template "
                    "would be silently dropped at ingest")

    def test_tier_three_asks_for_both_enantiomer_peaks(self) -> None:
        fields = set(self.by_tier[3].readout_fields)
        self.assertIn("peak_area_target_enantiomer", fields)
        self.assertIn("peak_area_opposite_enantiomer", fields)

    def test_detection_limits_are_null_or_carry_a_unit(self) -> None:
        for path, tpl in ASSAY_TEMPLATES.items():
            with self.subTest(path=path.name):
                if tpl.limit_of_detection is None:
                    self.assertIsNone(tpl.limit_unit)
                else:
                    self.assertIsNotNone(tpl.limit_unit)


# ---------------------------------------------------------------------------
# The pilot task
# ---------------------------------------------------------------------------

class TestPilotTask(unittest.TestCase):

    @property
    def task(self) -> TaskSpec:
        return TaskSpec(**load_yaml(TASK_PATH))

    def test_it_loads_as_an_enzyme_mining_task(self) -> None:
        task = self.task
        self.assertEqual(task.task_id, "KRED_PILOT_001")
        self.assertEqual(task.task_mode, TaskMode.ENZYME_MINING)
        self.assertEqual(task.reaction.reaction_class,
                         ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
        self.assertEqual(task.parent_enzymes, [])
        self.assertEqual(task.assumptions, [])

    def test_the_nulls_survive_a_round_trip(self) -> None:
        """Pydantic must not coerce an undecided field into a default value."""
        task = self.task
        self.assertIsNone(task.reaction.substrate.isomeric_smiles)
        self.assertIsNone(task.reaction.substrate.molfile)
        self.assertIsNone(task.reaction.substrate.is_prochiral)
        self.assertIsNone(task.reaction.product.isomeric_smiles)
        self.assertIsNone(task.reaction.product.creates_new_stereocenter)
        self.assertIsNone(task.reaction.atom_mapped_reaction_smiles)
        self.assertEqual(task.reaction.product.target_stereochemistry,
                         Stereochemistry.UNSPECIFIED)
        self.assertIsNone(task.conditions.pH)
        self.assertIsNone(task.conditions.temperature_C)
        self.assertIsNone(task.conditions.solvent_system)
        self.assertIsNone(task.conditions.expression_host)

    def test_the_file_explains_that_the_nulls_are_deliberate(self) -> None:
        header = TASK_PATH.read_text(encoding="utf-8").lower()
        self.assertIn("deliberate", header)
        self.assertIn("operator", header)

    def test_the_reaction_spec_gate_reports_exactly_its_requirements(self) -> None:
        task = self.task
        self.assertEqual(task.unresolved_for("reaction_spec_confirmed"),
                         list(GATE_REQUIREMENTS["reaction_spec_confirmed"]))
        with self.assertRaises(UnresolvedFieldError) as caught:
            task.require("reaction_spec_confirmed")
        self.assertEqual(sorted(caught.exception.paths),
                         sorted(GATE_REQUIREMENTS["reaction_spec_confirmed"]))

    def test_the_synthesis_gate_also_wants_the_conditions(self) -> None:
        unresolved = self.task.unresolved_for("synthesis_authorized")
        for path in ("conditions.pH", "conditions.temperature_C",
                     "conditions.expression_host"):
            self.assertIn(path, unresolved)

    def test_no_approval_has_been_given(self) -> None:
        approval = self.task.approval
        self.assertFalse(approval.reaction_spec_confirmed)
        self.assertFalse(approval.synthesis_authorized)
        self.assertFalse(approval.functional_criteria_confirmed)

    def test_the_budget_is_the_planned_one(self) -> None:
        budget = self.task.budget
        self.assertEqual(budget.initial_sequence_target, 2000)
        self.assertEqual(budget.family_qc_pool_target, 600)
        self.assertEqual(budget.structure_pool_target, 600)
        self.assertEqual(budget.detailed_complex_target, 300)
        self.assertEqual(budget.new_constructs_round_1, 96)
        self.assertGreaterEqual(budget.detailed_complex_target,
                                budget.candidate_slots)

    def test_the_task_is_not_yet_a_stereochemical_one(self) -> None:
        """False here means *undetermined*, not 'the product is achiral'."""
        task = self.task
        self.assertFalse(task.stereo_task)
        self.assertIsNone(task.reaction.product.creates_new_stereocenter)

    def test_resolution_still_demands_an_authority(self) -> None:
        task = self.task
        with self.assertRaises(Exception):
            task.resolve("reaction.substrate.isomeric_smiles", "CC(=O)c1ccccc1",
                         source="model")


# ---------------------------------------------------------------------------
# The tool registry
# ---------------------------------------------------------------------------

class TestToolRegistry(unittest.TestCase):

    @property
    def raw(self) -> dict[str, Any]:
        return load_yaml(TOOL_REGISTRY_PATH)

    @property
    def registry(self) -> ToolRegistry:
        return ToolRegistry.from_config(self.raw)

    @staticmethod
    def _split(key: str) -> tuple[str, str]:
        tool, _, facet = key.rpartition(".")
        return tool, facet

    def test_it_loads_into_the_registry_the_tools_consult(self) -> None:
        self.assertTrue(self.registry.keys())

    def test_every_tool_has_all_four_licence_facets(self) -> None:
        """One licence line per tool hides the conflict this registry exists for."""
        facets: dict[str, set[str]] = {}
        for key in self.registry.keys():
            tool, facet = self._split(key)
            facets.setdefault(tool, set()).add(facet)
        self.assertTrue(facets)
        for tool, present in sorted(facets.items()):
            with self.subTest(tool=tool):
                self.assertEqual(
                    LICENCE_FACETS - present, set(),
                    f"{tool} is missing a licence facet; code, model weights, "
                    f"input databases and outputs carry different terms")

    def test_the_tools_the_pipeline_may_invoke_are_registered(self) -> None:
        tools = {self._split(k)[0] for k in self.registry.keys()}
        for tool in REQUIRED_TOOLS:
            with self.subTest(tool=tool):
                self.assertIn(tool, tools)
        self.assertTrue(
            any("dock" in t or "vina" in t for t in tools),
            "a docking engine must be registered before one is invoked")

    def test_each_facet_is_registered_under_its_own_kind(self) -> None:
        for key in self.registry.keys():
            entry = self.registry.get(key)
            _, facet = self._split(key)
            with self.subTest(key=key):
                self.assertEqual(entry.kind, ToolKind(facet),
                                 "the key's facet and the entry's kind disagree")

    def test_commercial_use_is_recorded_as_permitted_restricted_or_unknown(self) -> None:
        for key in self.registry.keys():
            entry = self.registry.get(key)
            with self.subTest(key=key):
                self.assertIn(entry.permits_commercial_use, (True, False, None))
                if entry.permits_commercial_use is not None:
                    self.assertIsNotNone(
                        entry.license,
                        "a commercial-use permission with no licence rests on "
                        "nothing")
                if entry.license is not None:
                    self.assertTrue(
                        entry.license_source,
                        "a licence statement must name where it was read")

    def test_an_unknown_licence_is_flagged_for_review(self) -> None:
        """Unknown must not read as permitted; it must read as unfinished."""
        for key in self.registry.keys():
            entry = self.registry.get(key)
            with self.subTest(key=key):
                if entry.license is not None:
                    continue
                if NOT_APPLICABLE in entry.notes.upper():
                    continue
                self.assertTrue(
                    entry.needs_legal_review,
                    f"{key} records no licence and is not marked NOT "
                    f"APPLICABLE, so it must still be flagged for review")
                self.assertIsNone(entry.permits_commercial_use)

    def test_every_entry_tells_a_curator_what_to_read(self) -> None:
        for key in self.registry.keys():
            entry = self.registry.get(key)
            with self.subTest(key=key):
                self.assertTrue(entry.notes.strip())
                if entry.needs_legal_review:
                    self.assertIn("curator must", entry.notes.lower())

    def test_weight_based_tools_register_real_weights(self) -> None:
        """A placeholder here is how a permissive code licence gets cited for
        non-commercial weights."""
        for tool in TOOLS_WITH_REAL_WEIGHTS:
            key = f"{tool}.model_weights"
            with self.subTest(tool=tool):
                entry = self.registry.get(key)
                self.assertNotIn(NOT_APPLICABLE, entry.notes.upper())
                self.assertTrue(entry.needs_legal_review)

    def test_no_url_is_asserted(self) -> None:
        """No endpoint could be verified here, and a wrong one would be fetched."""
        text = TOOL_REGISTRY_PATH.read_text(encoding="utf-8").lower()
        self.assertNotIn("http://", text)
        self.assertNotIn("https://", text)

    def test_keys_are_unique(self) -> None:
        keys = [item["key"] for item in self.raw["tool_registry"]]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":  # pragma: no cover - convenience runner
    unittest.main(verbosity=2)
