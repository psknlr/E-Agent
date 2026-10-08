# Templates

The template library is the **only place a threshold in this pipeline is allowed
to come from.** A distance window hard-coded in a scoring function would become
this project's private definition of catalysis, and no reviewer could trace it
back to the systems it was fitted on. So every window, every catalytic residue,
every mutable zone and every positive criterion lives in a YAML file under
`configs/templates/`, loaded and cross-checked by
`eagent.harness.templates.TemplateLibrary`.

**Scope and status.** Eleven templates ship: 1 reaction, 3 family, 3 catalytic,
1 engineering, 3 assay. **All 17 geometry constraints in the library are
uncalibrated** — see §5, which is the most important section of this document.
Every shipped template records `curated_by: "e-agent build agent
(model-drafted ...; NOT reviewed by a human curator)"` and a `needs_curation`
note listing exactly what a curator must supply.

```
configs/templates/
  reaction/    ketone_to_secondary_alcohol.yaml
  family/      sdr.yaml  akr.yaml  mdr_adh.yaml
  catalytic/   sdr_nadph_carbonyl_reduction.yaml
               akr_nadph_carbonyl_reduction.yaml
               mdr_zn_nadh_carbonyl_reduction.yaml
  engineering/ sdr_substrate_pocket.yaml
  assay/       tier1_expression_screen.yaml
               tier2_product_confirmation.yaml
               tier3_quantitative_characterisation.yaml
```

The **directory is the declaration of kind**. A catalytic template placed in
`family/` would validate against the wrong model and fail with a confusing field
error, so `TEMPLATE_KINDS` maps sub-directory → pydantic model and nothing else
decides.

Read the library from the command line:

```bash
eagent templates list          # ids, kinds, file paths, families, reaction classes
eagent templates show <id>     # one template's fields and provenance
eagent templates lint          # calibration report + integrity problems
```

---

## 1. `ReactionTemplate` — what counts as the target transformation

| Field | Meaning |
| --- | --- |
| `template_id`, `reaction_class`, `description` | identity and prose |
| `required_substrate_motif` | SMARTS the substrate must match |
| `forbidden_substrate_motif` | motifs whose **presence makes the target transformation ambiguous** — not motifs that merely make a poor substrate |
| `product_motif` | SMARTS for the product |
| `chemoselectivity_notes` | what each forbidden motif is for, and what the template cannot decide |
| `stereo_requirement` | `Stereochemistry`; ships as `unspecified` |
| `creates_stereocenter` | tri-state: `true` / `false` / **`null`** |
| `provenance` | see §6 |

Two curation points in the shipped file are worth copying as a pattern.

**`creates_stereocenter: null` is the sourced answer, not a gap.** Reducing a
prochiral ketone creates a stereocentre; reducing an aldehyde or a symmetric
ketone does not. The reaction class cannot tell them apart, and
`normalize_reaction` promotes a non-null value here into the task spec as a
`template:<id>` authority for **whatever substrate is loaded**. Leaving it null
makes the step report the field unresolved and ask the operator, which is the
correct behaviour.

**Forbidding too much is as wrong as forbidding too little.** The shipped file
forbids an aldehyde (two reducible carbonyls, the aldehyde usually wins, so the
target transformation is not identifiable from a conversion number) and a
1,2-diketone (up to two new stereocentres, and which product is intended cannot
be represented here). It deliberately does **not** forbid esters, amides, nitro
groups, alkenes, nitriles or α,β-unsaturated carbonyls: a keto-ester is a normal
and valuable ketoreductase target, and those cases go to the candidate-level
chemoselectivity check in `evaluate_catalysis`, which compares which group
actually occupies the reactive position in a pose.

---

## 2. `FamilyTemplate` — how to recognise a family, and what it implies

| Field | Meaning |
| --- | --- |
| `family_name` | e.g. `SDR`, `MDR/ADH`, `AKR`. Matched through `normalise_family()`, which strips case and punctuation, so `MDR/ADH`, `mdr_adh` and `MDR-ADH` are one family rather than three |
| `pfam_ids`, `interpro_ids` | accessions; **empty is honest**, an invented IPR number would become a hard filter in mining |
| `domain_architecture` | prose describing the fold and the variable regions |
| `conserved_motifs` | each `{name, pattern, role, evidence}` — a motif **supports** a family assignment and is not an activity prediction |
| `cofactor_preference` | cofactor → evidence string |
| `typical_fold`, `oligomeric_state` | |
| `catalytic_template_ids` | the mechanisms this family may be evaluated against |
| `seed_accessions` | mining seeds; empty here because none was verified |
| `caveats` | |
| `min_independent_signals` | fixed at **3** — a family call needs agreement across signal types |

The three shipped families are **three separate mechanistic hypotheses**. They
share no fold, no catalytic residues and no cofactor-recognition logic, and
applying one family's rule to another produces a confident, fully populated,
entirely fictional mechanism. `annotate_family` makes that a construction error
rather than a runtime possibility: every catalytic mapping and cofactor rule is
reachable only through a `FamilyHypothesis` binding one family template, one
catalytic template of the **same** `family_name`, and one verified reference
sequence.

---

## 3. `CatalyticTemplate` — residues, cofactor, geometry

| Field | Meaning |
| --- | --- |
| `family_name` | must match a loaded `FamilyTemplate`, or `integrity_problems()` reports it |
| `mechanism_summary` | prose, including what the mechanism does **not** decide |
| `catalytic_residues` | each `{label, residue_types, role, functional_atoms, evidence}`, plus optional `needs_curation` / `curator_must_supply` |
| `required_cofactor` | e.g. `NADPH` |
| `required_cofactor_state` | **mandatory whenever a cofactor is named** — the model raises otherwise |
| `cofactor_ligand_codes` | wwPDB chemical component ids, e.g. `NDP` for NADPH |
| `metals` | e.g. `[ZN]` for MDR/ADH |
| `assembly_state` | the biological assembly that must be modelled |
| `geometry_constraints` | see below |
| `reference_structures` | PDB entries; **empty here, which is why nothing is calibrated** |
| `provenance` | see §6 |

```python
@model_validator(mode="after")
def _reduced_state_is_explicit(self):
    if self.required_cofactor and self.required_cofactor_state is CofactorState.UNKNOWN:
        raise TemplateError(
            f"{self.template_id}: cofactor {self.required_cofactor} declared without "
            f"an oxidation state; NAD(P)+ and NAD(P)H are not interchangeable in a "
            f"hydride-transfer model")
```

`residue_types` is a **list** so that a superfamily carrying Ser in part of its
membership and Thr in another is representable, and `functional_atoms` lists both
`OG` and `OG1`. The consequence is the right one: a Thr-carrying member presents
`OG1`, the `...ser_OG` constraint becomes **unmeasured rather than failed**, and
unmeasured is a different outcome from failed everywhere downstream.

The shipped SDR template's `mechanism_summary` carries a sentence that is worth
imitating in any new family: *"which enantiomer is formed is not decided by any
of this"*. The hydride leaves one fixed face of the nicotinamide, so the product
configuration is set by which prochiral face the pocket presents — a property of
the pocket and of the particular substrate. The template predicts the mechanism,
not the configuration.

### `GeometryConstraint`

| Field | Meaning |
| --- | --- |
| `name` | unique within the template |
| `kind` | `distance` / `angle` / `dihedral`; an angle or dihedral needs `atom_c`, and the unit auto-corrects to degrees |
| `atom_a` … `atom_d` | **role tokens**, e.g. `cofactor.hydride_donor_C4`, `substrate.electrophile`, `protein.catalytic_tyr.OH` — never raw atom indices |
| `target` + `tolerance`, or `min_value` / `max_value` | the window. A constraint defining neither is refused at load |
| `unit` | |
| `severity` | `gating` / `scoring` / `advisory` |
| `calibrated_on` | **the systems of known activity whose distribution set this window** |
| `source` | where the numbers came from, in full |

`satisfied_by(value)` is three-valued: `True`, `False`, or **`None` when the
value could not be measured**. An unresolvable role token yields `None`, and the
caller must then report the constraint as *unevaluated*, which is a different
thing from *failed*.

---

## 4. `EngineeringTemplate` and `AssayTemplate`

**`EngineeringTemplate`** — what may change, what is frozen, what has already
failed. Keyed by **family name**, not by template id, because
`propose_mutations` looks it up by the parent's family.

| Field | Meaning |
| --- | --- |
| `frozen_roles` | catalytic and cofactor-anchoring role labels, frozen in round 1 |
| `mutable_zones` | each `{name, selector, rationale, ...}` — a **search scope**, not a catalytic criterion |
| `default_shell_min_angstrom` / `default_shell_max_angstrom` | 4.0 / 8.0 |
| `known_beneficial_mutations` | each needs an evidence field; **empty here**, because no sourced precedent could be recorded |
| `known_failure_modes` | |
| `max_simultaneous_mutations_round1` | |

The shipped SDR file states the point in its own header: *"within 4.5 Å of the
substrate" says where to look for positions to consider; it says nothing about
whether a position matters.* The consuming code records bare shell membership as
scope and demands a separate reason — a measured contact, a clash, a sourced
precedent — before a position is proposed on.

**`AssayTemplate`** — how the result is measured and what counts as a hit.

| Field | Meaning |
| --- | --- |
| `tier` | 1–3 |
| `method` | prose, specific enough to run |
| `confirms_product_identity` | **a tier ≥ 2 template with this false is refused at load** |
| `chiral_capable`, `requires_authentic_standard` | |
| `controls_required` | one line per control, saying what it establishes |
| `positive_criteria` | the **pre-registration**; keys are restricted to those `ingest_results` honours, and an unrecognised key raises there rather than being ignored |
| `limit_of_detection`, `limit_unit` | |
| `replicates`, `readout_fields` | column names matching `select_batch.ASSAY_RESULT_COLUMNS`, so a returned plate parses without renaming |

Every numeric bar in the three shipped assay templates is `null`. That is
deliberate: the bar depends on this plate reader, this lysate preparation and the
measured background. With the bars null, `ingest_results` reports every well as
**undecided** rather than as a pass — the loud failure that forces the operator
to pre-register the bar before the plate is read. See
[`PROTOCOL.md`](PROTOCOL.md) §6 for what each tier may claim.

---

## 5. The 17 uncalibrated windows

```
$ eagent templates lint
geometry windows
----------------
  17 of 17 geometry constraint(s) carry no calibration set; 0 come from a
  theoretical-model template
  no window in this library was fitted to systems of known activity: a pass
  against any of them is not evidence of catalytic competence, and a failure
  may not disqualify a candidate
  ...
gating windows that are not calibrated
--------------------------------------
  none: no unfitted window is allowed to reject a candidate
```

| Template | Constraints | Severities |
| --- | --- | --- |
| `cat.sdr.nadph_carbonyl_reduction.v1` | 5 | 2 advisory, 3 scoring |
| `cat.akr.nadph_carbonyl_reduction.v1` | 6 | 2 advisory, 4 scoring |
| `cat.mdr_adh.zn_nadh_carbonyl_reduction.v1` | 6 | 2 advisory, 4 scoring |
| **total** | **17** | **6 advisory, 11 scoring, 0 gating** |

**Not one of them is `gating`, and that is the consequence of being
uncalibrated, not an accident of configuration.**

### Why these windows are not calibrated

`calibrated_on` is meant to list the systems of known activity whose measured
distribution set the window. The shipped windows are attributable to **general
chemistry** instead: van der Waals contact distances, standard hydrogen-bond
heavy-atom ranges, the Bürgi–Dunitz approach trajectory. Those are real facts
about molecules and they are not facts about *this family's catalytically
competent complexes*. `reference_structures` is empty in all three templates
because no PDB entry could be verified in this environment *when the templates
were written*, and that emptiness is why nothing could be fitted. A verified,
hash-pinned set now exists (see the subsection below); the lists stay empty
because deciding which entries qualify, and recording them with the ligand codes
their files actually use, is a curator's act and not an automatic one.

The shipped SDR hydride-transfer constraint shows the discipline in miniature:

- `min_value: 2.9` — the sum of two carbon van der Waals radii less the overlap
  tolerance `science.geometry` permits. Below it the two carbons are clashing,
  which is a modelling artefact. This bound is attributable.
- `max_value: null` — **deliberately absent.** The separation at which hydride
  transfer becomes implausible in this family is exactly the quantity that must
  be fitted to experimental complexes. The 3.2–4.0 Å figures that circulate are
  *modelling restraints from individual studies* — one group's choice, made to
  keep a simulation in a basin of interest — not measured decision boundaries.
  Copying one here as a gating window would turn that group's convention into
  this pipeline's definition of catalysis.
- `severity: advisory` — so the constraint can only detect a clash, and says so.

The Bürgi–Dunitz angle band is the same pattern: the **centre** (105–107°) is
attributable to small-molecule crystallography; the **width** (90–130°) is not a
measurement, it is set wide on purpose so that an uncalibrated angle cannot push
a candidate out.

### Why an uncalibrated window must never reject a candidate

A geometric window is a decision boundary. A boundary that was never fitted to
systems whose activity is known has an **unknown false-negative rate** — and the
error is not symmetric in its consequences:

- A **false pass** costs one well in a 96-well round, and the experiment
  discovers the error.
- A **false rejection** removes a candidate before any experiment can see it.
  The pool shrinks, the shrinkage looks principled because it was applied
  uniformly, and nothing downstream records that a criterion nobody validated
  did the removing. The campaign then concludes "this family does not work" when
  what happened is that a hand-written number excluded it.

Worse, the error is **correlated across the whole pool**. The same window is
applied to every candidate of a family, so a window that is 0.5 Å too tight
rejects not one candidate but the entire family simultaneously — and a round
that tests nothing from a family cannot learn that the window was wrong. This is
the specific mass-rejection failure that `WindowAuthority` exists to prevent.

So the authority model is three-valued and only one level may reject:

| `WindowAuthority` | `may_reject` | Caveat it carries into any report |
| --- | --- | --- |
| `calibrated` | **yes** | "window calibrated on named systems" |
| `uncalibrated` | no | "carries no calibration set; it may not drive a rejection, and a pass against it is not strong evidence" |
| `theoretical_model` | no | "template is a labelled theoretical model (theozyme); its windows may not drive rejections and passes against them are provisional" |

and the enforcement is layered:

1. **`GeometryConstraint.severity`** — the curator writes `scoring` or
   `advisory` on an unfitted window.
2. **`ConstraintRecord.may_reject`** in the library combines severity *and*
   authority, so a window marked `gating` while uncalibrated still cannot
   reject.
3. **`TemplateLibrary.integrity_problems()`** reports any gating-but-uncalibrated
   window up front, so the run records the contradiction in its manifest rather
   than discovering it one candidate at a time.
4. **`PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW`** — a failure against a
   provisional window leaves `GeometryReport.gating_passed` at `None`
   (*undecided*, not failed), is **excluded from the robustness denominator** so
   it cannot drive G to zero for a whole family, and raises an explicit
   `Uncertainty` naming the constraints.
5. **The independent verifier** checks for disqualification on an uncalibrated
   window alone and reports it as a finding.

A pass against an uncalibrated window is likewise not evidence of catalytic
competence. `TemplateLibrary.confidence_caveats(template_id)` returns the
sentences a report **must** carry when it quotes a verdict from that template —
returned as text rather than as a numeric penalty, because there is no measured
conversion from "the window was never fitted" to a number of confidence points,
and inventing one would dress a known gap up as a quantity.

### What calibration would actually require

For each window a curator must supply:

1. a set of **experimental complexes of this family** with known catalytic
   activity — ideally ternary complexes with the cofactor in the right oxidation
   state and a substrate or close analogue bound;
2. the measured distribution of the quantity across them;
3. the window chosen from that distribution, with the decision rule stated;
4. the complex identifiers written into `calibrated_on`;
5. only then may `severity` be raised to `gating`.

Until that exists, the honest statement is the one the linter prints: *17 of 17*.

### What the first real reference set found (2026-10-08)

A set of 19 experimental entries and 28 kinetic records, delivered as a
spreadsheet, is now stored and checked under `configs/references/kred_calibration/`
(see its `README.md` and `NOTICE.md`). It was audited against this template with
the project's own calibration machinery -- in memory, under two declared
policies, writing no calibration record. The result, recorded in
`docs/results/kred_reference_audit.{txt,json}`:

- **The count is still 17 of 17.** One of the 19 entries (1IPF, tropinone
  reductase II with NADPH and tropinone) meets this template's requirements as
  shipped; there are no known-inactive references. A window needs at least two
  independent actives to be proposed at all and **14** for a modest claim
  (80 % coverage at 80 % confidence; 38 for 90 %/90 %).
- **Independence is counted in lineages, not files.** The 19 entries are six
  lineages (the two *Lactobacillus* enzymes are 88 % identical over the aligned
  region), and only two of
  them place a substrate or product in the site.
- **The cofactor requirement is what excludes most of the set.** Under
  `required_cofactor: NADPH` / `reduced` / `NDP`, the HBDH, LbADH and SmBdh
  complexes carry NAD+ or NADP+, which is not a hydride donor. Dropping the
  requirement admits one more entry (6ZZO, with its own caveats) and one more
  lineage -- and still not 14.
- **Two of the five constraints can be measured on a reference at all.** The
  catalytic Tyr/Ser/Lys constraints stay unmeasured: binding them by proximity to
  the ligand would make the later measurement circular, so a curator must supply
  them from each enzyme's literature.
- **The advisory angle band does not describe the real complexes.** Measured
  C4N–C(carbonyl)–O angles are 72.6° (1IPF), 81.6° (6ZZO) and 76.3°/77.7°
  (6ZZP), all below the shipped 90–130° band, and the donor-to-carbon distances
  are 3.25–4.06 Å, all above the 2.9 Å clash floor. Three entries from two lineages
  cannot replace the band, but they are a concrete reason it is `advisory`, and
  must stay so, until a larger set says otherwise.

Nothing in the three templates was edited: `calibrated_on` is still empty
everywhere and `reference_structures` is still `[]`. A suggested patch -- the
single PDB entry that qualifies, with the ligand codes its file actually uses
(`NDP`, `TNE`) -- is a curator's decision, not an automatic one.

---

## 6. Curation rules

### A template must trace to a source, and the source type is part of the claim

`TemplateProvenance` requires both a `source_type` and at least one
`identifiers` entry, and refuses `UNSOURCED` at load:

| `TemplateSourceType` | Means | For a catalytic template |
| --- | --- | --- |
| `experimental_structure` | a PDB entry with the relevant ligands bound | the strongest basis; the only one that can support a calibrated window |
| `mechanism_literature` | an M-CSA entry or a primary mechanism paper | acceptable; what the three shipped catalytic templates use |
| `curated_database` | a curated resource | acceptable with the resource's own ceiling in mind |
| `theoretical_model` | a theozyme or other computed arrangement — **must be labelled** | admissible as a hypothesis to measure against; **never** a basis for rejection |
| `unsourced` | — | **refused at load** |

A set of coordinates a language model produced from memory is not a mechanism,
and the loader rejects it. A theozyme is a hypothesis about an arrangement of
atoms; an experimental structure is an observation of one. Collapsing them into
"sourced" would let a sketched active site reject real enzymes, so
`is_theoretical` stays reachable through `window_authority()` and the evaluation
layer downgrades confidence instead of silently trusting it.

The shipped templates' provenance blocks are themselves worth reading as a
model. The SDR catalytic template says, explicitly, why it is
`mechanism_literature` and not the other two: *"no coordinates were computed"*
(so not theoretical), *"no structure was read"* (so not experimental).

### Honesty rules the shipped files follow

- **No identifier that could not be verified appears anywhere.** No PDB entry,
  no RHEA id, no M-CSA id, no PMID, no DOI, no URL. `seed_accessions`,
  `interpro_ids`, `reference_structures` and `known_beneficial_mutations` are
  empty for that reason, not because the family has none. A plausible-looking
  identifier is *worse* than an absent one, because downstream code fetches,
  parses and trusts it.
- **Identifier prefixes say what kind of thing they are.** `EC:`,
  `PFAM:`, `textbook:` (standard teaching-level biochemistry carrying no
  external identifier yet), `eagent:` (a cross-reference into this repository).
- **`curated_by` names the drafter and its status.** Every shipped file says
  *model-drafted ... NOT reviewed by a human curator*.
- **`needs_curation: true` plus a numbered list of what a curator must supply.**
  Not "this needs work": "(1) at least one experimental ternary-complex
  structure for `reference_structures`, with the ligand codes that structure
  actually uses; (2) the M-CSA entry or primary mechanism reference; (3)
  calibrated windows, with the complexes listed in `calibrated_on`, before any
  constraint here is promoted above scoring; (4) confirmation of the 4-pro-S
  hydride face for the members in this project's pool."
- **A null that is the sourced answer says so.** The reaction template's
  `creates_stereocenter: null` carries `needs_curation: false — this null is the
  sourced answer, not a gap`.

### Collisions are refused, never resolved

`TemplateLibrary._add` raises on a duplicate `template_id` (a lookup would return
one at random and the manifest would not say which), on two family templates
claiming one family after normalisation (two family templates for one family are
two different mechanistic hypotheses), and on two engineering templates for one
family (*"silently keeping one of two would unfreeze a catalytic residue"*).

`resolve_catalytic(reaction_class, family_name)` likewise raises rather than
choosing when a family declares more than one catalytic template and nothing
says which serves the reaction class: *"A curator must declare the mapping; this
library will not guess it from the identifiers."*

A file that does not validate **aborts the whole load**. Skipping it would start
a run with a silently smaller library, and "no catalytic template for this
family" would then read as a fact about the enzyme rather than as a broken file.

---

## 7. How to add a family

Worked as a checklist. Two of the three shipped families are currently missing
their engineering template, which is exactly what `templates lint` reports:

```
integrity (2)
-------------
  - family 'AKR' has no engineering template; variants for it cannot be proposed
  - family 'MDR/ADH' has no engineering template; variants for it cannot be proposed
```

**1. Write `configs/templates/family/<name>.yaml`.**
Give it a `family_name` that `normalise_family()` will key on consistently, the
Pfam/InterPro accessions **you have verified**, the domain architecture, the
conserved motifs each with `{name, pattern, role, evidence}`, the cofactor
preference *with its evidence*, and the `catalytic_template_ids` you are about to
write. Leave anything unverified empty and say why in `caveats`.

**2. Write `configs/templates/catalytic/<name>_<mechanism>.yaml`.**
Same `family_name`. List the catalytic residues with their role, their
`functional_atoms` (all of them, where the superfamily varies) and the evidence
for each. Name the cofactor **and its oxidation state** — the model refuses
otherwise — and give the wwPDB ligand codes for the state you mean, not the
other one. Add geometry constraints addressed by **role tokens**, each with a
window, a `severity` and a `source` that says where the numbers came from. If
you have not fitted the window, `calibrated_on: []` and `severity: scoring` or
`advisory`.

**3. Point the family at it.** Add the catalytic template's id to the family
template's `catalytic_template_ids`. If a family needs more than one mechanism,
the mapping from reaction class to catalytic template must be made explicit —
`resolve_catalytic` refuses to guess.

**4. Write `configs/templates/engineering/<name>_<zone>.yaml`** if variants will
ever be proposed for this family. `frozen_roles` must use the **role labels from
your catalytic template**, because `propose_mutations` re-checks every emitted
proposal against the frozen index set and raises `FabricationGuardError` on a
violation. Define `mutable_zones` as scopes with rationales, and leave
`known_beneficial_mutations` empty unless each entry carries real evidence.

**5. Supply seeds and references.** `seed_accessions` on the family template
anchors mining; `reference_structures` on the catalytic template is what makes
calibration possible later. Both empty is a legitimate state and the linter will
keep saying so.

**6. Run the linter and read all of it.**

```bash
eagent templates lint --strict     # non-zero exit when problems remain
```

`integrity_problems()` reports: a family naming a catalytic template that is not
loaded; a family declaring none; a catalytic template claiming a family that is
not loaded; a family with no engineering template; and any gating-but-uncalibrated
window. None of these raises, because every one is a legitimate state for a
library that is honest about being incomplete — they are returned so a run can
put them in its manifest instead of meeting them one candidate at a time. The
CLI prints the calibration report at the head of **every** run for the same
reason.

**7. Pin the library if the run must be reproducible.**
`EAGENT_TEMPLATE_DIR`, or `eagent run --templates <dir>`, points a run at a
curated copy rather than at whatever is in the working tree.

---

## 中文摘要

### 模板库是本流水线里**唯一**允许产生阈值的地方

一个写死在打分函数里的距离窗口，会变成本项目私有的"催化"定义，而没有任何评审者能把它追溯
回它被拟合的那些体系。所以每一个窗口、每一个催化残基、每一个可变区、每一条阳性判据都写在
`configs/templates/` 下的 YAML 里。**子目录就是类型声明**：放错目录的模板会按错误的模型校验。

出厂 11 个模板：反应 1、家族 3、催化 3、改造 1、检测 3。

### 五类模板的要点

**`ReactionTemplate`**：`creates_stereocenter: null` **是有依据的答案，不是缺口**——还原
手性前体酮会产生立体中心，还原醛或对称酮不会，而反应类别分不开这两者；非 null 值会被
`normalize_reaction` 当作 `template:<id>` 权威写进任务规格，对**任何**载入的底物生效。
禁止基团也不能多禁：出厂文件只禁醛和 1,2-二酮，**不禁**酯、酰胺、硝基、烯烃、腈——酮酯是
正常且有价值的酮还原酶底物。

**`FamilyTemplate`**：`min_independent_signals` 固定为 3。Pfam/InterPro 为空是**诚实**，
编造一个 IPR 号会在挖掘步骤里变成硬过滤器。三个家族是**三个相互独立的机理假设**。

**`CatalyticTemplate`**：声明了辅因子就**必须**声明氧化态，否则模型直接拒绝加载——
"NAD(P)+ 和 NAD(P)H 在氢负离子转移模型里不能互换"。`residue_types` 和 `functional_atoms`
都是列表，所以带 Thr 的成员呈现 `OG1` 时，那条 `...ser_OG` 约束变成**未测量而不是未通过**，
这两者在下游是完全不同的结果。

**`EngineeringTemplate`** 按家族名索引；`frozen_roles` 必须用催化模板里的角色标签，
`propose_mutations` 会对每条产出提案复查冻结位点集合，违反就抛 `FabricationGuardError`。
`mutable_zones` 是**搜索范围**，不是催化判据。

**`AssayTemplate`**：二级及以上 `confirms_product_identity: false` 会在加载时被拒。
三个出厂模板的所有数值门槛都是 `null`，这是故意的——门槛取决于这台酶标仪、这批裂解液和实测
背景；门槛为 null 时每个孔都被报为"未判定"，这个**响亮的失败**逼着操作者在读板之前先把门槛
预注册下来。

### 17 条未标定的几何约束

`eagent templates lint` 打印："17 of 17 geometry constraint(s) carry no calibration set"。
**没有一条是 `gating`**——这是"未标定"的后果，不是配置上的偶然：6 条 advisory、11 条 scoring。

**为什么没标定**：`calibrated_on` 应当列出"活性已知、据其实测分布定出这个窗口"的那些体系。
出厂窗口的依据是**普遍化学**——范德华接触距离、标准氢键重原子范围、Bürgi–Dunitz 进攻轨迹。
这些是关于分子的真事实，但不是关于**这个家族催化活性复合物**的事实。三个催化模板的
`reference_structures` 全为空（写模板时本环境里没有任何 PDB 条目能被核实），而这个"空"正是什么都
拟合不出来的原因。（现在已有一份哈希固定的参考集，见下段；列表仍为空，因为"哪些条目合格"要由人来定。）

出厂的 SDR 氢负离子转移约束把这套纪律浓缩在一条里：下界 2.9 Å 可归因（两个碳的范德华半径
之和减去几何模块容许的重叠量，低于它就是碰撞，是建模伪影）；**上界故意留 null**——"在这个
家族里氢负离子转移变得不可信的距离"恰恰是必须用实验复合物拟合出来的量，坊间流传的 3.2–4.0 Å
是**个别研究的建模约束**（某个课题组为了把模拟留在感兴趣的势阱里而做的选择），不是测出来的
判决边界；把它抄成 gating 窗口，就是把一个课题组的惯例变成本流水线对"催化"的定义。
Bürgi–Dunitz 角同理：**中心**（105–107°）可归因，**宽度**（90–130°）不是测量值，它被故意
放宽，就是为了让一个未标定的角度推不走候选。

**第一份真实参考集的审计结果（2026-10-08）**：用户上传的 KRED 实验复合物参考集（19 个条目、28 条动力学记录）
已保存在 `configs/references/kred_calibration/`，并用本项目自己的标定机制按现有 SDR 模板审计（在内存中运行，
不写任何标定记录）。结论：**仍然是 17/17 未标定**。19 个条目里只有 1IPF（NADPH + 底物）符合模板要求，
没有已知无活性的反例；19 个条目只是 6 个谱系，真正放了底物/产物的谱系只有 2 个；80%/80% 的最低要求是 14 个
独立阳性（90%/90% 要 38 个）。放宽辅因子要求后可多纳入 6ZZO，仍远不够。实测的接近角（72.6°–81.6°）全部落在
模板 90–130° 咨询窗口之外——这只是 3 个条目的观察，但足以说明该窗口必须保持 `advisory`。催化残基（Tyr/Ser/Lys）
未绑定，需要人按各酶的文献补上。没有修改任何模板。

### 为什么未标定的窗口**绝不能**淘汰候选

几何窗口是一条判决边界。没有在"活性已知的体系"上拟合过的边界，其**假阴性率是未知的**，
而两类错误的代价**不对称**：

- **误放行**：浪费 96 孔里的一孔，而且实验本身会发现这个错误。
- **误淘汰**：候选在任何实验看到它之前就被删掉了。池子变小，而且因为是统一施加的，这种变小
  看起来**很有原则**；没有任何下游产物记录"做这件事的是一条没人验证过的判据"。项目最后得出
  "这个家族不行"，而实际发生的是一个手写的数字把它排除了。

更糟的是这个错误在整个池子里是**相关的**：同一个窗口施加于该家族的每一个候选，所以一个紧了
0.5 Å 的窗口淘汰的不是一个候选，而是**整个家族同时**——而一轮里完全没测过该家族的实验，
永远不可能发现窗口错了。这就是 `WindowAuthority` 存在要防的"大规模误杀"。

强制分五层：约束自身的 `severity`；库里 `ConstraintRecord.may_reject` 同时看 severity 和
authority（标成 gating 但未标定的窗口仍然不能淘汰）；`integrity_problems()` 把这种矛盾在
运行开头就报出来；`PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW` 让 `gating_passed` 停在
`None`（**未判定**，不是未通过）并**排除出鲁棒性分母**，使它无法把一个家族的 G 压到零；
最后独立校验器会把"仅凭未标定窗口做的淘汰"报成 finding。

反过来，**通过**一个未标定窗口同样不是催化能力的证据。`confidence_caveats()` 返回报告**必须**
附带的那几句话——返回的是文字而不是数值惩罚，因为"这个窗口从没被拟合过"到"扣多少置信度分"
之间没有任何实测换算，编一个出来就是把一个已知的缺口打扮成一个量。

### 策展规则

模板必须可溯源，且**来源类型本身是主张的一部分**：实验结构 / 机理文献 / 策展数据库 /
**明确标注的**理论模型 / 未溯源（加载即拒）。语言模型凭记忆产生的一组坐标不是机理。
theozyme 是关于原子排布的**假设**，实验结构是对它的**观测**；把两者并成"已溯源"，就会让一个
草图化的活性位点去否决真实的酶。

出厂文件遵守的诚实规则：**任何无法核实的标识符一律不出现**（PDB、RHEA、M-CSA、PMID、DOI、
URL 全都没有）——一个看起来像模像样的错误标识符比没有标识符**更糟**，因为下游代码会去抓取、
解析并信任它；`curated_by` 写明"模型起草、**未经人工策展者复核**"；`needs_curation: true`
后面跟**编号清单**，逐条写出策展者必须补什么。

冲突一律**拒绝**而不是择一：重复的 `template_id`、同一家族的两个家族模板、同一家族的两个改造
模板（"悄悄留下两者之一会解冻一个催化残基"）。某个文件校验不过会**中止整个加载**——跳过它会让
运行以一个悄悄变小的库开始，而"这个家族没有催化模板"就会被读成关于这个酶的事实，而不是一个
坏掉的文件。

### 怎么加一个家族

写家族模板 → 写催化模板（同一 `family_name`，辅因子**带氧化态**，约束用**角色记号**寻址，
没拟合就 `calibrated_on: []` + `scoring`/`advisory`）→ 把催化模板 id 写进家族模板的
`catalytic_template_ids`（一个家族多个机理时，反应类别到催化模板的映射必须显式声明，
`resolve_catalytic` 拒绝猜）→ 要做变体就写改造模板（`frozen_roles` 用催化模板的角色标签）→
补种子与参考结构 → 跑 `eagent templates lint --strict` 并**把输出全部读完** →
用 `EAGENT_TEMPLATE_DIR` 或 `--templates` 把运行钉在一份策展副本上。

当前 AKR 和 MDR/ADH **都还没有改造模板**，所以这两个家族的变体提不出来——linter 每次都会这么说。
