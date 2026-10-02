# Research Protocol

The protocol E-Agent actually implements: what happens, in what order, what each
stage is allowed to conclude, and where it stops and asks a person.

**Scope and status.** This describes code under `src/eagent/tools/`,
`src/eagent/science/` and `configs/templates/`. No round described here has been
run in a laboratory. The numbers in the funnel are the shipped defaults of
`configs/tasks/KRED_PILOT_001.yaml`; they are a **resource plan**, not a
prediction, and §3 says why that distinction is load-bearing.

---

## 1. Three layers, in order of risk

The project is deliberately layered, and the layers are not alternatives. Each
one is only meaningful on top of the one before it.

### Layer 1 — natural enzyme mining (the first version)

Find natural enzymes that already perform the target transformation on the
target substrate. This is the version that has to work first, for a reason that
is diagnostic rather than conservative: **only a confirmed natural parent
separates "we never found the right family" from "we found a workable scaffold
with the wrong pocket".** Those two worlds need opposite next steps, and without
a confirmed hit the campaign cannot tell them apart. Layer 1 is the whole of
`PROTOCOL_ORDER` steps 1–9.

### Layer 2 — substrate-directed engineering (the second stage)

Take a parent that an experiment confirmed, and change the substrate pocket so
it accepts the target molecule in a productive orientation with the wanted
configuration. This is `propose_mutations`, and it is **tenth** in the protocol
order on purpose. It starts from `ExperimentRecord`s that confirm a parent, and
running it on an unconfirmed candidate requires an operator to record a
`ParentOverride` with a reason — which is then attached to every resulting
proposal in `contradicting_evidence`, so the weakness travels with the variant
into the plate map and into whatever is eventually written up.

Round 1 of engineering freezes the catalytic and cofactor-anchoring roles the
`EngineeringTemplate` lists. Mutating a catalytic residue and a pocket residue in
the same round produces a dead variant and no information: the single-mutant
controls that would attribute the loss do not exist, and "it stopped working" is
consistent with both changes. The freeze is enforced, not advised — a frozen role
that cannot be located on a parent (because the catalytic mapping is incomplete)
means the freeze **cannot be checked**, and the parent is skipped with a blocker
rather than designed on.

### Layer 3 — de novo design (a later, high-risk branch, evaluated separately)

Designing a catalyst from scratch is a different risk class with a different
success rate and a different evidence standard, and folding it into the same
funnel would let a very low expected hit rate be averaged into the pipeline's
headline number. It is therefore positioned **after** layers 1 and 2, with its
own acceptance criteria, in `docs/ROADMAP.md`. No part of it is implemented.
`configs/tool_registry.yaml` registers RFdiffusion2 only so that its four
licence facets exist to be filled in by a curator; nothing calls it.

The data layer carries the same ordering independently, as typed work packages:
`datalayer/plan.py` defines `STAGE_1` (close the natural-enzyme discovery loop),
`STAGE_2` (variant-effect and kinetics corpora, gated on a parent family that
stage 1 produced), `STAGE_3` (novelty and generality, under an explicit budget).
`validate_plan()` refuses a plan where a package depends on something no earlier
package produces, and `answer_stage2_gate()` makes stage 2's gating question
computable rather than assertable.

---

## 2. The staged funnel

```
  retrieval pool            2000   mine_sequences      multi-seed, offline,
                                                       nothing invented
         │
         ▼
  family QC pool             600   annotate_family     ≥3 independent signals
         │                                             per family call
         ▼
  structure pool             600   prepare_structures  one structure per
         │                                             candidate, with the reason
         ▼
  detailed complex pool      300   model_complexes     enzyme+substrate+cofactor,
         │                         evaluate_catalysis  two routes, all poses kept
         ▼
  round-1 construct batch     96   select_batch        behind a recorded approval
```

### These are resource plans, not attrition rates

Nothing in the funnel asserts that 600 of 2000 sequences *will* survive family
QC. The numbers say how each stage is **sized** — how much compute, curation and
GPU time the round is budgeting — and they are written in the task file where a
site can change them. The schema calls them planning targets
(`Budget`, "These are planning targets, not predetermined pass rates"), the CLI
prints them under the heading *"planned scale (from the task file, not a
prediction)"*, and the task file's own comment block repeats it.

The distinction matters because the opposite reading licenses two bad moves. If
600 is a *rate*, a stage that yields 200 looks like a failure to be fixed by
relaxing a filter — and `mine_sequences` is built so that this is impossible:
the `RetentionPolicy` is fingerprinted on entry and re-checked after filtering,
so a future edit that mutates it mid-run raises `FabricationGuardError` instead
of quietly producing a padded pool. And if 96 is a rate, a short batch looks like
a bug rather than a finding; `BatchPlan` refuses a short batch that does not
record `shortfall_reason`, and there is no code path anywhere that lengthens a
batch by relaxing a standard.

### The detailed-modelling pool must be materially larger than the batch

If the detailed complex pool is the same size as the construct batch, "selection"
is just a rename of the pool: every candidate that got modelled gets ordered, the
gates have nothing to exclude from, the diversity and uncertainty roles have
nothing to choose between, and the round tests whatever the modelling budget
happened to reach.

`Budget` enforces the floor —
`detailed_complex_target >= candidate_slots`, raising
*"detailed_complex_target must exceed the construct slots, otherwise selection
has nothing to select from"* — and the shipped ratio is **300 : 96**, roughly
three to one. Note honestly that the validator permits exact equality while its
message says "exceed"; equality passes the check and still leaves selection no
room, so treat 1:1 as a misconfiguration even though the schema will load it.

`candidate_slots` is also not always the construct count. If
`constructs_include_controls` is true, the reserved control slots come out of the
cap first, because a positive-control enzyme needs a gene like everything else —
see §5.

---

## 3. Stage by stage, and what each may conclude

**1. `normalize_reaction` — make the task falsifiable.**
Reads what the operator supplied, verifies what can be verified, and produces a
precise list of what is still missing. It does the opposite of completing the
specification. Three refusals are deliberate: a name is not a structure; whether
a stereocentre is created is **not** inferred from the reaction class (an
aldehyde and a symmetric ketone both give an achiral alcohol, and the class label
is itself an operator assertion that is frequently the thing that is wrong); and
in mode B it refuses to name a best enzyme for an unspecified substrate. Even
with rdkit installed, a perceived stereocentre is offered as a *proposal
requiring confirmation* — `Assumption` admits no "a tool said so" authority.

**2. `retrieve_evidence` — assemble the evidence and show its holes.**
The **query plan is written before anything is retrieved**, because recall cannot
be judged from results, only from queries: a human reads
`evidence_query_plan.yaml` and says "you never searched the cyclic ketones",
which is a conversation that cannot happen if the plan lives only inside the
code. The evidence matrix is binned on family × substrate chemotype with the
outcome classes kept apart, and separates wild-type from engineered success, so
a chemotype that only works after protein engineering cannot be read as a
chemotype with mature natural enzymes. A cache miss is reported as a gap naming
the exact file a curator must place — **a cache miss and a measured negative are
opposite facts, and only one of them is evidence.**

**3. `mine_sequences` — widen the pool from characterised seeds.**
Multi-seed by construction: a single-seed run inherits one scaffold's unlucky
property (an unstable fold, a narrow pocket, a cofactor preference) into every
candidate, so the whole round fails for one reason while the pool looks large. A
one-element seed set requires `allow_single_seed=True` **and** a written
justification, and is flagged and recorded. Seeds must carry at least
homolog-level experimental support; mining from annotation produces a pool with
no experimental anchor that still looks well-founded. Search output gives
subject identifiers; residues come from the local sequence database file, and an
identifier that cannot be resolved there is dropped and counted, never
reconstructed.

**4. `annotate_family` — a family call from several separable signals.**
`GxxxGxG` appears in NAD(P)-binding proteins across unrelated superfamilies;
`YxxxK` appears in proteins that are not SDRs. Any one of these, alone, assigns a
family that then silently licenses a mechanism, a cofactor and a set of catalytic
residues. Four separable signal types are gathered — identity to a family
reference, domain architecture, conserved positions, catalytic-residue
correspondence — and the count goes to `FamilyAnnotation.recompute_confidence`
against the family template's `min_independent_signals` (3). One signal yields
`WEAK`, never a family.

The three ketoreductase families are kept apart as **three separate mechanistic
hypotheses**, because they share no catalytic residues, no fold and no
cofactor-recognition logic:

| Family | Fold | Catalytic machinery | Metal |
| --- | --- | --- | --- |
| SDR | Rossmann | Ser–Tyr–Lys(–Asn) | none |
| MDR/ADH | two-domain, different Rossmann insertion | metal-polarised carbonyl, proton relay | catalytic Zn |
| AKR | (β/α)₈ barrel, no Rossmann motif | Asp–Tyr–Lys–His tetrad | none |

There is deliberately no shared default anywhere in that module: every catalytic
mapping and cofactor rule is reachable only through a `FamilyHypothesis` that
binds one family template, one catalytic template of the **same** family, and one
reference sequence whose residues at the template's catalytic positions are
verified on construction. Applying one family's rule to another is a
construction error, not a runtime possibility.

The sequence-similarity network is an exploration aid. A connected component at a
chosen threshold is a set of sequences linked by pairwise similarity above that
threshold — not a clade, not monophyletic, no implication of shared substrate,
and it changes shape when the threshold changes.

**5. `prepare_structures` — pick a structure, and say why that one.**
The priority order is explicit and every choice carries a reason string written
into the QC table:

1. a matching experimental complex (this protein, with the cofactor the template
   demands, in the state it demands);
2. a usable experimental structure of the enzyme (apo, or missing part of the
   catalytic system);
3. a reusable predicted structure already on disk;
4. a new prediction (costs money and GPU time, so it is late);
5. an experimental structure of a **homologue** — ranked *below* a predicted
   model of the actual candidate. People get this backwards: a 2.0 Å crystal
   structure of a 45 %-identical relative is a beautiful picture of a different
   protein, and its pocket residues are literally not the residues that will be
   mutated.

Two traps are handled explicitly. **The crystallographic NAD(P)⁺ trap**: a large
fraction of deposited ketoreductase structures carry the oxidised cofactor
(`NAD`, `NAP`); the reduced forms are different chemical components (`NAI`,
`NDP`). An oxidised nicotinamide has no hydride to donate, so a "hydride transfer
distance" measured from its C4 is a perfectly reasonable number describing
nothing. The step raises a **blocker** and does not relabel the ligand. **Mean
pLDDT hides a disordered active site**: the mean is dominated by the
well-predicted core, and a model at mean 92 can have a pocket at 55, so both are
recorded and `StructureRecord.pocket_confidence` reads the pocket-local one.
Numbering is rebuilt by alignment for every structure, never assumed.

**6. `model_complexes` — build the whole catalytic system, twice.**
A complex missing the cofactor is not a model of the reaction: hydride comes from
the nicotinamide C4 of NAD(P)H, and without it the substrate relaxes into the
space the cofactor should occupy. `validate_assembly` checks the requested
assembly component by component against the family's catalytic template —
assembly state, substrate, cofactor *in the required oxidation state*, required
metals — and a missing or mis-oxidised component is a blocker. An oxidised NAD⁺
in the pocket is treated as a **missing** cofactor, because mechanistically that
is what it is.

Two routes, because they fail differently: template-guided pocket docking
inherits every error in a fixed experimental receptor (including a side-chain
rotamer refined against a different ligand); joint protein–ligand–cofactor
prediction builds everything at once and can close a pocket around a substrate
that does not belong there. Agreement between them is worth something precisely
because their artefacts are unrelated.

All major poses are kept. Retaining the single best-scoring pose is how a
modelling artefact becomes a conclusion: the stereochemical call then rests on a
sample of one. Poses are grouped by heavy-atom RMSD so "six poses" and "six
copies of one pose" can be told apart, but nothing is discarded. Every pose
records the restraints it was built under, and
`assert_restraints_recorded` raises rather than emitting a pose whose restraints
were forgotten.

**7. `evaluate_catalysis` — a mechanism-shaped question, never a score.**
It replaces "dock the substrate, keep the good scores under 10 Å". A docking
score is a pseudo-energy belonging to one scoring function that ranks pocket
volume and heavy-atom count at least as strongly as catalysis. A bare distance
cutoff is usually measured between the wrong things — substrate centroid to
protein centroid, or a ligand atom to a Cα. Catalysis is a statement about the
atoms the chemistry touches: for an NAD(P)H-dependent carbonyl reduction, the
nicotinamide C4 and the carbonyl carbon, with the carbonyl oxygen engaged by the
family's stabilising residues and the donor approaching from the face that gives
the wanted enantiomer.

Five pose outcomes, and the three that a naive pipeline collapses stay apart:

| `PoseOutcome` | Means | Counts toward robustness? |
| --- | --- | --- |
| `MECHANISM_SATISFIED` | measured, inside a trusted window | yes |
| `MECHANISM_VIOLATED` | measured, outside a **calibrated** window — a computational negative, still not an experimental result | yes |
| `OUTSIDE_UNCALIBRATED_WINDOW` | failed a window nobody fitted — leaves `gating_passed` at `None` (undecided, not failed) | **no** |
| `NOT_MEASURABLE` | the pose could not be built, parsed or measured; an atom role was never bound; the cofactor is absent — **nothing was tested** | no |
| `INPUT_ERROR` | wrong cofactor, wrong oxidation state — a defect to repair, not a property of the enzyme | no |

**8. `select_batch` — see §4 and §5.**

**9. `ingest_results` — see §7.**

**10. `propose_mutations` — three evidence classes, and proximity is not one.**
A 4–8 Å shell around the substrate is a **search scope**: where a mutation
*could* matter. It is not a claim that every residue in it is worth changing — a
typical shell holds 25–40 residues, most of which are scaffold, and saturating
all of them spends a plate learning that a protein tolerates conservative
substitutions. `SiteEvidence` keeps three classes apart (structural, family,
experimental) and sites are ranked on how many agree. A site supported only by
shell membership is still emitted — it is a lead — but marked proximity-only,
ranked last, and carries the measured distance so a reviewer can see how thin the
justification is. A mutant residue identity may only come from a stated source:
an experimental precedent naming the substitution, a residue actually observed at
that position in a subfamily with the wanted substrate range, or a smaller
residue **only where a steric clash was measured**.

Every proposal states, before testing, what it is expected to improve and what it
may cost, on three axes kept deliberately apart — substrate fit, catalytic
function, stability/expression risk. `MutationProposal` refuses a proposal with
no `possible_cost`, and refuses a combination variant with no single-mutant
decomposition controls, because an improvement from a double mutant with no
singles is unattributable.

---

## 4. Batch composition by role

Taking the top 96 maximises expected hits **only if the ranking is calibrated**.
It is not: most axes are model outputs with unquantified error, and — worse — the
errors are *correlated*, since the same structure predictor, the same docking
function and the same family template act on every candidate. The top 96 are
usually 96 close relatives of whatever the models already understood, and a
systematic error produces 96 simultaneous failures that teach nothing about where
the error was.

So a round is composed of three populations with different jobs
(`science/diversity.compose_batch`):

| Role | Default share of 96 | Job |
| --- | --- | --- |
| `HIGH_EVIDENCE` | 48 | Most likely to work. Pays for the round. |
| `DIVERSITY` | 24 | Covers pocket space, so a whole-batch failure localises to something other than "we only tried one clade". |
| `UNCERTAINTY_PROBE` | 24 | Mechanistically sound candidates the models could not decide about. **The only wells that can tell you the model was wrong**, and the first thing a pure top-k selection cuts. |

48 / 24 / 24 is a stated starting configuration, not a validated optimum. No
experiment in this repository shows it beats 64/16/16 or 32/32/32; it encodes one
judgement — half the round pays for itself, half buys information — and it is
passed in as configuration precisely so a reviewer can argue with it and round 2
can revise it against round-1 outcomes.

Two further properties of the selection:

**Diversity is measured on the pocket, not on global identity.** Global identity
is dominated by the scaffold: two enzymes 45 % identical overall can have
identical substrate pockets, and two 85 % identical can differ at the three
positions that set substrate size. `pocket_distance` returns `None` rather than
`1.0` when a signature is unavailable, so a candidate nothing is known about does
not win every diversity slot by looking maximally different.

**Only gate-passing candidates enter any pool**, and a candidate whose gate could
not be *evaluated* is excluded and counted **separately** — "we did not check" is
not a reason to spend a construct and is also not a rejection. The shortfall
reason names them, which makes the cheapest way to lengthen the batch visible.

Round-2 variants enter as whole attributable groups: a double mutant drags its two
single-mutant controls into the same plate, or it does not go in at all.

---

## 5. 96 genes is not 96 wells

"96 constructs" is a synthesis order. The plate it implies is

```
candidate wells = constructs × cofactor conditions × replicates
control   wells = controls   × cofactor conditions × replicates
total wells     = candidate wells + control wells
plates          = ceil(total wells / wells per plate)
```

For 96 genes at 2 cofactor conditions in triplicate that is **576 candidate wells
before a single control** — six 96-well plates, not one. With four controls at
the same multiplicity, 24 control wells, 600 total, seven plates.
`MeasurementFootprint.describe()` prints exactly this, next to the order, and the
figure goes into the batch-authorisation payload so the operator approves the
plate they are actually buying. Discovering the multiplier after the order is
placed means either dropping replicates (and losing the ability to call anything)
or dropping conditions (and losing the cofactor question).

**Controls are genes too.** A positive-control enzyme has to be synthesised like
everything else. If the construct count is a hard cap and the controls were not
reserved up front, the batch is either 98 constructs (the cap was not a cap) or 94
chosen candidates plus 2 the selection never saw (the selection was not the
selection). `Budget.constructs_include_controls` makes that choice explicit and
`reserve_control_slots` takes the slots **first**. `BatchPlan` refuses a plan in
which a control needing a new gene does not occupy a batch slot.

---

## 6. The three experiment tiers

`configs/templates/assay/`. Each tier's `positive_criteria` ship with every
numeric bar `null`, on purpose: the bar depends on this plate reader, this lysate
preparation and the measured background, none of which exist yet, and a number
invented now would be a threshold that was never calibrated and that a later
reader would take for a validated cutoff. With the bars null, `ingest_results`
reports every well as **undecided** rather than as a pass — the loud failure that
forces the operator to pre-register the bar before the plate is read.

### Tier 1 — expression and indirect activity screen

Microplate screen on clarified lysate, following nicotinamide cofactor depletion
at 340 nm against the target ketone, with a soluble-expression readout per
construct in the same round.

`confirms_product_identity: false`, and `AssayTemplate` forbids that from tier 2
upward. **A tier-1 positive is a shortlisting signal and nothing else.** Lysate
contains many enzymes that consume NAD(P)H, the cofactor oxidises slowly on its
own, and an endogenous host reductase can reduce the ketone. No hit may be
reported outside the project on tier-1 data alone.

Controls required: `no_enzyme`, `empty_vector`, `no_cofactor`, `no_substrate`,
`positive_enzyme`, `heat_inactivated_lysate`.

### Tier 2 — product identity confirmation

LC-MS or GC-MS of the quenched reaction against an authentic standard: retention
time and mass match, **with co-injection** of the standard into a positive sample,
so that a shared retention time is demonstrated rather than inferred.

This is the first point at which "this enzyme reduces our ketone to our alcohol"
can be said at all. The tier is deliberately `chiral_capable: false`: a standard
reversed-phase or non-chiral GC method cannot separate enantiomers, so it
confirms **constitution, not configuration**, and reporting an ee from it is the
specific error the tier is designed to prevent.

Controls required: authentic standard injection, co-injection spike,
substrate-only injection, `no_enzyme`, `empty_vector`, solvent blank,
`no_cofactor` (repeated here, because the detection chain has changed).

### Tier 3 — quantitative characterisation with configuration

Chiral GC or chiral HPLC with a **validated** separation: baseline resolution
demonstrated on a racemic standard, and each peak assigned to a configuration
with authentic single-enantiomer standards. Elution order is not assumed from
another column, another temperature programme or another paper. Conversion is
quantified against a calibration curve.

**Kinetic constants are measured, not back-calculated from a single endpoint.**
Where the programme justifies them, kcat and Km come from an initial-rate series
on purified enzyme under stated conditions. A single endpoint conversion yields
neither, and the record schema keeps `measurement_type` explicit so an initial
rate, a conversion, a specific activity, a growth readout and a binding readout
are never pooled onto one numeric scale. The template's own note adds that
kinetic parameters belong to the conditions they were measured under and are not
comparable across lysate and purified enzyme, across cosolvent fractions, or
across recycling systems.

Controls required: racemic standard, single-enantiomer standards, calibration
series, substrate-only injection, `no_enzyme`, `empty_vector`, `no_cofactor`, and
a **racemisation control** — standard product incubated under the reaction and
work-up conditions, to show the measured ee is not an artefact of the work-up.

### What a control is entitled to prove

The most expensive confusion in a screening round is between

> *"the assay system works"*  and  *"the target substrate is turned over"*

A positive-control enzyme acting on its own known substrate demonstrates the
first and says **nothing** about the second — it is there so that a plate of
negatives can be attributed to the enzymes rather than to a dead reagent.
`ControlClaim` makes these different enum values, and `validate_control_claims`
refuses a plan in which a background or system control has been written up as
evidence of target turnover, refuses a target-turnover claim with no
product-identifying method behind it, and refuses a plan with no control claiming
that the assay system works at all.

In round 1, **none** of the default four controls carries
`TARGET_SUBSTRATE_TURNED_OVER`. Nothing is known to turn the target substrate
over — that is the question the round asks — and a control claiming it would be
assuming the answer.

| Default control | Claim | Answers |
| --- | --- | --- |
| `no_enzyme` | `BACKGROUND_WITHOUT_ENZYME` | abiotic and reagent background, including slow non-enzymatic reduction |
| `empty_vector` | `BACKGROUND_FROM_HOST` | *E. coli* lysate carries endogenous reductases |
| `no_cofactor` | `COFACTOR_DEPENDENCE` | separates cofactor-driven chemistry from a cofactor-independent artefact |
| `positive_enzyme` | `ASSAY_SYSTEM_WORKS` | the detection chain is alive end to end — omitted when no such enzyme exists, and then the shortfall is **reported** rather than the claim being quietly transferred to another control |

---

## 7. Signed ee, and reading the plate back

**ee is signed toward the target enantiomer.** `ee_target(n_target, n_opposite)`
returns `(n_target − n_opposite) / (n_target + n_opposite) × 100`, so a
beautifully selective failure reports **−94 %**, not "94 % ee". An absolute ee
would let a batch, half of which made the wrong enantiomer, average into a
success. The tier-3 template therefore requires *both* peak areas in the returned
plate: a single `ee` column with no peak areas cannot be audited and cannot
distinguish 98 % of the wanted enantiomer from 98 % of the unwanted one.

ee is also not always defined. If the substrate is an aldehyde or a symmetric
ketone the product has no stereocentre, and an "ee" from an achiral product is
integrated noise. `TaskSpec.stereo_task` — which requires
`creates_new_stereocenter is True` **and** a target configuration — is the switch,
and a task where it is false must not be scored on the ee bar at all.

`ingest_results` maps a returned plate into every branch of `OutcomeClass`
against the criterion that `select_batch` wrote into `experiment_plan.yaml`
before the plate ran. `criterion_override` is accepted as an argument,
**refused**, and recorded as a protocol deviation in the round summary and in
provenance — writing the override down is the point, because an endpoint that
moves silently is unfalsifiable and one that moves on the record is at least
arguable. An unrecognised key in `positive_criteria` raises rather than being
ignored, because a criterion with a typo in it silently passes everything.

Two row-level refusals: a row that fails the criterion with **no detection limit**
(and no pre-registered limit to fall back on) is held as unresolved rather than
written down as a negative; a row claiming a positive on an **indirect signal
alone** is surfaced as a blocker and parked in `pending_confirmation.jsonl` — not
converted into a negative, which would be the opposite error.

The active-learning update partitions every record and then **checks the
arithmetic**: the partitions must sum to the input count or `FabricationGuardError`
is raised. Expression failures are separated from catalytic negatives rather than
discarded — they are training data for the expression-risk model and are silent
about chemistry.

---

## 8. A round with no hits is a result

When nothing turns over, `diagnose_no_hits()` produces a **five-way differential**,
not a verdict. Each hypothesis gets `consistent` / `unlikely` / `undetermined`
from evidence that is actually on the plate, plus the one experiment that would
separate it from the others. Nothing is ranked — ranking them would imply a prior
nobody has.

| Hypothesis | The question | Discriminating move |
| --- | --- | --- |
| `SEARCH_SCOPE_WRONG` | did we look in the wrong families, or too narrow a slice of them? | widen mining to the families the round-1 quota excluded, and include a positive control from a family known to reduce a similar ketone |
| `COFACTOR_MISMATCH` | were the enzymes offered a cofactor they do not use? | re-run the same constructs under the other nicotinamide condition |
| `EXPRESSION_FAILURE` | did the constructs ever become soluble protein? | re-express the non-expressers (host, temperature, tag, chaperones) before concluding anything about catalysis |
| `DETECTION_CONDITIONS_UNSUITABLE` | would the assay have seen the product if it had been there? | spike the authentic standard into the matrix and establish the limit of detection |
| `NO_NATURAL_CATALYST` | does this substrate simply have no natural catalyst? | **never returned as `consistent`** |

The last one deserves its own sentence. One round, over one sampled slice of
sequence space, with one assay, cannot support "there is no natural catalyst",
and the moment it is written down as the answer the campaign stops. It is
reported as `undetermined` with the scope that would be needed even to start
arguing for it.

The round then feeds the next one through `next_round_quotas()`, by an **ordinal,
stated** rule — one round produces nothing like enough data to fit anything:

- **expand** — the family produced a confirmed hit, *or* a wrong-configuration
  product. The second is often the better engineering target of the two: a
  working scaffold with a fixable problem.
- **re-express** — the family's records are mostly expression failures. Its quota
  is **not** reduced; nothing was learnt about its catalysis, and cutting it would
  record a protein-production problem as a biological conclusion.
- **reduce** — catalytic negatives at stated limits and no hits. Reduced, but
  never to zero: a floor of 2 keeps the boundary sampled, because a family cut to
  nothing can never overturn one unlucky round.
- **untested** — unchanged, because no information arrived.

A Wilson interval on each family's hit rate is attached, so a 0 / 4 family is not
read as a demonstrated zero.

Finally, `LayerUpdate` reports separately which of three layers actually moved —
the data layer (records, limits, expression, product identity), the model layer
(specificity, expression risk, mutation effects, uncertainty) and the decision
layer (next-round quotas, exploration scope, selection) — each with an explicit
`changed` flag. A round that adds records without moving any model is a real and
common outcome, and saying so is more useful than implying the system learned.

---

## 中文摘要

### 三个层次，按风险排序

**第一层：天然酶挖掘（第一个版本）。** 必须先跑通这一层，理由是诊断性的而非保守：
**只有一个被实验确认的天然亲本，才能把"家族就找错了"和"骨架能用但口袋形状不对"分开。**
这两种情况需要相反的下一步。

**第二层：底物导向改造（第二阶段）。** `propose_mutations` 刻意排在第十位。它从
**确认亲本的实验记录**出发；要在未确认的候选上跑，必须由操作者留下带理由的
`ParentOverride`，而这条记录会被挂进每一条产出提案的 `contradicting_evidence`，
一路跟进板图。第一轮改造冻结催化与辅因子锚定位点：同一轮里既改催化残基又改口袋残基，
得到的是一个死掉的变体和零信息——能归因这次损失的单点对照并不存在。冻结是**强制**的：
亲本的催化映射不完整导致冻结**无法核验**时，该亲本被直接跳过并记 blocker，而不是硬着头皮
去设计。

**第三层：从头设计（后置的高风险分支，单独评估）。** 它的成功率与证据标准都是另一个量级，
塞进同一个漏斗会让一个极低的期望命中率被平均进流水线的总指标。它被放在 `docs/ROADMAP.md`
里，有自己的验收标准，**一行都没有实现**。

### 分级漏斗：这是资源计划，不是预设淘汰率

2000 → 600 → 600 → 300 → 96。这些数字说的是**每一级按多大规模配置算力、策展和 GPU 时间**，
不是"2000 条里会有 600 条通过家族 QC"。相反的读法会纵容两个坏动作：把 600 当成**比率**，
那么只产出 200 就像个"该靠放宽过滤器修掉"的失败（`mine_sequences` 把 `RetentionPolicy`
在入口指纹化、过滤后复查，中途被改会抛 `FabricationGuardError`）；把 96 当成比率，那么批次
不足就像 bug 而不是发现（`BatchPlan` 拒绝没有 `shortfall_reason` 的短批次，而且**没有任何
代码路径**能靠降低标准把批次凑长）。

**精细建模池必须明显大于构建体批次。** 如果两者一样大，"筛选"只是"池子"的另一个名字：
每个被建模的候选都会被下单，门没有东西可排除，多样性槽和不确定性槽没有可选的对象，这一轮
测的只是"建模预算刚好够到谁"。`Budget` 强制
`detailed_complex_target >= candidate_slots`，出厂比例是 **300 : 96**。诚实地说：校验器
允许恰好相等，而它的报错文案写的是"必须超过"——相等能通过校验，却仍然没给筛选留空间，所以
1:1 应当视为配置错误。

### 批次按角色构成

取"排名前 96"只在排序**已标定**时才最大化期望命中——而它没有标定，更糟的是它的误差是
**相关的**：同一个结构预测器、同一个打分函数、同一套家族模板作用在每个候选上。前 96 名通常
是模型本来就懂的那类东西的 96 个近亲，一个系统性错误会制造 96 个同时发生的失败，而且什么
也教不了你。

所以一轮由三群构成（默认 48 / 24 / 24，这是**写明的起始配置而不是验证过的最优解**）：
`HIGH_EVIDENCE`（最可能成，为这一轮买单）、`DIVERSITY`（覆盖口袋空间，使整批失败能定位到
"不只是只试了一个支系"之外的原因）、`UNCERTAINTY_PROBE`（机理上说得通但模型拿不准的候选，
**这是唯一能告诉你模型错了的孔**，也是纯 top-k 第一个砍掉的）。

多样性按**口袋**而不是全局同一性衡量：全局同一性被骨架主导，45% 同一的两个酶可能口袋完全
一样，85% 同一的两个酶可能恰好在决定底物大小的三个位置上不同。签名缺失时
`pocket_distance` 返回 `None` 而不是 `1.0`——不能让"一无所知"的候选靠"看起来最不一样"
赢走所有多样性槽。

### 96 个基因不是 96 个孔

孔数 = 构建体数 × 辅因子条件数 × 重复数。96 个基因、2 个辅因子条件、三复孔，就是
**576 个候选孔**，还没算对照——六块 96 孔板而不是一块。再加四个对照同样倍乘就是 24 个对照
孔、共 600 孔、七块板。这个数字会打印在订单旁边，并进入批次审批载荷。

**对照也是基因。** 阳性对照酶同样要合成。如果 96 是硬上限而对照没有预留，那么这批要么是
98 个构建体（上限不是上限），要么是 94 个被选中的候选加 2 个筛选从未看过的
（筛选不是筛选）。

### 三级实验与对照能证明什么

**一级**（340 nm 辅因子消耗 + 可溶表达读数）：`confirms_product_identity: false`，
**一级阳性只是一个入围信号**。裂解液里有很多消耗 NAD(P)H 的酶，辅因子自己也会慢慢氧化，
宿主内源还原酶也能还原这个酮。只有一级数据，不得对外报告任何命中。

**二级**（LC-MS/GC-MS 对标准品 + **共注射**）：第一次可以说"这个酶把我们的酮还原成了我们的
醇"。刻意设为 `chiral_capable: false`——常规反相或非手性 GC 分不开对映体，这一级确认的是
**结构式而不是构型**，从这一级报 ee 正是它要防的错误。

**三级**（手性 GC/HPLC，经过验证的分离）：外消旋标准品证明基线分离，单一对映体标准品把每个
峰指派到构型，洗脱顺序**不得**从另一根柱子、另一套程序或另一篇文章照搬。
**动力学常数是测出来的，不是从单点终点反算的**：kcat 与 Km 来自纯酶在写明条件下的初速度
序列。记录层的 `measurement_type` 保持显式，初速度、转化率、比活、生长读数、结合读数
**永不**并到同一个数值标度上。

**对照最贵的混淆**是"检测体系是活的"与"目标底物被转化了"。阳性对照酶作用在它自己的已知底物
上，只证明前者，对后者**一个字都没说**——它存在的意义是让一板阴性能归因于酶而不是死掉的试剂。
`ControlClaim` 把两者做成不同的枚举值，`validate_control_claims` 会拒绝把背景对照写成
"目标被转化"的方案。第一轮里四个默认对照**没有一个**声称目标底物被转化——那正是这一轮要问的
问题，声称它就是在假设答案。

### 有符号 ee

`ee_target` 朝目标对映体取符号：选择性极佳但方向错了，报的是 **−94 %** 而不是"94 % ee"。
绝对值会让"一半做出了错的对映体"的一批数据平均成一个成功。三级模板因此要求**两个峰面积**
都写进回板文件：只有一列 `ee` 无法审计，也分不清 98% 是想要的那个还是不想要的那个。
底物是醛或对称酮时产物没有立体中心，此时的 "ee" 是对噪声的积分；`TaskSpec.stereo_task`
是那个开关。

### 一轮没有命中，本身就是结果

`diagnose_no_hits()` 给出**五路鉴别诊断**而不是结论：检索范围错了 / 辅因子不匹配 /
表达失败 / 检测条件不适用 / 这个底物根本没有天然催化剂。每一条给出板上实际支持它的证据、
反对它的证据，以及**能把它和其它几条分开的那一个实验**；不排序——排序等于假装有一个谁也没有的
先验。

最后一条**永远不会被标成 `consistent`**。一轮实验、一片被采样到的序列空间、一种检测方法，
支撑不起"没有天然催化剂"这个结论，而它一旦被写成答案，整个项目就停了。

下一轮配额按**序数化的、写明的**规则更新：出了确认命中**或者出了构型错误的产物**→ 扩大
（后者常常是更好的改造靶点：能工作的骨架 + 一个可修的问题）；记录以表达失败为主 → 重新表达，
配额**不减**（那是蛋白生产问题，减配额等于把它记成生物学结论）；在写明检测限下的催化阴性且
无命中 → 缩减，但**不归零**（下限为 2，被砍到零的家族永远没机会推翻一次倒霉的实验）；
没测过 → 不变。每个家族的命中率都附 Wilson 区间，所以 0/4 不会被读成"已证明为零"。
