# Roadmap

Three milestones, then a branch. Each milestone has an acceptance criterion that
can fail, because a milestone nobody can fail is a description rather than a
plan.

**Where the project is today.** The harness is complete and tested: the whole suite
passes (2919 tests plus 696 subtests at the time of writing). **Four public
databases have been called live** (UniProtKB, RCSB PDB, Rhea, Zenodo) and one
real dataset has been downloaded through a checksum-verified route; a first
real reference set of KRED complexes (19 entries, 28 kinetic records) has been
stored, hash-pinned and audited, and cannot calibrate any shipped window. **No
structure predictor, docking program, inverse-folding model or search binary has
ever been run, and no plate has ever been read.** Milestone 1 is therefore
*built but not demonstrated*, and milestones 2 and 3 have not started because
they require a laboratory, not more code. §5 is the component-by-component
table, and §7 says what the latest round of work did and did not establish.

**What the repository can claim, in one sentence:** it has a sound machinery of
claiming, a sequence-to-annotated-class model that is better than a
nearest-neighbour baseline on a few classes and no better on most, and a
simulation showing that feeding labels back helps *on annotation-derived data*.
It does **not** support the sentence "the agent has learned from experiments and
validated new-enzyme discovery": no experiment has been run.

---

## Milestone 1 — the evidence layer and reproducible screening

**What it means.** A named substrate goes in; a batch of constructs comes out,
behind recorded human approvals, with every candidate traceable to the evidence
that justified it and the whole run recomputable from the manifest. Nothing
about this milestone requires an enzyme to work. It requires the *machinery of
claiming* to be sound.

**Built.** The typed data model, the six-layer data architecture, the ten
interfaces, the template library, the controller, the approval queue, the
independent verifier, the evaluation package, the CLI and the deliverables
bundle.

**Not yet demonstrated, and the gap is specific:**

- **Four data sources of 49 have been connectivity-tested.** UniProtKB,
  RCSB PDB, Rhea and Zenodo were called from this container by
  `eagent sources verify` and carry the record of those calls; the other 45
  carry `connectivity_verified: false`, and all 49 still carry
  `needs_curation: true`. SABIO-RK was tried and answered with its own 404 page
  on every documented path; that is recorded as an observation, not a verdict.
- **Four connectors can use their route.** A verified base URL is not a
  verified request: UniProtKB, RCSB PDB, Rhea and Zenodo have clients written
  against recorded probes and live responses (which found three real defects --
  see `docs/DATASOURCES.md`), while every other connector's client is generic
  and refuses with `RequestShapeNotVerifiedError` rather than calling a URL
  nobody checked it against.
- **No search binary, structure predictor, docking program or inverse-folding
  model is installed.** Adapters now exist for MMseqs2 clustering, gnina and
  Boltz, and were tested against fakes that write output in the documented
  shape. **They have never been run against the real binaries.**
- **Licences: 4 of 56 facets were read, 4 are reported, 48 are unknown.** Read:
  Boltz code and weights (MIT, per the 2.2.1 release's LICENSE and README),
  AutoDock Vina code (Apache-2.0), the gnina licence statement (dual GPL /
  Apache, per its README; commercial use left unknown). Reported to the project
  but not read: VenusMine (CC BY-NC-ND 4.0, recorded as a restriction on
  commercial use and on derivative works) and AP Novo (Apache-2.0 code recorded
  with no permission claimed; weights and outputs recorded as non-commercial).
  Every entry still has `needs_legal_review: true`, and an unknown commercial
  permission **blocks** a commercial run.
- **No geometry window is calibrated.** 17 of 17. A mechanism turns a reference
  set into a tamper-evident calibration record, and a first real reference set
  (19 experimental entries, 28 kinetic records) is now in the repository,
  hash-pinned and audited -- but against the shipped NADPH-SDR template only
  one entry is eligible, with no known inactives, and a modest claim needs 14
  independent actives. No shipped window can pass it. See the reference-set
  subsection of §7.

**Acceptance criteria.** All five must hold, and each can fail:

1. A run on the pilot task reaches `select_batch` from a populated connector
   cache, with the three gates cleared by named actors, and produces an order
   form and an `experiment_plan.yaml` whose positive criterion is non-null.
2. `eagent bundle` reports **complete: YES** — all 18 standard items present,
   none partial — and `eagent bundle-verify` passes on a different machine.
3. The independent verifier runs over real candidates and real claims and
   returns a report with zero blockers, or with blockers that name real defects
   a curator then fixes.
4. The same run, re-executed from the same manifest, produces the same batch.
   Not "a similar batch".
5. At least one catalytic template carries a window with a non-empty
   `calibrated_on`, so the library can state what it was fitted on rather than
   only what it is uncalibrated against.

**What would falsify the milestone.** A bundle that reports complete while an
item is missing; a batch that differs between two runs of the same manifest; a
verifier that passes on a candidate whose structure is a different protein; a
window promoted to `gating` with `calibrated_on: []`.

---

## Milestone 2 — the natural-enzyme discovery loop, closed with real experimental feedback

**What it means.** A plate comes back and the system reads it into typed
records, classifies against the criterion registered *before* the plate ran,
reports both hit-rate denominators with their intervals, and produces either a
confirmed parent or a five-way no-hit differential. The loop is **closed** when
the output of a round changes the composition of the next one through
`next_round_quotas`, and the change is attributable.

**Built.** `select_batch` (plate map, measurement footprint, pre-registered
criterion, empty results template), `ingest_results` (all seven outcome classes,
the active-learning partition with its arithmetic checked, the layer update, the
no-hit diagnosis), `eval/metrics` (the moving-endpoint guard, both denominators,
signed ee), `datalayer/house_db` (the prediction freeze, whole batches including
failures, both denominators, the coverage audit).

**Not started.** Nothing has been ordered, expressed or assayed.

**Acceptance criteria:**

1. A round of ≥ 48 constructs is **submitted**, and every submitted construct —
   including the ones that never expressed and the ones never measured — has a
   row in the house database. A missing row is a failed criterion, not an
   omission.
2. The returned plate parses into `ExperimentRecord`s **without** a criterion
   override, and every negative carries its detection limit.
3. At least one `CONFIRMED_TARGET_PRODUCT` record exists, confirmed at tier 2 or
   above by a method that identifies the product. **Or**: no hit, and the
   five-way differential names a discriminating experiment for each hypothesis
   that is not `unlikely` — which is a pass, not a failure, provided the
   controls show the assay system worked.
4. `precision_at_k` runs against the registration written before the plate, with
   `criterion_in_use` supplied and matching. Both hit-rate denominators are
   reported with Wilson intervals.
5. Round 2's family quotas differ from round 1's, and the diff is explained by
   `next_round_quotas`' four verdicts rather than by a judgement call.

**What would falsify it.** A hit reported on tier-1 data alone. A negative with
no detection limit. A quota change nobody can attribute. An expression-failure
family whose quota was cut — that records a protein-production problem as a
biological conclusion.

**The honest risk.** The most likely outcome of round 1 is no confirmed hit.
That is why §8 of [`PROTOCOL.md`](PROTOCOL.md) exists and why
`NO_NATURAL_CATALYST` is never returned as `consistent`: a milestone whose only
pass condition is "it worked" would push the campaign to report one.

---

## Milestone 3 — the substrate-directed engineering loop, variants against parents under identical conditions

**What it means.** Starting from a parent an experiment confirmed, propose
variants with per-axis expectations stated before testing, build them, and
compare each against its parent **under one condition set**. The comparison is
the milestone; the improvement is a hoped-for outcome.

**Built.** `propose_mutations` (three evidence classes, frozen catalytic roles
enforced, mutant identity only from a stated source, combinations dragging their
single-mutant controls), `MutationProposal` (both numbering systems,
`intended_improvement` **and** `possible_cost`, per-axis expectations),
`select_batch` variant-group admission, `eval/metrics.variant_versus_parent` and
`round_two_report`, `house_db` mutation lineage as a first-class relation.

**Not started.** No variant has been designed from a real parent, and no
inverse-folding model is installed.

**Acceptance criteria:**

1. Every variant in the round has its parent **in the same plate, under the same
   conditions**. `variant_versus_parent` raising `ConditionMismatchError` is the
   mechanism; a round that triggers it has not met the criterion.
2. Every combination variant has its single mutants in the same plate.
   Unattributable improvements do not count.
3. Each variant's result is reported on the **three axes separately** —
   substrate fit, catalytic function, stability/expression — against the
   expectation recorded before testing. A variant that gained activity and lost
   expression is reported as both, and no code path combines them.
4. The frozen-role check passed for every proposal: no catalytic or
   cofactor-anchoring residue was mutated in round 1.
5. At least one pre-stated axis expectation is **refuted** by the data, and the
   refutation is recorded. A round in which every expectation was confirmed is
   more likely to indicate that the expectations were written vaguely than that
   the design was excellent.

**What would falsify it.** A parent-variant comparison across different pH,
cofactor state, substrate loading or endpoint. A combined "mutation quality"
number appearing anywhere — the house database raises `CollapsedScoreError`
rather than storing one.

**Dependency.** Milestone 3 cannot start before milestone 2 produces a confirmed
parent, and running it earlier under a `ParentOverride` buys a library whose
weakness travels with it. `datalayer/plan.answer_stage2_gate()` makes the same
question computable for the data layer: is there experimental data matching this
parent family and substrate space, and would using it be genuine extrapolation
rather than re-testing near neighbours of the training set?

---

## 4. The de novo branch — after the three, evaluated separately

Designing a catalyst for this reaction from scratch is **not** milestone 4. It is
a branch with a different risk class, a different success rate and a different
evidence standard, and it is positioned after the three milestones for one
structural reason: folding it into the same funnel would let a very low expected
hit rate be averaged into the pipeline's headline number, and a round that
mixed designed and mined candidates could not attribute its failure to either.

**Status: not started.** No module implements it. `rfdiffusion2` appears in
`configs/tool_registry.yaml` only so that its four licence facets exist to be
filled in; nothing calls it. `datalayer/plan.STAGE_3` covers *novelty and
generality* — reaching outside characterised families, under an explicit budget
— which is adjacent to but not the same as de novo design.

**Entry conditions.** All three must hold before the branch is opened:

1. Milestone 2 passed: at least one confirmed natural hit, so there is a
   positive control for the assay chain and a reference the designs are compared
   against.
2. Milestone 3 passed: the engineering loop demonstrated, so a design that
   needs optimisation has somewhere to go.
3. The target is one where mining and engineering have been shown **not** to
   suffice — otherwise the branch is being opened because it is interesting
   rather than because it is needed.

**Acceptance criteria, stated honestly.** The expected hit rate for a de novo
catalytic design round is low, and criteria that assume otherwise would push the
round to be reported dishonestly:

1. **Budget containment.** A de novo set never consumes a whole round.
   `datalayer/plan.check_novelty_budget()` already encodes the analogous rule
   for distant sequences without functional evidence: they may trickle in early
   and must never take an entire experimental round. The de novo branch inherits
   it.
2. **Designs and mined candidates are reported separately**, with separate
   denominators. Pooling them lets a good mining round carry a bad design round.
3. **The primary endpoint is unchanged**: a confirmed target product by a
   product-identifying method. A design that binds the substrate, or folds as
   predicted, has not met it — those are supporting observations.
4. **Zero confirmed hits is an acceptable, publishable outcome** of a first de
   novo round, provided the assay system control worked and the detection limit
   is recorded. The alternative is a branch that can only be reported when it
   succeeds.
5. **A design is never promoted on model confidence.** The same rule as
   everywhere else: a high ipTM is a statement about the prediction, not about
   catalysis, and `COMPUTATIONAL_CONSTRUCT` is where a designed sequence's
   evidence strength starts.
6. **Licence and disclosure cleared first.** A designed sequence is unpublished
   material; `AccessPolicy` refuses to submit one to an external service without
   a named human authorisation covering that exact sequence hash.

---

## 5. Current status, component by component

Three states only:

- **implemented** — code exists, is tested, and runs to completion in this
  environment.
- **seam** — contract, validation, provenance, artifact layout and refusal path
  implemented and tested; the external program or data is absent, so the step
  reports `tool_unavailable` (or a structured cache miss) and produces nothing.
- **not started** — no code.

### Foundation and data model

| Component | State | Note |
| --- | --- | --- |
| `errors.py`, `envelope.py`, `provenance.py`, `context.py` | implemented | typed failures, uniform envelope, run manifest, run context |
| `schemas/chem.py`, `reaction.py`, `record.py`, `candidate.py`, `variant.py`, `batch.py`, `templates.py` | implemented | pydantic v2, validators exercised by the suite |

### Deterministic science core (pure Python; numpy/scipy/rdkit/biopython absent)

| Component | State | Note |
| --- | --- | --- |
| `science/structure_io.py` | implemented | mmCIF and PDB reader/writer; anything it cannot interpret is an error, never a skipped line |
| `science/numbering.py` | implemented | Gotoh affine-gap alignment, residue maps, unobserved regions, wild-type verification |
| `science/family_numbering.py` | implemented, **reference data not shipped** | cross-subfamily position equivalence through a sourced family reference, validated against conserved anchors; refuses across families, below the identity floor, and into a gap. The machinery ships; **a curator must supply each scheme's reference sequence and where it came from**, because a fabricated reference would silently shift every position derived from it |
| `science/geometry.py` | implemented | distance, angle, dihedral, centroid, clash screen; measurement only, no thresholds |
| `science/stereo.py` | implemented | signed face calls; **does not perceive CIP priorities** and does not estimate ee |
| `science/robustness.py` | implemented | pose robustness, Wilson interval, circularity guard, cross-method agreement |
| `science/scorecard.py` | implemented | gates, ordinal levels, Pareto front, lexicographic rank, `refuse_linear_blend` |
| `science/diversity.py` | implemented | pocket signatures, submodular selection, family quotas, control reservation |

### The ten interfaces

| Interface | State | What is missing |
| --- | --- | --- |
| `normalize_reaction` | implemented | runs fully offline; rdkit stereocentre perception would be a *proposal* only, and rdkit is absent |
| `retrieve_evidence` | implemented, **data seam** | all connectors are cache-first; with an empty cache it reports 12 query gaps and 0 records |
| `mine_sequences` | implemented, **tool seam** | `blastp`, `mmseqs2`, `hmmsearch` absent; adapters drive injectable runners |
| `annotate_family` | implemented | needs seed accessions and references the templates do not yet carry |
| `prepare_structures` | implemented, **tool seam** | `StructurePredictor` absent (`UnavailablePredictor`); needs a local structure index |
| `model_complexes` | implemented, **tool seam** | `DockingRunner` and `ComplexPredictor` absent |
| `evaluate_catalysis` | implemented | runs; every verdict is caveated because all 17 windows are uncalibrated |
| `select_batch` | implemented | opens no socket; `submit_to` refused outright |
| `ingest_results` | implemented | has never read a real plate |
| `propose_mutations` | implemented, **tool seam** | LigandMPNN absent (`MissingLigandMPNN`); rational and literature-precedent paths work without it |

### Data layer

| Component | State | Note |
| --- | --- | --- |
| `datalayer/layers.py` | implemented | six layers, closed set of joins, similarity joins refused by name |
| `datalayer/registry.py` | implemented | 49 sources loaded and queryable |
| `datalayer/identity.py` | implemented | identity ladder, cofactor identity, merge policy enforced at the union point |
| `datalayer/intake.py` | implemented | four tiers, human-only promotion, outcome normalisation, direction check |
| `datalayer/lineage.py` | implemented | independent-evidence counting, leakage-safe groups |
| `datalayer/snapshot.py` | implemented | freeze, verify, diff, exclusion log |
| `datalayer/preconditions.py` | implemented | the five-step gate chain |
| `datalayer/house_db.py` | implemented | sqlite3; four storage-layer refusals |
| `datalayer/plan.py` | implemented | three rollout stages as typed, validated work packages |
| **Source connectivity** | **4 of 49** | UniProtKB, RCSB PDB, Rhea and Zenodo verified by `eagent sources verify` with the calls recorded in `connectivity.observed.yaml`; all four have clients checked against their probes and recorded live responses; SABIO-RK observed unreachable; 23 record their lineage as admittedly incomplete |
| `connectors/base.py` | implemented | cache-first contract, disclosure guard, evidence ceilings |
| Per-resource connectors (UniProt, PDB, Rhea, Zenodo) | implemented, **live-checked** | one each; the rest are still `OfflineConnector` |

### Harness

| Component | State | Note |
| --- | --- | --- |
| `harness/templates.py` | implemented | strict loading, calibration report, window authority, integrity problems |
| `harness/registry.py` | implemented | strict ten-interface registry, dependency problems, manifest-ready report |
| `harness/llm.py` | implemented | numeric guard (value-to-cell binding, verified against the run's own artifacts), restricted turn shape, `EchoClient` on the tested path, `AnthropicMessagesClient` **written from the API reference and never sent to the live service** |
| `harness/planner.py` | implemented | puts a model behind the controller's `widen` hook and two observer hooks; applies only search terms for `retrieve_evidence` (allowlist), records everything else as unapproved proposals; audited; never called for a remote client while the network is off. **Never run against a real model** |
| `harness/citation.py` | implemented | citation grammar, artifact index from the manifest, per-cell verification |
| `harness/approval.py` | implemented | three gates, named actors, persisted queue, hard batch block |
| `harness/controller.py` | implemented | declared state machine, seven failure kinds, resume by input digest |
| `harness/verifier.py` | implemented | nine checks, re-derived from primary material |

### Evaluation, deliverables, interface

| Component | State | Note |
| --- | --- | --- |
| `eval/splits.py` | implemented | three regimes, grouped splitting, six-category audit |
| `eval/metrics.py` | implemented | pre-registration guard, both denominators, signed ee, round-two report |
| `eval/baselines.py` | implemented, **1 seam** | `homology_multi_seed`, `docking_score_ranking`, `full_agent` run; `substrate_specificity_model` accepts `science/enzyme_substrate.candidate_scorer`; `family_function_prediction` reports unavailable by name |
| `eval/retrospective.py` | implemented | label recovery on the SDR deposit, grouped folds, leakage priced; **annotation-derived labels** |
| `eval/feedback_simulation.py` | implemented | feedback on/off over identical replicates; **annotation-derived labels revealed as if assayed** |
| `science/enzyme_substrate.py` | implemented | spectrum-kernel heads, probabilities only where a grouped out-of-fold calibration shows skill |
| `science/calibration.py`, `tools/calibrate_windows.py` | implemented | Wilks tolerance-interval calibration records, verified by digest; run on a real reference set, which cannot calibrate any shipped window (§7) |
| `eval/kred_reference.py`, `kred_workbook.py`, `kred_coordinates.py`, `kred_complexes.py`, `kred_sources.py` | implemented | the KRED reference set: converted, hash-pinned, recomputed, cross-checked, audited against the templates, and compared with its sources; `eagent reference` |
| `tools/prediction_backends.py` | implemented, **never run on a real binary** | gnina (route A) and Boltz (route B) adapters |
| `tools/open_branches.py` | implemented as **refusing seams** | open function discovery and de novo design; no generator installed |
| `eval/ablations.py` | implemented | four ablations; `active_learning` reports not-evaluable without a `PriorRound` |
| `deliverables/bundle.py` | implemented | 18 declared items, derived-file validation, hash verification |
| `cli.py` | implemented | 10 commands, 5 exit codes |

### Configuration and curation

| Artefact | State | Note |
| --- | --- | --- |
| `configs/tasks/KRED_PILOT_001.yaml` | implemented | ships with all four gate-1 fields null, deliberately |
| Reaction template (1) | implemented, **needs curation** | SMARTS never parsed in this environment; no RHEA id verified |
| Family templates (3: SDR, AKR, MDR/ADH) | implemented, **needs curation** | `seed_accessions` and `interpro_ids` empty; no accession verified |
| Catalytic templates (3) | implemented, **needs curation** | `reference_structures` empty in all three; **17 of 17 windows uncalibrated**, 0 gating; a pinned reference set exists but qualifies one entry for one template |
| `configs/references/kred_calibration/v0.1` | implemented, **bindings unreviewed** | 19 entries, 28 kinetic records, 10 er; 25/28 records match their paper's table, 1 matches BRENDA only, 2 partly unread; coordinates pinned (not committed); protein roles unbound |
| Engineering templates (3) | implemented, **needs curation** | SDR, AKR and MDR/ADH all present; windows uncalibrated as above |
| Family numbering schemes | **not started** | `science/family_numbering.py` and `science/pocket.py` ship the machinery; no sourced reference sequence is bundled, so pocket signatures fall back to composition and say so |
| Assay templates (3 tiers) | implemented, **needs curation** | every numeric bar `null`; limits of detection must be measured on site |
| `configs/tool_registry.yaml` | implemented, **needs legal review** | 14 tools × 4 facets = 56 entries; 4 licences read, 4 reported, 48 unknown (see Milestone 1); an unknown commercial permission blocks a commercial run |
| `configs/datasources/*.yaml` | implemented, **4 of 49 tested** | see the data-layer table above |
| `datalayer/probe.py` | implemented | 10 shipped probes (4 of them RCSB: three data-API shapes and the file-download shape), marker checks, the observation file |
| De novo design | **seam only** | `tools/open_branches.py` refuses until licences, a confirmed reaction and a named approver are in place; no generator is installed |

### Known wart

`eagent/tools/__init__.py` carries a planning-era `INTERFACE_MODULES` table
naming six interfaces that no module implements (`map_catalytic_site`,
`build_complex`, `screen_geometry`, `rank_candidates`, `design_variants`,
`plan_batch`). Discovery is authoritative and finds the ten real interfaces, and
the controller's path (`build_interface_registry`, which names `PROTOCOL_ORDER`
explicitly) reports nothing missing — but a bare `build_registry()` attaches
those six to `missing_interfaces`. Dead weight to remove, not a live gap.

---

## 6. The shortest path to milestone 1

In the order that unblocks the most:

1. **Resolve the pilot task's four gate-1 fields** with an operator authority —
   substrate SMILES, product SMILES, `creates_new_stereocenter`, the atom-mapped
   reaction SMILES. Everything downstream is about a molecule until these exist.
2. **Populate the connector cache** for the twelve queries `retrieve_evidence`
   lists in `evidence_gaps.tsv`, each through
   `OfflineConnector.store_import(...)`. This is a curation task, not a coding
   task.
3. **Install one search binary** and point `mine_sequences` at a local sequence
   database file with a recorded snapshot date.
4. **Deposit mmCIF structures** into a local structure index, or install a
   predictor and record its version and licence facets.
5. **Calibrate one window.** Pick the SDR hydride-transfer distance, supply
   experimental ternary complexes, fit the upper bound, list them in
   `calibrated_on`, and only then consider promoting the severity.
6. **Read the licences** for whichever tools are actually installed, and record
   each with its `license_source` and the version it attaches to.
7. **Write the AKR and MDR/ADH engineering templates**, or accept that variants
   can be proposed only for SDR parents.

---

## 7. What the 2026-10 round established, and what it did not

The round followed a review of seven papers and competing agents whose verdict
was: keep the framework, and put the next phase into **fixing code defects that
change scientific conclusions, running a real KRED task, and building
enzyme-substrate prediction with experimental feedback.** This section is the
account of what that did and did not deliver. The headline is that the second
and third clauses are *not* done: running a real KRED task needs a laboratory
and a reference set, and nothing here substitutes for either.

**Established, with a test that fails without it**

- Aggregation uses only wells that were actually tested, reconciles units or
  refuses, and groups by construct, method and endpoint; a plan column can no
  longer be read as a measurement.
- A pocket is read in the right numbering frame; a stereochemical call is
  corrected for circularity; an uncalibrated margin cannot reject; a numeric
  claim in model prose must cite the cell it came from, and the cell is checked.
- A window's `calibrated_on` is a record anyone can open: a `calibration:<digest>`
  citation is resolved, its digest recomputed, its verdict re-run and its window
  compared -- and anything that cannot be shown fails closed to *uncalibrated*.
  Free text is still accepted by default and is *named* in every report;
  `strict_calibration` refuses it.
- Four live routes (UniProtKB, RCSB PDB, Rhea, Zenodo) with clients checked
  against recorded responses, which found three defects no unit test had.
- A real dataset enters through a route that hashes before it writes.
- `eagent benchmark sdr` and `eagent benchmark feedback` run on that dataset.

**Measured on the SDR deposit (annotation-derived labels), recorded in
`docs/results/`**

- Recovering an annotated class from sequence, with test sequences held out by
  cluster: macro AUROC **0.667** for the spectrum-kernel model against **0.710**
  for nearest-neighbour. The model is better on 2 of 12 targets (the NAD/NADP
  cofactor and one substrate cluster), worse on 1, and indistinguishable on 9.
- Ungrouped folds inflate AUROC by about **0.26-0.28** (0.71 to 0.97). That gap
  is the reason grouping is mandatory, and it is the size of the error a
  benchmark makes when it does not group.
- A probability is reported only where an out-of-fold calibration shows skill:
  **3 of 12** targets earned one. The others rank but state no probability.
- Feeding measured labels back into selection: on the full pool the model beats
  random by a wide margin and feedback adds 2-6 hits over a frozen model on
  most classes; on a one-per-cluster pool most of that advantage disappears
  (phenol: no difference from random). A similarity-guided method looks good
  largely because the pool is redundant.

**Not established**

- Nothing about ketoreductase activity on any substrate. The labels are
  annotations, "measuring" is revealing an annotation, and the SDR classes are
  not a KRED panel.
- Nothing about the agent's gates or ranking: they cannot be scored against
  annotation labels.
- That any adapter works against its real binary (gnina, Boltz, MMseqs2) or that
  `AnthropicMessagesClient` works against the live API.
- That a language model is useful in the loop. The planner is built so that it
  can do little harm; whether it does any good has not been measured.
- A calibrated window. The mechanism exists and a reference set now exists
  (next section); the set cannot calibrate any shipped window.
- Any licence beyond the four read facets and the four reported ones.

**What would change this.** More experimental KRED complexes with activity labels
and known inactives (calibration -- see below for how many); a plate of real
results from a registered criterion (every claim about learning from
experiments); the real binaries installed and one run recorded (the adapters);
and a person reading the repositories of the two reported licences.

### The KRED reference set (added 2026-10-08)

A spreadsheet of 19 experimental PDB entries and 28 kinetic records, compiled by
an AI assistant and uploaded by the repository's owner, is stored, hash-pinned and
checked under `configs/references/kred_calibration/v0.1` (its `README.md` and
`NOTICE.md` are the long form). This is the first real reference data in the
repository. What it established:

- **It loads honestly.** Every stated count holds; all 74 cached unit
  conversions equal a recomputation; every cross-reference is followed both
  ways; all 118 ligand-validation rows name residues that are in the pinned
  coordinate files (19 files, 16 MB, fetched through a route that was probed
  first and verified by hash, not committed).
- **Its kinetic numbers match their sources.** 25 of 28 records match the paper's
  own table, 1 matches BRENDA's extraction, 2 match BRENDA for `kcat`/`Km` with
  six quantities unread; all 10 `er` values match; nothing mismatched.
- **It is smaller than it looks.** Nineteen entries are six lineages, and the 22
  "core" kinetic records are three; the two *Lactobacillus* enzymes the workbook
  counts separately are 88 % identical. Only two lineages have a substrate or
  product placed in the site.
- **It cannot calibrate a shipped window.** Against the NADPH-SDR template one
  entry of 19 is eligible (1IPF) and there are no known inactives; relaxing the
  cofactor requirement admits a second lineage. A modest claim needs 14
  independent actives, a conventional one 38. Milestone-1 criterion 5 stays
  unmet, and the count stays 17 of 17.
- **It already says something about one window.** The measured hydride-donor
  approach angles (72.6°-81.6°, three entries, two lineages) all lie outside the
  shipped advisory 90-130° band.

**Not established by it:** any window; that its bindings are right (rule-derived,
unreviewed, protein roles unbound); anything about the 2026 Ssal-KRED ortholog
extension, whose files were not received; and any kinetic claim beyond what the
sources print.

---

## 中文摘要

### 现状一句话

框架已完成并通过测试（写作时 2919 个测试 + 696 个子测试）。**已对 4 个公共数据库做过真实调用**
（UniProtKB、RCSB PDB、Rhea、Zenodo），并通过校验和核对的路径下载了一份真实数据集；但**没有跑过
任何真实的结构预测器、对接程序、反向折叠模型或搜索二进制，也没有读过任何真实的实验板**。里程碑 1
是"已建成但未演示"；里程碑 2 和 3 尚未开始，因为它们需要的是实验室，不是更多代码。

**这个仓库目前能支持的说法**：具备可靠的"做出主张的机器"；有一个"序列 → 已标注类别"的模型，
在少数类别上优于最近邻、多数类别上与之无差别；有一个模拟表明在**由标注得来的数据**上，把标签反馈
回选择确实有帮助。**不能支持**"智能体已经从实验中学习并验证了新酶发现能力"——至今没有做过任何实验。
详见 §7。

**KRED 参考集（2026-10-08 加入）**：用户上传的 KRED 实验复合物参考集（19 个 PDB 条目、28 条动力学记录、
10 条 er）已原样保存并哈希固定（`configs/references/kred_calibration/v0.1`），坐标文件通过先探测后下载、
下载后按哈希核对的路径取得（19 个文件共 16 MB，不入库）。检查结果：声明的每个计数都成立；74 个缓存的单位
换算与独立重算一致；交叉引用双向成立；动力学数值与原始表格比对，25/28 条与论文自己的表一致、1 条只与
BRENDA 一致、2 条部分未读原文，**无不一致项**。但它**比看上去小**：19 个条目只是 6 个谱系，22 条"核心"记录
只是 3 个谱系，真正放了底物/产物的谱系只有 2 个；按现有模板只有 1 个条目合格，**没有任何窗口能被它标定**，
里程碑 1 的第 5 条验收标准仍未满足。实测的氢负离子供体接近角（72.6°–81.6°）全部落在模板 90–130° 的
咨询窗口之外——3 个条目、2 个谱系，不足以替换窗口，但足以让它继续只是 `advisory`。

### 里程碑 1：证据层与可复现筛选

**含义**：输入一个具名底物，输出一批构建体，背后有被记录的人工批准，每个候选都能追溯到支撑
它的证据，整次运行可以从清单重算。这个里程碑**不要求任何一个酶能工作**，它要求的是
"做出主张的机器"是可靠的。

**具体缺口**（2026-10 更新）：49 个数据源中只有 **4 个**做过连通性测试（UniProtKB、RCSB PDB、
Rhea、Zenodo，调用已记录；SABIO-RK 实测不可达，记为观察而非结论），其余 45 个未测，全部
`needs_curation`；搜索二进制、结构预测器、对接程序、反向折叠模型**一个都没装**——gnina 与
Boltz 的适配器已写好，但只用"按文档格式写输出的假程序"测过，**从未跑过真二进制**；56 个许可面中
**4 个读过原文**（Boltz 代码与权重 MIT、Vina 代码 Apache-2.0、gnina README 的双许可声明）、
**4 个是他人转述未读原文**（VenusMine 记为 CC BY-NC-ND 4.0：限制商用与衍生；AP Novo 代码
记为 Apache-2.0 但不声称任何许可，权重与输出记为非商用）、其余 48 个未知，未知的商用许可会
**阻断**商业运行；17 条几何窗口**全部未标定**——标定机制已建成；仓库里现在有了第一份真实参考集（19 个实验条目、
28 条动力学记录，已哈希固定并审计，见"§7 之后的 KRED 参考集"），但按现有 NADPH-SDR 模板只有 1 个条目
合格、没有已知无活性的反例，最低要求（80%/80%）需要 14 个独立阳性，所以没有任何窗口能通过。

**验收标准（五条，每条都可能失败）**：从已填充的连接器缓存跑到 `select_batch`，三个闸门由
具名的人清掉，产出订单与非空阳性判据的 `experiment_plan.yaml`；`bundle` 报告
**complete: YES** 且在另一台机器上 `bundle-verify` 通过；独立校验器在真实候选与真实主张上
零 blocker（或者报出的 blocker 是策展者随后修掉的真缺陷）；同一份清单重跑产出**同一批**
（不是"相似的一批"）；至少一条催化窗口有非空的 `calibrated_on`。

### 里程碑 2：天然酶发现闭环，接上真实实验反馈

**含义**：板子回来，系统把它读成带类型的记录，按**上板前**注册的判据分类，报告两个命中率分母
及其区间，产出一个被确认的亲本或一份五路无命中鉴别。闭环的标志是：这一轮的产出通过
`next_round_quotas` 改变了下一轮的构成，且这个改变**可归因**。

**验收标准**：≥48 个构建体被**提交**，且每一个提交过的构建体——包括没表达的和没测的——
在自建库里都有一行（缺一行就是不通过，不是疏漏）；回板在**没有判据覆盖**的情况下解析成记录，
每个阴性都带检测限；至少一条二级及以上、由能鉴定产物的方法确认的
`CONFIRMED_TARGET_PRODUCT`——**或者**没有命中，但五路鉴别为每个未被判为 `unlikely` 的假设
指出一个判别实验（只要对照显示检测体系是活的，这**算通过**）；`precision_at_k` 针对上板前的
注册运行；第二轮的家族配额与第一轮不同，且差异可由四种裁决解释。

**诚实的风险**：第一轮最可能的结果就是没有确认命中。这正是为什么
`NO_NATURAL_CATALYST` 永远不会被标成 `consistent`——一个只有"成功了"才能通过的里程碑，会把
项目逼成"报告成功"。

### 里程碑 3：底物导向改造闭环，变体与亲本在**完全相同条件**下比较

**验收标准**：每个变体的亲本必须**在同一块板、同一组条件下**
（`variant_versus_parent` 抛 `ConditionMismatchError` 就是没达标）；每个组合变体的单点突变体
在同一块板上；结果按**三个轴分别**报告（底物适配 / 催化功能 / 稳定性与表达），对照**测试前**
写下的预期；冻结位点检查对每条提案都通过；**至少有一条事先写下的轴预期被数据证伪**并被记录
——一轮里所有预期都被证实，更可能说明预期写得含糊，而不是设计得优秀。

**依赖**：里程碑 3 不能在里程碑 2 产出确认亲本之前开始。

### 从头设计分支：放在三个里程碑之后，单独评估

**不是里程碑 4**，而是一个风险等级、成功率和证据标准都不同的分支。放在后面有一个结构性理由：
塞进同一个漏斗，会让一个极低的期望命中率被平均进流水线的总指标，而一轮里混合了设计体和挖掘
候选的失败将无法归因。

**状态：未开始。** 没有任何模块实现它；`rfdiffusion2` 出现在工具注册表里，只是为了让它的四个
许可面存在、等人来填，没有任何代码调用它。

**诚实的验收标准**：预算封顶（从头设计集**永不**吃掉整整一轮）；设计体与挖掘候选**分开报告、
分开分母**；主要终点不变（由能鉴定产物的方法确认目标产物——"能结合底物"或"折叠如预测"都不算）；
**第一轮零确认命中是可接受、可发表的结果**，只要体系对照工作且检测限有记录（否则这个分支就
变成只有成功时才能被报告的分支）；设计体**永不**凭模型置信度晋级；许可与披露先行
（设计序列是未发表材料，没有覆盖该序列哈希的具名人工授权，不得提交给外部服务）。

### 逐组件状态

只有三种状态：**已实现**（代码存在、有测试、在本环境能跑完）、**接缝**（契约、校验、溯源、
产物布局、拒绝路径都已实现并有测试，但外部程序或数据缺席，所以步骤报 `tool_unavailable`
或结构化的缓存未命中，什么都不产出）、**未开始**（没有代码）。

- **基础与数据模型、确定性科学内核（全部纯 Python）、六层数据层、harness、评估包、交付包、
  CLI**：已实现。
- **接缝**：`mine_sequences`（blastp / mmseqs2 / hmmsearch）、`prepare_structures`
  （结构预测器）、`model_complexes`（对接器与联合复合物预测器）、`propose_mutations`
  （LigandMPNN）、`retrieve_evidence` 与整个连接器层（缓存优先，无具体资源客户端）、
  `eval/baselines` 里的两个比较器。
- **未开始**：其余 45 个数据源的连通性验证（现为 **4 / 49**）、AKR 与 MDR/ADH 的**改造模板**
  （因此这两个家族提不出变体）、从头设计的真实生成器（现在只有会拒绝的接缝）。
- **已实现但需要策展**：全部 11 个模板（催化模板的 `reference_structures` 全空、
  **17 / 17 窗口未标定、0 条 gating**；检测模板的数值门槛全为 `null`）；KRED 参考集
  （`configs/references/kred_calibration/v0.1`：绑定记录由规则导出、**未经人审**，催化残基未绑定）；工具注册表
  （14 个工具 × 4 个许可面 = 56 条：4 读过、4 转述、48 未知）。
- **已知的瑕疵**：`eagent/tools/__init__.py` 里残留着规划期的 `INTERFACE_MODULES` 表，
  列了六个没有任何模块实现的名字。发现机制是权威的，控制器走的路径不报缺失；这是该删掉的
  死重，不是真正的缺口。

### 通往里程碑 1 的最短路径

按"解锁最多"排序：补齐试点任务的四个一号闸门字段 → 按 `evidence_gaps.tsv` 列出的十二条查询
填充连接器缓存（这是策展任务，不是写代码）→ 装一个搜索二进制并指向带快照日期的本地序列库 →
放入 mmCIF 结构或装一个预测器并记录版本与许可 → **标定一条窗口**（选 SDR 的氢负离子转移距离，
给出实验三元复合物，拟合上界，写进 `calibrated_on`，然后才考虑提升 severity）→ 读实际装上的
那些工具的许可并连同版本记录 → 写 AKR 与 MDR/ADH 的改造模板，否则只有 SDR 亲本能提变体。
