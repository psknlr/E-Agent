# Walkthrough — the pilot task, end to end

The shipped pilot task is `KRED_PILOT_001`: asymmetric ketone reduction to a
chiral secondary alcohol by a ketoreductase. This document runs it and shows
what actually happens.

**Everything below is real output.** The commands were executed against this
tree on 2026-10-02 with `PYTHONPATH=/path/to/E-Agent/src` and `eagent` meaning
`python3 -m eagent.cli`, from a working directory containing
`configs/tasks/KRED_PILOT_001.yaml`. Nothing has been edited except the removal
of a block of repeated `[MISSING]` lines, which is marked where it happens.

**What this walkthrough demonstrates is mostly refusal.** The run stops twice,
at two different kinds of stop, and produces an incomplete bundle that says so.
That is the system working. A walkthrough in which the pipeline sailed through
to a batch of 96 constructs would mean every null had been filled by something
— and nothing in this repository is allowed to do that.

---

## 0. The starting state

The task file ships with its four gate-1 fields null, deliberately, with a long
header explaining each one. The short version:

```yaml
task_id: KRED_PILOT_001
task_mode: enzyme_mining
reaction:
  reaction_class: ketone_to_secondary_alcohol
  atom_mapped_reaction_smiles: null
  substrate:  { isomeric_smiles: null, is_prochiral: null, ... }
  product:    { isomeric_smiles: null, creates_new_stereocenter: null,
                target_stereochemistry: unspecified, ... }
  ec_hint: "1.1.1.-"          # a search hint; EC 1.1.1.- is written in the
                              # OXIDATION direction, the reverse of the target
conditions: { cofactor_options: [], pH: null, temperature_C: null, ... }
approval:   { reaction_spec_confirmed: false, synthesis_authorized: false,
              functional_criteria_confirmed: false }
```

---

## 1. `eagent validate` — what is unresolved, gate by gate

```
$ eagent validate configs/tasks/KRED_PILOT_001.yaml --all-gates
```

```
task KRED_PILOT_001  (enzyme_mining)
file configs/tasks/KRED_PILOT_001.yaml
reaction class: ketone_to_secondary_alcohol

gate reaction_spec_confirmed  [asked about]
-------------------------------------------
  Is this the reaction the project is about: this substrate structure, this product structure, this configuration?
  required fields:         4
  unresolved:              4
    - reaction.substrate.isomeric_smiles: null -- an operator must supply this
    - reaction.product.isomeric_smiles: null -- an operator must supply this
    - reaction.product.creates_new_stereocenter: null -- an operator must supply this
    - reaction.atom_mapped_reaction_smiles: null -- an operator must supply this
  flag in the task file:   no
    a flag in a file is not an approval: the gate is cleared only by `eagent approve`, which records who decided and when in the run manifest

gate synthesis_authorized  [asked about]
----------------------------------------
  Authorise this batch: these constructs, these wells, this cost?
  required fields:         5
  unresolved:              5
    - reaction.substrate.isomeric_smiles: null -- an operator must supply this
    - reaction.product.isomeric_smiles: null -- an operator must supply this
    - conditions.pH: null -- an operator must supply this
    - conditions.temperature_C: null -- an operator must supply this
    - conditions.expression_host: null -- an operator must supply this
  flag in the task file:   no
    a flag in a file is not an approval: the gate is cleared only by `eagent approve`, which records who decided and when in the run manifest

gate functional_criteria_confirmed  [asked about]
-------------------------------------------------
  Which experimental result counts as supporting a functional claim, decided before the data exist?
  required fields:         1
  unresolved:              1
    - reaction.product.isomeric_smiles: null -- an operator must supply this
  flag in the task file:   no
    a flag in a file is not an approval: the gate is cleared only by `eagent approve`, which records who decided and when in the run manifest

assumptions on record
---------------------
  none: nothing in this task has been filled in, so nobody has had to justify anything yet

error: 3 gate(s) you asked about cannot be satisfied: reaction_spec_confirmed needs reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles, reaction.product.creates_new_stereocenter, reaction.atom_mapped_reaction_smiles; synthesis_authorized needs reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles, conditions.pH, conditions.temperature_C, conditions.expression_host; functional_criteria_confirmed needs reaction.product.isomeric_smiles
blocked on a human decision: reaction_spec_confirmed
next: resolve each field through TaskSpec.resolve() with a named authority -- operator:, literature:, database:, template: or experiment: -- and validate again
```

**Exit code 3** — "something is unresolved". Two things to notice: the tool
reports the assumption ledger as *empty* rather than omitting it, and it states
explicitly that the `false` flags in the YAML are not refusals-to-approve, they
are simply not approvals at all.

---

## 2. `eagent run --dry-run` — what would happen, with nothing executed

```
$ eagent run configs/tasks/KRED_PILOT_001.yaml --rundir runs/kred-001 --dry-run
```

```
task:                    KRED_PILOT_001
run directory:           runs/kred-001
run id:                  KRED_PILOT_001-20261002T011445Z
resumed:                 no
network:                 disabled
template gaps:           2
  - family 'AKR' has no engineering template; variants for it cannot be proposed
  - family 'MDR/ADH' has no engineering template; variants for it cannot be proposed
geometry windows:        17 of 17 geometry constraint(s) carry no calibration set; 0 come from a theoretical-model template

dry run -- nothing was executed
-------------------------------
  no interface was executed, no external binary was started and no network call was made

steps that would run
--------------------
  - normalize_reaction v0.1.0
  - retrieve_evidence v0.1.0
  - mine_sequences v0.1.0
  - annotate_family v0.1.0
  - prepare_structures v0.1.0
  - model_complexes v0.1.0
        waits on: reaction.substrate.isomeric_smiles
  - evaluate_catalysis v0.1.0
        waits on: reaction.substrate.isomeric_smiles, reaction.product.creates_new_stereocenter
        needs approval: reaction_spec_confirmed
  - select_batch v0.1.0
        waits on: reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles, conditions.pH, conditions.temperature_C, conditions.expression_host
        needs approval: synthesis_authorized
  - ingest_results v0.1.0
        waits on: reaction.product.isomeric_smiles
        needs approval: functional_criteria_confirmed
  - propose_mutations v0.1.0
        waits on: reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles

human decision points
---------------------
  - reaction_spec_confirmed: 4 unresolved field(s): reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles, reaction.product.creates_new_stereocenter, reaction.atom_mapped_reaction_smiles
  - synthesis_authorized: 5 unresolved field(s): reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles, conditions.pH, conditions.temperature_C, conditions.expression_host
  - functional_criteria_confirmed: 1 unresolved field(s): reaction.product.isomeric_smiles

cost
----
  this harness has no price list: compute time, synthesis and plate costs are site-specific. A total is left null rather than filled with a plausible figure, because a figure in a plan gets approved as though it were checked
  recorded so far:         nothing has been recorded by any step of this run
  ceilings:                none configured for this run

planned scale (from the task file, not a prediction)
----------------------------------------------------
  initial_sequence_target:         2000
  family_qc_pool_target:           600
  structure_pool_target:           600
  detailed_complex_target:         300
  new_constructs_round_1:          96
  constructs_include_controls:     no
  reserved_control_slots:          0

wrote runs/kred-001/dry_run_plan.json
No run manifest was written: nothing ran, and a manifest for a run that did not happen would be read as one that did.
```

Four lines here carry most of the design:

- **`17 of 17 geometry constraint(s) carry no calibration set`** is printed at
  the head of *every* run, not hidden in a lint command. See
  [`../docs/TEMPLATES.md`](../docs/TEMPLATES.md) §5.
- **`planned scale (from the task file, not a prediction)`** — the funnel
  numbers are a resource plan. Nothing asserts that 600 of 2000 will survive.
- **The cost block refuses to invent a total.** "A figure in a plan gets
  approved as though it were checked."
- **No manifest is written for a dry run**, because a manifest for a run that
  did not happen would be read as one that did.

---

## 3. `eagent run` — the first stop

```
$ eagent run configs/tasks/KRED_PILOT_001.yaml --rundir runs/kred-001
```

```
run
---
  . start -> confirm_reaction_spec: run started
  . confirm_reaction_spec -> await_reaction_approval: waiting for the reaction spec to be confirmed
  . await_reaction_approval -> awaiting_human: the reaction spec has not been confirmed; nothing downstream may run, because every later step would be evidence about an unconfirmed molecule
  [confirm_reaction_spec] normalize_reaction: partial (attempt 1, needs_human)
      reaction spec is not usable yet: 4 blocking problem(s), 4 unresolved field(s)
      qc BLOCKER: substrate_unspecified [reaction.substrate] -- no substrate was given; in enzyme-mining and engineering modes the substrate is the index of the whole search
      qc BLOCKER: product_unspecified [reaction.product] -- no product was given; without it the assay has no target and 'confirmed_target_product' cannot be defined
      qc BLOCKER: stereocentre_undetermined [reaction.product.creates_new_stereocenter] -- whether the reaction creates a new stereocentre is undetermined (both substrate and product structures are needed before the question can even be asked); it is not inferred from the reaction class, because an aldehyde and a symmetric ketone both give an achiral alcohol
      qc warn: cofactor_unspecified [conditions.cofactor_options] -- no cofactor option is declared; a hydride-transfer geometry cannot be defined without naming the donor
      qc BLOCKER: atom_map_missing -- no atom-mapped reaction SMILES; the reactive-atom specification cannot be checked and the reaction_spec_confirmed gate requires it
      uncertainty stereocentre_undetermined: Does this reaction create a new stereocentre in the product? (resolvable by operator input, or a sourced ReactionTemplate)
      uncertainty cofactor_identity: Which cofactor (and which oxidation state) does the target reaction use? (resolvable by operator input, or a sourced CatalyticTemplate)
      uncertainty atom_map_inconsistent: Which atom-mapped reaction SMILES and reactive-atom ids describe this transformation correctly? (resolvable by operator input, or an atom mapper whose output a chemist has reviewed)
      uncertainty unresolved_fields: Which values should fill: reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles, reaction.product.creates_new_stereocenter, reaction.atom_mapped_reaction_smiles? (resolvable by operator input)
      next (human): resolve_fields -- The reaction_spec_confirmed gate cannot open while these fields are null; this step will not fill them, because a guessed structure or stereocentre propagates into a synthesis order

outcome
-------
stage:                   awaiting_human
outcome:                 awaiting_human
reason:                  the reaction spec has not been confirmed; nothing downstream may run, because every later step would be evidence about an unconfirmed molecule
cost recorded:           nothing recorded by any step

waiting on a human
------------------
  - gate reaction_spec_confirmed (request 362a9b5c5ddd7ed0)
        Is this the reaction the project is about: this substrate structure, this product structure, this configuration?
        The reaction_spec_confirmed gate cannot open while these fields are null; this step will not fill them, because a guessed structure or stereocentre propagates into a synthesis order
        answer it with: eagent approve reaction_spec_confirmed --rundir runs/kred-001 --actor <your name>
  - gate reaction_spec_confirmed (request d286d0b4225d4ce4)
        Is this the reaction the project is about: this substrate structure, this product structure, this configuration?
        confirm the substrate structure, the product structure and the target configuration before any search starts
        answer it with: eagent approve reaction_spec_confirmed --rundir runs/kred-001 --actor <your name>

manifest: runs/kred-001/run_manifest.json
report:   runs/kred-001/controller_report.json
error: the reaction spec has not been confirmed; nothing downstream may run, because every later step would be evidence about an unconfirmed molecule
blocked on a human decision: reaction_spec_confirmed
next: eagent approve reaction_spec_confirmed --rundir runs/kred-001 --actor <your name>
```

**Exit code 4** — "a human decision point is blocking". The run is intact and
resumable.

### What the system refused to do here, and why

| It could have | It did not, because |
| --- | --- |
| Resolved "the usual model ketone" into acetophenone | A name is not a structure. It fixes neither tautomer, salt form nor stereochemistry, and a guessed structure redefines the project — every sequence mined afterwards would be evidence about a different molecule. |
| Inferred `creates_new_stereocenter: true` from `reaction_class: ketone_to_secondary_alcohol` | An aldehyde and a symmetric ketone both give an achiral alcohol. The class label is itself an operator assertion and is frequently the thing that is wrong, so inferring from it makes the error unfalsifiable. |
| Assumed NADPH, because most ketoreductases of interest use it | The cofactor decides what the `no_cofactor` control means and what the hydride-transfer geometry is measured against. `cofactor_unspecified` is a warning here and would become a blocker at modelling. |
| Carried on to `retrieve_evidence` anyway | Every later step would be evidence about an unconfirmed molecule. |

Note also that `normalize_reaction` returned **`partial`, not `failed`** — it
did its job, which is to triage. And it still wrote `reaction_spec.yaml`,
because a spec and its QC record belong in the same file: *a spec file that
looks complete, read six months later without its QC record, is
indistinguishable from one that was completed by guesswork.*

### What was written

```
runs/kred-001/
  reaction_spec.yaml       the spec + what is wrong with it
  run_manifest.json        hashes, versions, seeds, steps, approvals, cost
  controller_report.json   the path, every attempt, pending decisions
  approvals.json           the persisted queue
  dry_run_plan.json        from step 2
```

`reaction_spec.yaml` ends with exactly the fields the gate is waiting for:

```yaml
checks:
  stereocentre:
    value: null
    basis: undetermined
    detail: both substrate and product structures are needed before the question can
      even be asked
    source: null
  atom_map:
    parsed: false
    issues:
    - code: atom_map_missing
      severity: blocker
      message: no atom-mapped reaction SMILES; the reactive-atom specification cannot
        be checked and the reaction_spec_confirmed gate requires it
  unresolved_for_gate:
  - reaction.substrate.isomeric_smiles
  - reaction.product.isomeric_smiles
  - reaction.product.creates_new_stereocenter
  - reaction.atom_mapped_reaction_smiles
assumptions: []
```

---

## 4. `eagent status` — where the machine is

```
$ eagent status runs/kred-001
```

```
run:                     KRED_PILOT_001-20261002T011446Z
task:                    KRED_PILOT_001
created:                 2026-10-02T01:14:46+00:00
task input sha256:       bf48ce05ac1243dda7cc5015d50158218b7a176a4e05e1ec6fd56f97aa67acc9

state machine
-------------
stage:                   awaiting_human
outcome:                 awaiting_human
stop reason:             the reaction spec has not been confirmed; nothing downstream may run, because every later step would be evidence about an unconfirmed molecule
path:                    start -> confirm_reaction_spec -> await_reaction_approval -> awaiting_human

steps (1)
---------
  - confirm_reaction_spec:normalize_reaction: partial (2026-10-02T01:14:46+00:00 -> 2026-10-02T01:14:46+00:00)
        reaction spec is not usable yet: 4 blocking problem(s), 4 unresolved field(s)

cost
----
  nothing was recorded by any step. This is not a claim that the run was free: it means no step reported a cost

approvals
---------
  none recorded; no gate in this run has been cleared by a named person

pending decisions (2)
---------------------
  - reaction_spec_confirmed (request 362a9b5c5ddd7ed0) raised by confirm_reaction_spec:normalize_reaction
        The reaction_spec_confirmed gate cannot open while these fields are null; this step will not fill them, because a guessed structure or stereocentre propagates into a synthesis order
  - reaction_spec_confirmed (request d286d0b4225d4ce4) raised by controller
        confirm the substrate structure, the product structure and the target configuration before any search starts

open uncertainties (4)
----------------------
  - [confirm_reaction_spec:normalize_reaction] stereocentre_undetermined: Does this reaction create a new stereocentre in the product?
        resolvable by: operator input, or a sourced ReactionTemplate
  - [confirm_reaction_spec:normalize_reaction] cofactor_identity: Which cofactor (and which oxidation state) does the target reaction use?
        resolvable by: operator input, or a sourced CatalyticTemplate
  - [confirm_reaction_spec:normalize_reaction] atom_map_inconsistent: Which atom-mapped reaction SMILES and reactive-atom ids describe this transformation correctly?
        resolvable by: operator input, or an atom mapper whose output a chemist has reviewed
  - [confirm_reaction_spec:normalize_reaction] unresolved_fields: Which values should fill: reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles, reaction.product.creates_new_stereocenter, reaction.atom_mapped_reaction_smiles?
        resolvable by: operator input

blocking QC flags (4)
---------------------
  - [confirm_reaction_spec:normalize_reaction] substrate_unspecified: no substrate was given; in enzyme-mining and engineering modes the substrate is the index of the whole search
  - [confirm_reaction_spec:normalize_reaction] stereocentre_undetermined: whether the reaction creates a new stereocentre is undetermined (both substrate and product structures are needed before the question can even be asked); it is not inferred from the reaction class, because an aldehyde and a symmetric ketone both give an achiral alcohol
  ...
```

`cost` is worth reading twice: **"nothing was recorded by any step. This is not
a claim that the run was free: it means no step reported a cost."** It does not
print `0.0`.

---

## 5. `eagent approve` — and what it tells you before you decide

In a real campaign the operator would now resolve the four fields and re-run.
To show the next stop, this walkthrough approves the gate **with the fields
still null**, which the CLI makes uncomfortable on purpose:

```
$ eagent approve reaction_spec_confirmed --rundir runs/kred-001 \
      --actor "J. Chemist" \
      --note "demonstration: the four required fields are still null"
```

```
gate reaction_spec_confirmed
----------------------------
  Is this the reaction the project is about: this substrate structure, this product structure, this configuration?
  if this is wrong: every sequence mined, every complex modelled and every gene ordered afterwards is evidence about the wrong molecule
request:                 d286d0b4225d4ce4
raised by:               controller
raised at:               2026-10-02T01:14:46+00:00
detail:                  confirm the substrate structure, the product structure and the target configuration before any search starts

what this decision covers
-------------------------
  task_id:                   KRED_PILOT_001
  unresolved:                reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles, reaction.product.creates_new_stereocenter, reaction.atom_mapped_reaction_smiles
warning: the request does not carry: reaction.substrate.isomeric_smiles, reaction.product.isomeric_smiles, reaction.product.target_stereochemistry, reaction.product.creates_new_stereocenter, reaction.atom_mapped_reaction_smiles, assumptions -- decide only if you have those facts from elsewhere

recorded
--------
decision:                approve
actor:                   J. Chemist
at:                      2026-10-02T01:14:47+00:00
reason:                  demonstration: the four required fields are still null

resume with: eagent run <task.yaml> --rundir runs/kred-001
```

Three things:

1. **The decision answers a specific request.** There is no way to approve a
   gate in the abstract; the request id, who raised it and when are all printed.
2. **The consequence of getting it wrong is printed before the decision.**
3. **The CLI lists what the request does *not* carry** — the `must_show` fields
   of the `DecisionPoint` — and says "decide only if you have those facts from
   elsewhere". The approval goes through, with the operator's name and reason on
   it. The system does not pretend to know better than a named human; it makes
   sure the human knows what they are signing.

The grant is written to the **manifest**, not to the task file. Re-running with
a *different* payload would create a new request rather than inheriting this
grant.

---

## 6. `eagent run` again — resume, and the second stop

```
$ eagent run configs/tasks/KRED_PILOT_001.yaml --rundir runs/kred-001
```

```
resumed:                 yes
...
run
---
  . start -> confirm_reaction_spec: run started
  . confirm_reaction_spec -> retrieve_evidence: reaction spec confirmed by a recorded decision
  . retrieve_evidence -> awaiting_human: retrieve_evidence is waiting on a human decision: no_evidence_retrieved: no experiment record was retrieved. This is an empty evidence base, not a negative result: nothing downstream may treat it as evidence that no enzyme performs this reaction.
  [confirm_reaction_spec] normalize_reaction: partial (attempt 1, needs_human)
      ... (the same four blockers; the gate is approved, the fields are still null)
  [retrieve_evidence] retrieve_evidence: partial (attempt 1, needs_human)
      0 record(s) retrieved; 12 query gap(s) and 0 rejected payload record(s). Gaps are missing data, not negative results.
      qc BLOCKER: no_evidence_retrieved [evidence] -- no experiment record was retrieved. This is an empty evidence base, not a negative result: nothing downstream may treat it as evidence that no enzyme performs this reaction.
      qc warn: evidence_gaps [evidence] -- 12 of 12 planned queries returned no payload; see evidence_gaps.tsv for the exact cache files needed. Absence here is absence of data, never of activity.
      qc info: layers_not_searched [query_plan] -- not searched: pdb (structure retrieval is keyed on accessions, which do not exist until the sequence-mining step has run), alphafold (structure retrieval is keyed on accessions, which do not exist until the sequence-mining step has run)
      uncertainty evidence_base_empty: Which curated imports must be placed in the connector cache before this task has an evidence base? (resolvable by a curator populating the cache paths listed in evidence_gaps.tsv)
      next (human): populate_connector_cache -- A curator imports the named records so the run can be replayed offline with the gaps closed

outcome
-------
stage:                   awaiting_human
outcome:                 awaiting_human
reason:                  retrieve_evidence is waiting on a human decision: no_evidence_retrieved: no experiment record was retrieved. This is an empty evidence base, not a negative result: nothing downstream may treat it as evidence that no enzyme performs this reaction.
cost recorded:           nothing recorded by any step

waiting on a human
------------------
  - gate reaction_spec_confirmed (request 362a9b5c5ddd7ed0)
        ...
        this gate already carries a recorded grant; the request is a duplicate raised by a later step
  - operator_task populate_connector_cache (request 1eab6b38e80a363a)
        A curator imports the named records so the run can be replayed offline with the gaps closed
```

Exit code **4** again.

### The second refusal is the important one

Twelve queries were planned and twelve returned nothing, because the connector
cache is empty and the network is disabled. A less careful pipeline reports
"no known enzymes for this reaction" here. This one reports:

> **This is an empty evidence base, not a negative result: nothing downstream
> may treat it as evidence that no enzyme performs this reaction.**

A cache miss is a fact about this machine, not about the literature.

Note also the `layers_not_searched` **info** flag. PDB and AlphaFold were not
queried, and the reason is given — structure retrieval is keyed on accessions,
which do not exist until mining has run — along with what unblocks them. A
layer that was skipped is reported as skipped, not silently absent from the
plan.

### `evidence_query_plan.yaml` — written *before* retrieval

Recall cannot be judged from results, only from queries. The file exists so a
human can say "you never searched the cyclic ketones":

```yaml
# evidence_query_plan.yaml -- written BEFORE retrieval.
target_reaction:
  reaction_class: ketone_to_secondary_alcohol
  substrate_smiles: null
  ec_hint: 1.1.1.-
  direction_requirement: records measured in the reverse direction do not support
    the target direction and are counted separately
candidate_families: [AKR, MDR/ADH, SDR]
queries:
- query_id: 549cdc970df55223
  connector: rhea
  data_layer: reaction
  purpose: find the balanced reactions behind the EC hint
  query: {ec: 1.1.1.-}
  evidence_strength_ceiling: annotation_only
  must_not_be_used_for:
  - 'Names no protein: a Rhea id on an entry is an annotation, never evidence that
    that sequence catalyses the reaction.'
  ...
skipped_connectors:
- connector: pdb
  data_layer: structure
  reason: structure retrieval is keyed on accessions, which do not exist until the
    sequence-mining step has run
  unblocked_by: the candidate accession list from mine_sequences
recall_caveats:
- 'A cache miss is a fact about this machine, not about the literature: it is never
  evidence that no enzyme performs this reaction.'
- Substrate matching in the kinetics layer is by name or EC, so a substrate whose
  records use a different synonym will be missed; the synonym list is the recall limit
  of this whole step.
- Records keyed to an EC number and an organism are capped at ec_species_mapped and
  must not be read as sequence-level evidence.
- Several of these resources re-publish one another's records; counting hits overstates
  corroboration, so the matrix counts independent sources instead.
- No substrate name or synonym was supplied, so the kinetics layer could only be searched
  by EC class, if at all.
plan_sha256: 10a0ac411c9384a1b9dab17e13467eef12db587822f1f7ac5cb8f4ba31d39dfa
```

### `evidence_gaps.tsv` — what a curator must actually place

```
# What is missing from the evidence base, and what would close it.
# kind=query_gap: a planned query returned no payload. This is missing data, not a negative result.
# kind=record_rejected: a cached record could not be typed; it was not repaired, because the missing field is a measurement.
kind	connector	subject	reason	needed
query_gap	rhea	549cdc970df55223	not cached and network access is disabled for this run	a curated import of rhea (reaction layer) answering {"ec":"1.1.1.-","op":"search"}; placed with OfflineConnector.store_import('search', <query>, payload), which writes …
query_gap	enzymemap	df76c7a56b423829	not cached and network access is disabled for this run	…
query_gap	brenda	b192fe4518fa232a	not cached and network access is disabled for this run	…
query_gap	sabio_rk	6d8ee5f2730ddc5a	not cached and network access is disabled for this run	…
query_gap	mcsa	a4c9144287844e74	not cached and network access is disabled for this run	…
query_gap	uniprot	2058491fa4dc7145	not cached and network access is disabled for this run	… {"ec":"1.1.1.-","family":"AKR","op":"search"} …
```

Each row names the connector, the exact query, and the call that would place the
file. This is a curation worklist, not an error log.

### `evidence_matrix.tsv` — empty, and legible anyway

The matrix header defines its own vocabulary before any data exists, which is
what lets an empty matrix still be read correctly:

```
#   c   confirmed target product (target direction only)
#   wt  confirmed by a wild-type sequence
#   var confirmed only by an engineered variant
#   rev confirmed, but measured in the reverse direction; this does not support the target direction
#   nd  tested, target product not detected at the stated limit
#   ef  expression or solubility failure; catalytic ability unknown
#   op  turnover to another product or the wrong configuration
#   ut  present but not tested
#   comp computational entries only; never an experimental negative
#   ind number of independent sources behind the confirmations
#   str strongest evidence any record in the cell is entitled to claim
#
# Maturity vocabulary:
#   mature_natural: independent wild-type successes exist for this chemotype
#   natural_single_report: one wild-type success; not corroborated independently
#   engineered_only: success reported only for engineered variants; a wild-type starting point for this chemotype is not evidenced here
#   tested_negative: tested, target product not detected at the stated detection limit
#   expression_limited: constructs failed expression; catalytic ability is undetermined, which is not the same as inactive
#   reverse_direction_only: activity reported only in the reverse direction; it does not support the target direction
#   no_reliable_data: only annotation-level or computational entries; essentially no reliable experimental data
#   untested: no record at all for this combination
#
# Corroboration counted by: no confirmed records to count
# A cell reading 'untested' means no record was retrieved; it is not a negative result.
# Built from 0 record(s) over 12 planned quer(ies), of which 12 returned no payload.
# Chemotype columns are assigned only from a curated declaration or an operator table; nothing is perceived from structure.
```

Four distinctions live in that legend that a normal screening table destroys:
wild-type versus engineered success, target versus reverse direction, tested
negative versus expression failure versus untested, and number of **independent
sources** versus number of rows.

---

## 7. `eagent verify` — an empty verification is not a passed one

```
$ eagent verify runs/kred-001
```

```
error: there is nothing in runs/kred-001 for the verifier to check: no candidates.json, no claims.json and no ingest_results/experiment_records.jsonl
next: export the candidates the run produced, or point at them with --candidates; an empty verification is not a passed one
```

**Exit code 3.** The verifier refuses to return a clean report over nothing.
This matters because "verification passed" on an empty input is the most
dangerous possible output of a verification step.

---

## 8. `eagent bundle` — a package that says what is missing

```
$ eagent bundle runs/kred-001 --out packages/kred-001
```

```
bundle: packages/kred-001
run: KRED_PILOT_001-20261002T011446Z  task: KRED_PILOT_001
items: 4 present, 0 partial, 14 missing, of 18 declared
complete: NO
  [ok]      reaction_spec.yaml
  [ok]      evidence_records.jsonl
  [MISSING] candidate_sequences.fasta
            reason: not produced by this run and not present in runs/kred-001
            supply: run the mine_sequences step
  [MISSING] sequence_annotations.tsv
            reason: not produced by this run and not present in runs/kred-001
            supply: run the annotate_family step
  [MISSING] family_analysis
            reason: not produced by this run and not present in runs/kred-001
            supply: run the annotate_family step
  [MISSING] structures
            reason: not produced by this run and not present in runs/kred-001
            supply: run the prepare_structures step, or deposit the chosen mmCIF files
  [MISSING] complexes
            reason: not produced by this run and not present in runs/kred-001
            supply: run the model_complexes step
  [MISSING] confidence_metrics
            reason: not produced by this run and not present in runs/kred-001
            supply: copy the predictor's confidence files; a plot image or a single mean is not a substitute
  [MISSING] residue_atom_mapping.tsv
            reason: not produced by this run and not present in runs/kred-001
            supply: run the prepare_structures step
  [MISSING] catalytic_geometry.tsv
            reason: not produced by this run and not present in runs/kred-001
            supply: run the evaluate_catalysis step

          [--- 6 further MISSING items elided: candidate_scorecards.tsv,
               candidate_explanations.txt, selected_batch_96.csv,
               mutation_proposals.tsv, experiment_plan.yaml,
               assay_results_template.csv ---]

  [ok]      run_manifest.json
  [ok]      research_report.md

manifest: packages/kred-001/bundle_manifest.json
report:   packages/kred-001/research_report.md

verify it anywhere with: eagent bundle-verify packages/kred-001
warning: the package is incomplete: 14 missing, 0 partial -- it is recorded as such in the manifest, so it will not be mistaken for a complete one
```

Exit code **0** — assembling an incomplete package is not an error; *hiding*
that it is incomplete would be. Note that every missing item carries **what it
is for** and **what a curator must do**, and that `confidence_metrics` names its
own refusal in advance: a plot image or a single averaged number is not a
substitute for the per-residue file.

### What a complete bundle contains

Eighteen declared items, in the order a reader walks them — what the task was,
what was found, what was built, what was measured, what was ordered, what came
back, and the record tying it together:

| # | Item | Why it is in the package |
| --- | --- | --- |
| 1 | `reaction_spec.yaml` | the chemistry the whole package is about, with the fields still unresolved |
| 2 | `evidence_records.jsonl` | one row per retrieved record, with source, strength and licence |
| 3 | `candidate_sequences.fasta` | the sequences every later claim is about |
| 4 | `sequence_annotations.tsv` | the family call per sequence and the signals it rests on |
| 5 | `family_analysis/` | alignments, motif hits, clusters — the working a family call can be rechecked from |
| 6 | `structures/` | **mmCIF**: the coordinates every geometric measurement was made on |
| 7 | `complexes/` | the enzyme–substrate–cofactor poses, ligand coordinates retained |
| 8 | `confidence_metrics/` | per-residue pLDDT/PAE, so a pocket-local figure can be recomputed |
| 9 | `residue_atom_mapping.tsv` | candidate index ↔ author numbering, residue by residue |
| 10 | `catalytic_geometry.tsv` | every measurement against every window, with which windows were restrained |
| 11 | `candidate_scorecards.tsv` | gates and ordinal levels — **no total column, and none may be derived** |
| 12 | `candidate_explanations.txt` | the per-candidate justification a reviewer reads instead of a score |
| 13 | `selected_batch_<n>.csv` | the order form; **named from the actual row count** |
| 14 | `mutation_proposals.tsv` | each substitution with the axis it should move and what it may cost |
| 15 | `experiment_plan.yaml` | roles, quotas, controls, measurement footprint, **the pre-registered criterion** |
| 16 | `assay_results_template.csv` | one row per well, in the columns the ingest step reads back |
| 17 | `run_manifest.json` | input hashes, database and model versions, seeds, costs, approvals |
| 18 | `research_report.md` | the narrative, generated from the artifacts; every quantitative claim names its file |

Three bundle rules worth knowing:

- **mmCIF is the record; a PDB beside it is a validated extra.** A directory
  holding only converted PDB files is reported as *missing its primary records*,
  because PDB format cannot hold a residue number past 9999, a chain id longer
  than one character or a component id longer than three.
- **A filename may not state a number the file does not contain.** A batch file
  called `selected_batch_96.csv` holding 71 constructs becomes "we screened 96"
  at the next meeting, so the file is renamed from the actual row count and the
  rename is recorded.
- **Partial is not present.** A confidence directory holding only pictures is
  `partial`, not `present`, and `complete` stays false.

### The generated report refuses to fill gaps

```markdown
## How to read this report

- Every quantitative claim below names the file it came from in square brackets.
  A number with no such citation is a defect in this generator, not a finding.
- There is no total score anywhere in this package, and none may be derived from
  the scorecard columns: the axes have different units and no calibrated exchange rate.
- An absent artifact is reported as absent. Nothing here is estimated, interpolated
  or filled in from a typical value.
- **This package is incomplete**: 14 declared item(s) missing and 0 partial, of 18
  [source: bundle_manifest.json].

## The reaction

- reaction class: ketone_to_secondary_alcohol [source: reaction_spec.yaml]
- substrate: null -- not supplied by an operator [source: reaction_spec.yaml]
- product: null -- not supplied by an operator [source: reaction_spec.yaml]
- target configuration: unspecified [source: reaction_spec.yaml]
...
## Candidates

- no candidate explanations are in this bundle, so this report states none
- No per-candidate justification is available, so this report makes no claim about any candidate.

## The batch

- the composition of the ordered batch: unavailable -- no batch order form is in this
  bundle; no construct count may be quoted
```

---

## 9. `eagent bundle-verify` — re-checkable anywhere

```
$ eagent bundle-verify packages/kred-001
```

```
bundle:                  packages/kred-001
items checked:           18
files checked:           4
declared complete:       no

problems (0)
------------
  none: every declared file is present and hashes as recorded
```

Exit code **0**. "Zero problems" and "complete" are two different statements and
both are printed. The verifier also reports any file *inside* a declared
directory that the manifest never saw, because an unrecorded file in a traceable
package is either a tampered one or a provenance gap.

---

## 10. Mode B: the system refuses to name a best enzyme

A second task, with the reaction fixed and the substrate open:

```
$ eagent init KRED_SPACE_001 --mode reaction_space_exploration \
      --reaction-class ketone_to_secondary_alcohol
wrote KRED_SPACE_001.yaml

Every field that is null is a decision waiting for an operator. Nothing in this file was inferred.
Next: fill the substrate and product structures, then run `eagent validate KRED_SPACE_001.yaml --all-gates`.

$ eagent run KRED_SPACE_001.yaml --rundir runs/kred-space --step confirm_reaction_spec
```

```
  [confirm_reaction_spec] normalize_reaction: partial (attempt 1, needs_human)
      reaction spec is not usable yet: 2 blocking problem(s), 4 unresolved field(s)
      qc BLOCKER: stereocentre_undetermined [reaction.product.creates_new_stereocenter] -- ...
      qc warn: cofactor_unspecified [conditions.cofactor_options] -- ...
      qc BLOCKER: atom_map_missing -- ...
      qc warn: substrate_class_undecided [reaction.substrate] -- mode B: the reaction is fixed but the substrate is not; the chemical sub-space must be chosen before candidates mean anything
      qc info: best_enzyme_claim_refused [reaction.substrate] -- no 'best enzyme' is reported for an unspecified substrate: selectivity and activity are properties of an enzyme-substrate pair, so a ranking without a substrate would be a ranking of nothing
      uncertainty substrate_sub_space: Which substrate sub-space is in scope: aromatic_ketone, aliphatic_ketone, cyclic_ketone, functionalised_ketone? And which representative member will be ordered? (resolvable by operator decision)
      next (human): choose_substrate_sub_space -- An operator picks one or more sub-spaces and a representative substrate for each; evidence retrieval then runs per sub-space
```

Two differences from mode A are the whole point:

- `substrate_unspecified` and `product_unspecified` are **not** blockers here —
  mode B is allowed to have no substrate. Two blockers instead of four.
- `best_enzyme_claim_refused` is recorded as an **INFO flag on the record**, so
  the refusal is in the artifact rather than only in the terminal. The refusal
  object written into `reaction_spec.yaml` carries all three parts:

```yaml
refusals:
- request: name the best enzyme for this reaction
  refused_because: activity and stereoselectivity are properties of an enzyme-substrate
    pair; with the substrate unspecified there is nothing to rank against
  unblocked_by: choose a sub-space and a representative substrate structure
```

Instead of a ranking, the step emits the four sub-spaces as a decision the
operator has to take, each with its definition, the decision required, why it
matters, the evidence to gather and the representative question — for example,
for aliphatic ketones:

> *why it matters:* when the two substituents are similar in size, face
> discrimination has little to work with, so an enantioselectivity objective may
> be unattainable for part of this sub-space. A symmetric ketone creates no
> stereocentre at all and must be excluded explicitly.
>
> *representative question:* is the chosen member prochiral at the carbonyl
> carbon, or are its two substituents identical?

---

## 11. Reading the library and the registry

```
$ eagent templates lint
```

```
templates loaded:        11

geometry windows
----------------
  17 of 17 geometry constraint(s) carry no calibration set; 0 come from a theoretical-model template
  no window in this library was fitted to systems of known activity: a pass against
  any of them is not evidence of catalytic competence, and a failure may not
  disqualify a candidate
  - cat.sdr.nadph_carbonyl_reduction.v1: hydride_donor_to_carbonyl_carbon (advisory, uncalibrated)
  - cat.sdr.nadph_carbonyl_reduction.v1: hydride_approach_angle_burgi_dunitz (advisory, uncalibrated)
  - cat.sdr.nadph_carbonyl_reduction.v1: carbonyl_oxygen_to_catalytic_tyr_OH (scoring, uncalibrated)
  ... (14 more)

gating windows that are not calibrated
--------------------------------------
  none: no unfitted window is allowed to reject a candidate

integrity (2)
-------------
  - family 'AKR' has no engineering template; variants for it cannot be proposed
  - family 'MDR/ADH' has no engineering template; variants for it cannot be proposed
```

```
$ eagent sources list
```

```
49 registered sources (connectivity_verified=false for every one of them)
  reaction_and_chemistry       7 sources: chebi, enzyme_explorenz, enzymemap, metanetx, pubchem, retrorules, rhea
  enzymology_evidence          6 sources: brenda, enzengdb, oed, retrobiocat_db, sabio_rk, strenda_db
  sequence_family_evolution   11 sources: akr_superfamily, cath_funfam, eggnog, interpro, mgnify_proteins, ncbi_protein, pfam, sdred, uniparc, uniprotkb, uniref
  structure_and_mechanism      6 sources: alphafill, alphafold_db, mcsa, rcsb_pdb, sifts, wwpdb_ccd
  mutation_and_performance    10 sources: catpred_db, enzengdb, esibank, fireprotdb, intenzydb, mavedb, protabank, proteingym, reactzyme, skid
  literature_and_feedback     10 sources: bacdive, enzchemred, enzymeml, equilibrator, europe_pmc, kegg, machine_literature_extraction, metacyc_biocyc, pubmed, zenodo
  needing curation: 49
  recording an endpoint: 0
  network mode but no endpoint: 26
  human-import only: 6 (akr_superfamily, oed, protabank, retrobiocat_db, sdred, strenda_db)
  13 of 28 source group(s) countable as independent after lineage collapse
  lineage admittedly incomplete: 23
```

`eagent sources show brenda` goes further, and the `not good for` section is the
reason the registry exists:

```
evidence ceiling:        ec_species_mapped
licence:                 CC BY 4.0
licence source:          Stated by the project owner as their current information in the brief
                         for this registry. NOT independently verified against the resource's own
                         terms, which is why needs_legal_review stays true and
                         redistribution_allowed stays unknown.

not good for
------------
  - A record is frequently tied to an EC number and an organism rather than to a specific
    sequence, so it must not be ingested as sequence-level evidence ...
  - Values are extracted from heterogeneous papers under different assays, so kinetic
    numbers from different references must not be pooled onto one scale.
  - Substrate strings are often prose names without stereochemistry, which is fatal for
    an asymmetric reduction task unless re-resolved to a structure.
  - Reaction direction is often implicit, so a reported activity may be an oxidation
    measurement in a resource read as reduction evidence.
```

---

## 12. Summary: where the run stopped, and what is still null

| Stop | Stage | Exit | Why | What unblocks it |
| --- | --- | --- | --- | --- |
| 1 | `awaiting_human` at `await_reaction_approval` | 4 | 4 blockers, 4 unresolved fields; the reaction spec is not confirmed | an operator resolves the four fields with a named authority, then `eagent approve reaction_spec_confirmed` |
| 2 | `awaiting_human` at `retrieve_evidence` | 4 | 0 records from 12 planned queries; empty evidence base, **not** a negative result | a curator places the 12 cached imports named in `evidence_gaps.tsv` |

**Nulls still unresolved at both stops:**

| Field | Who must supply it | Why nothing may guess |
| --- | --- | --- |
| `reaction.substrate.isomeric_smiles` | operator (chemist) | the substrate is the index of the whole search |
| `reaction.product.isomeric_smiles` | operator | the assay has to detect a specific structure |
| `reaction.product.creates_new_stereocenter` | operator, or a sourced reaction template | an aldehyde or symmetric ketone gives an achiral alcohol, and then no ee may be reported |
| `reaction.atom_mapped_reaction_smiles` | operator, or a reviewed atom mapper | it ties the reactive atoms to the geometry checks |
| `reaction.product.target_stereochemistry` | operator | a commercial decision; guessing it inverts the objective |
| `conditions.*` (pH, T, solvent, cosolvent, buffer, host, loading, time) | operator | part of the record **key**, not metadata |
| `conditions.cofactor_options` | operator | decides what the `no_cofactor` control means |
| `reaction.rhea_id` | curator | an invented identifier would be used as a join key |
| 17 geometry windows' `calibrated_on` | curator, from experimental complexes | an unfitted window may not reject a candidate |
| every datasource endpoint and licence | curator | 0 of 49 connectivity-tested |

Nothing in this run produced a number that was not read from a file, and no
field was filled by anything other than a named human decision.

---

## 中文摘要

### 这份走查演示的主要是"拒绝"

出厂试点任务 `KRED_PILOT_001`（酮不对称还原成手性仲醇）跑起来会**停两次**，最后产出一个
**自述不完整**的交付包。这就是系统正常工作的样子。如果走查里流水线一路畅通跑到 96 个构建体，
那只能说明每一个 null 都被什么东西填上了——而本仓库里没有任何东西被允许这么做。

**下面所有输出都是真实执行的结果**（2026-10-02），除了一处被标注出来的、重复的 `[MISSING]`
行省略之外，没有任何编辑。

### 第 1 站：`validate` 退出码 3

三个闸门分别列出未解析字段；并明确指出 YAML 里的 `false` **不是"拒绝批准"，而根本不是批准**
——闸门只能由 `eagent approve` 清掉，并把"谁、什么时候决定的"写进运行清单。助记：
"a flag in a file is not an approval"。

### 第 2 站：`--dry-run` 什么都不执行

四行信息承载了大部分设计：每次运行开头都打印 **"17 of 17 窗口未标定"**（不是藏在 lint 里）；
漏斗数字标注为 **"planned scale (from the task file, not a prediction)"**；成本块拒绝编造
总额（"计划里的一个数字会被当作已核对过而获批"）；**dry run 不写运行清单**——为一次没发生的
运行写清单，会被读成它发生过。

### 第 3 站：第一次停机，退出码 4

四个 blocker：没有底物、没有产物、立体中心未定、没有原子映射。系统**本可以但没有**：
把"常用的那个模型酮"解析成苯乙酮（名字固定不了互变异构体、成盐形式和立体化学）；从
`reaction_class` 推断 `creates_new_stereocenter: true`（醛和对称酮都给非手性醇，而反应类别
标签本身就是操作者的断言，而且往往正是错的那个）；默认 NADPH（辅因子决定了 `no_cofactor`
对照意味着什么）；或者干脆继续往下跑（后面每一步都会是关于一个未确认分子的证据）。

注意 `normalize_reaction` 返回的是 **`partial` 而不是 `failed`**——它完成了它的工作，即分诊；
而且它**仍然写出了** `reaction_spec.yaml`，因为规格和它的 QC 记录必须在同一个文件里：
"半年后在没有 QC 记录的情况下读到的、看起来完整的规格文件，与一个靠猜补全的规格文件无法区分。"

### `status`：成本不打印 0.0

"nothing was recorded by any step. **This is not a claim that the run was free**:
it means no step reported a cost."

### `approve`：在你拍板之前先告诉你缺什么

决策回答的是一个**具体请求**（打印请求 id、谁提的、什么时候提的）；决策**之前**先打印
"如果搞错了会怎样"；并且列出这个请求**没有携带**的那些 `must_show` 字段，附一句
"只有当你从别处掌握这些事实时才做决定"。批准会通过，带着操作者的姓名和理由。系统不假装比一个
具名的人更懂，它只确保这个人知道自己在签什么。授权写进**清单**，不写进任务文件。

### 第 4 站：第二次停机——最重要的那次拒绝

十二条查询全部落空（缓存为空、网络禁用）。一个不够小心的流水线在这里会报
"没有已知的酶能做这个反应"。这个系统报的是：

> **这是一个空的证据库，不是一个阴性结果：下游任何环节都不得把它当作"没有酶能做这个反应"的
> 证据。**

缓存未命中是关于**这台机器**的事实，不是关于文献的事实。另外 `layers_not_searched` 是一条
**info** 标记：PDB 和 AlphaFold 没被查，原因写明（结构检索以登录号为键，而登录号要等挖掘步骤
跑完才存在），并写明什么能解锁它们。**被跳过的层被报告为"跳过"，而不是悄悄从计划里消失。**

`evidence_query_plan.yaml` 在**检索之前**就写好了——召回率无法从结果判断，只能从查询判断，
这个文件的存在就是为了让人能说"你压根没搜环酮"。`evidence_gaps.tsv` 是一份**策展工作清单**
（每行给出连接器、确切查询、以及放置该文件的那一行调用），不是错误日志。
`evidence_matrix.tsv` 即使为空也可读，因为它的图例先定义了四组被普通筛选表毁掉的区分：
野生型成功 vs 工程化成功、目标方向 vs 逆方向、测过为阴 vs 表达失败 vs 没测过、
**独立来源数** vs 行数。

### `verify`：空的校验不是通过的校验

退出码 3："an empty verification is not a passed one"。在空输入上输出"校验通过"，是一个校验
步骤所能给出的最危险的结果。

### `bundle`：一个说得清自己缺什么的交付包

18 个声明项里 4 个在、14 个缺；**退出码 0**——组装一个不完整的包不是错误，**隐瞒**它不完整才是。
每个缺失项都带"它是干什么用的"和"策展者要做什么"，而 `confidence_metrics` 事先声明了自己的
拒绝：一张置信度曲线截图或一个平均数**不能**替代逐残基文件。

三条打包规则：mmCIF 是记录，旁边的 PDB 只是**经过校验的附赠**（只有转换后 PDB 的目录会被报为
"缺少原始记录"）；**文件名不得声称文件里没有的数字**（叫 `selected_batch_96.csv` 却只有 71 行，
到下次开会就变成"我们筛了 96 个"，所以文件按实际行数改名并记录改名）；**partial 不是 present**。

### Mode B：拒绝给出"最好的酶"

与 A 模式的两个差别就是全部要点：这里 `substrate_unspecified` 和 `product_unspecified`
**不是** blocker（B 模式允许没有底物），所以是两个 blocker 而不是四个；
`best_enzyme_claim_refused` 被记成**记录上的 INFO 标记**，于是这条拒绝存在于**产物里**，
而不只是终端上。写进 `reaction_spec.yaml` 的拒绝对象包含三部分：被拒绝的请求、拒绝的理由
（活性与立体选择性是"酶—底物对"的性质，底物未定就没有可排序的对象）、以及解锁条件
（选一个子空间和一个代表性底物结构）。取代排名的是四个化学子空间，各带定义、需要做的决策、
为什么重要、要取什么证据、以及一个代表性问题。

### 两次停机与仍然为 null 的字段

| 停机 | 阶段 | 退出码 | 原因 | 解锁方式 |
| --- | --- | --- | --- | --- |
| 1 | `await_reaction_approval` | 4 | 4 个 blocker、4 个未解析字段 | 操作者用具名权威解析四个字段，然后 `eagent approve` |
| 2 | `retrieve_evidence` | 4 | 12 条查询 0 条记录；空证据库，**不是**阴性结果 | 策展者放入 `evidence_gaps.tsv` 点名的 12 份缓存导入 |

仍为 null：底物与产物结构、是否产生新立体中心、原子映射、目标构型、全部反应条件
（它们是记录**键**的一部分，不是元数据）、辅因子选项、RHEA 标识符、17 条几何窗口的
`calibrated_on`、以及全部 49 个数据源的端点与许可（**0 / 49 做过连通性测试**）。

这次运行里**没有产生任何不是从文件读出来的数字**，也**没有任何字段是被人工具名决策以外的东西
填上的**。
