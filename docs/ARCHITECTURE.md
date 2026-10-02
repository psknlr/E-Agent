# Architecture

How E-Agent is put together, and why each boundary is where it is.

**Scope and status.** Everything in this document exists as code under
`src/eagent/`. Nothing in it has been run against a live public database, a real
structure predictor, a real docking program or a real plate. Where this document
says "the system refuses X", it means there is a code path that raises or returns
a typed refusal and a test that exercises it — not that the refusal has fired on
real data. The components that are adapter seams are named as such in
[§8](#8-what-is-a-seam) and in `docs/ROADMAP.md`.

---

## 1. The shape: one controller, structured tools, an independent verifier, a feedback interface

```
                      ┌──────────────────────────────────────┐
   operator ────────► │  TaskSpec  (nulls are decisions)     │
                      └──────────────┬───────────────────────┘
                                     │
        ┌────────────────────────────▼─────────────────────────────┐
        │  ResearchController   (harness/controller.py)            │
        │  a declared state machine: stages, legal moves, retries  │
        │  decides WHAT RUNS NEXT. Computes nothing scientific.    │
        └───┬─────────────────────┬──────────────────────┬─────────┘
            │                     │                      │
            │ runs                │ queues               │ reads
            ▼                     ▼                      ▼
  ┌──────────────────┐  ┌───────────────────┐  ┌────────────────────┐
  │ ten interfaces   │  │ ApprovalQueue     │  │ RunManifest        │
  │ (tools/*.py)     │  │ three gates,      │  │ append-only record │
  │ one contract,    │  │ named actors,     │  │ hashes, versions,  │
  │ one envelope     │  │ persisted to disk │  │ seeds, cost        │
  └────────┬─────────┘  └───────────────────┘  └────────────────────┘
           │ artifacts on disk
           ▼
  ┌────────────────────────────────────────────────────────────────┐
  │ IndependentVerifier (harness/verifier.py)                      │
  │ re-derives from the PRIMARY material: the sequence string,     │
  │ the coordinate file, the template's own windows, the pose's    │
  │ restraint list. Never reads the producing step's verdict.      │
  └────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────────────┐
  │ experiment feedback: select_batch writes the plate map and the │
  │ pre-registered criterion BEFORE the plate runs; ingest_results │
  │ reads the plate back against that one criterion and no other.  │
  └────────────────────────────────────────────────────────────────┘
```

A language model is used in exactly one place — `harness/llm.py` — and for
exactly four jobs: **understand** the task, **organise** retrieval, **propose**
falsifiable hypotheses, **interpret** evidence into a plan. Everything
measurable is computed by deterministic code.

### Why not a committee of role-playing agents

A "chemist agent", a "structural biologist agent" and a "critic agent" debating
in natural language is an appealing picture and a bad instrument, for four
reasons that are specific rather than stylistic:

1. **Agreement between role-played agents is not independent corroboration.**
   The same model, with the same training data and the same prior, is asked the
   same question in three costumes. When they agree, nothing has been checked;
   the output is one opinion rendered three times, and it reads like consensus.
   Real independence here means *different physical bases for error* — an
   experimental structure versus a predicted one, template docking versus joint
   co-folding (`science/robustness.cross_method_agreement`), a measurement
   versus the template window it is compared against.

2. **A critic agent cannot verify a number it did not compute.** Review in prose
   checks whether a claim *sounds* defensible. The verifier here checks whether
   the coordinate file actually contains the sequence the report pairs with it,
   whether the ligand code in the pocket is the reduced cofactor the template
   demands, whether a residue token round-trips through the numbering map. Those
   are recomputations, not opinions, and a conversational critic has no access
   to them.

3. **A debate has no declared termination and no auditable path.** This
   controller's legal moves are a table (`TRANSITIONS`). Six months later,
   "how did this run get from modelling to ordering genes" is answered by
   reading that table and the run's recorded `path`, not by re-reading a
   transcript. An undeclared move raises.

4. **Diffuse responsibility.** When three agents converge on a wrong substrate,
   no artifact says who decided. Here the three decisions that matter are
   gates with a named human behind each one, and the decision, the actor, the
   time and the payload they were shown are in the manifest.

So: one controller that plans and interprets; deterministic code or a dedicated
scientific model for everything measurable; one verifier that re-derives rather
than agrees; and a feedback interface that closes the loop on real data.

---

## 2. The model boundary, and how it is enforced

The split is not requested in a prompt — prompts are advisory and the
characteristic failure of a research agent is not refusing to answer, it is
answering with a number that reads like a measurement and was never measured. So
the boundary is a guard on the output.

| The model may | The model may not |
| --- | --- |
| Read the brief and say what is still undecided | Decide it |
| Plan which layers and resources to query | Produce a database record |
| Propose a hypothesis **with the experiment that would refute it** | Assert a quantity |
| Interpret a results table into a next step | Compute an alignment, a coordinate, an atom mapping, a distance, an angle, a confidence, a docking score or a statistic |

**`NumericGuard`** (`harness/llm.py`) inspects every model response. A line
containing a quantity with a measurement-shaped unit — `%`, `Å`, `nm`,
`kcal/mol`, `s⁻¹`, `mM`/`µM`/`nM`, `pLDDT`, `ee`, `kcat`, `Km`, degrees — must
also carry an artifact citation of the form `[artifact:<name>]`,
`[table:...]`, `[file:...]`, `[record:...]` or `[evidence:...]`. If it does not,
the guard raises `FabricationGuardError` naming the offending tokens:

> "the hydride transfer distance in `[artifact:catalytic_geometry.tsv]` is 3.6 Å"
> — allowed; the value came from a file the verifier can open.
>
> "the transfer distance is about 3.6 Å" — refused; that is an estimate wearing
> the clothes of a measurement.

**`ModelTurn`** restricts a response to three shapes and nothing else:
`tool_calls` (only against registered interfaces), `hypotheses` (each needing
both a `test` and a `would_falsify`, or `validate_turn` rejects it as not
falsifiable), and `questions` (for decisions only a human may make). An
unparseable response becomes a turn carrying the raw text as a *question* rather
than an exception, so a long run surfaces it to the operator instead of crashing.

**`LLMClient`** is provider-agnostic and must not retry silently.
`EchoClient` is a deterministic offline client used in tests and dry runs,
deliberately on the executed path so the parsing and guarding code is exercised
by the test suite rather than only in production.

The controller itself contains no science at all. There is no distance, no
score, no confidence and no ranking computed anywhere in
`harness/controller.py`; every number it touches came out of a `ToolResult` or
out of the manifest's own bookkeeping. A single `if plddt > 70` there would
plant a threshold no template authorised and no reviewer would find.

---

## 3. The ten interfaces, and the contract they share

Every interface subclasses `ScientificInterface` (`tools/base.py`) and declares:

```python
name: ClassVar[str]                        # stable id, used in the manifest
description: ClassVar[str]
required_fields: ClassVar[tuple[str, ...]] # task fields that must be resolved
required_approvals: ClassVar[tuple[str,...]]  # gates that must be cleared
depends_on: ClassVar[tuple[str, ...]]      # interfaces whose artifacts it reads
version: ClassVar[str]                     # recorded in provenance
def execute(self, ctx: RunContext, **kwargs) -> ToolResult: ...
```

`run()` is a fixed wrapper the subclass does not override. It checks
preconditions (turning an unresolved field or an uncleared gate into a
*structured result* rather than a traceback, so the controller can route it to
the operator), calls `execute`, converts `ToolUnavailableError`, `LicenseError`
and every other `EAgentError` into a typed failure envelope, records an
unexpected exception **with its traceback** rather than masking it, and stamps
provenance: tool name, version, start and finish times, and a per-step seed
derived from the run's global seed.

| # | Interface | Depends on | Approvals | What it produces |
| --- | --- | --- | --- | --- |
| 1 | `normalize_reaction` | — | — | `reaction_spec.yaml`: the typed spec **plus the list of what is still wrong with it**. Never completes the spec itself. |
| 2 | `retrieve_evidence` | `normalize_reaction` | — | The query plan (written *before* retrieval), the family × chemotype evidence matrix with outcome classes kept apart, typed records, and `evidence_gaps.tsv` naming the exact cache file each gap needs. |
| 3 | `mine_sequences` | — | — | A candidate pool from **several** characterised seeds, with retrieval provenance per sequence. Refuses a single seed unless explicitly authorised with a justification. |
| 4 | `annotate_family` | `mine_sequences` | — | A family call per sequence built from four *separable* signals, never from one motif, plus the sequence-similarity network as an exploration aid. |
| 5 | `prepare_structures` | `annotate_family` | — | A chosen structure per candidate **with the reason**, a pocket-local confidence, and `residue_atom_mapping.tsv`. |
| 6 | `model_complexes` | `prepare_structures` | — | Enzyme–substrate–cofactor poses by two routes that fail differently; every pose kept; every restraint recorded. |
| 7 | `evaluate_catalysis` | `annotate_family`, `model_complexes` | `reaction_spec_confirmed` | Per-pose mechanism evaluation against the family's sourced template, the per-candidate scorecard (no total), and the explanations a reviewer reads instead of a score. |
| 8 | `select_batch` | `evaluate_catalysis` | `synthesis_authorized` | The order form, `experiment_plan.yaml` with the measurement footprint and the **pre-registered** positive criterion, and the empty results template. |
| 9 | `ingest_results` | `select_batch` | `functional_criteria_confirmed` | Typed records across all seven outcome classes, the active-learning partition with its arithmetic checked, the layer update, and — when nothing turned over — a structured no-hit differential. |
| 10 | `propose_mutations` | `evaluate_catalysis`, `ingest_results` | — | Variants from a confirmed parent, with per-axis expectations stated before testing and the catalytic machinery frozen. |

The **protocol order** (`harness/registry.PROTOCOL_ORDER`) is the list above. It
is written down rather than derived from `depends_on`, because `depends_on` only
records data flow: nothing links `normalize_reaction` to `mine_sequences` by an
artifact, yet the whole search is defined by the spec, so a controller walking
the topological order alone would be free to mine before the reaction was
normalised.

`build_interface_registry(strict=True)` refuses to build a registry missing any
of the ten. Discovery answers "what is here"; a run needs "is everything the
protocol requires here", because silently executing nine of ten steps produces a
batch plan that looks complete and nothing downstream can tell that the geometry
step never ran. `dependency_problems()` additionally reports a declared
dependency that is not registered, since such a step would run anyway and report
"no candidates supplied", which reads like an empty search result rather than
like a missing step.

> **An inaccuracy worth naming.** `eagent/tools/__init__.py` still carries a
> planning-era `INTERFACE_MODULES` table listing six names that no module
> implements (`map_catalytic_site`, `build_complex`, `screen_geometry`,
> `rank_candidates`, `design_variants`, `plan_batch`). A bare
> `build_registry()` therefore attaches those six to `missing_interfaces`.
> Module discovery is authoritative and finds the ten real interfaces, and the
> controller's path (`build_interface_registry`, which names
> `PROTOCOL_ORDER` explicitly) reports nothing missing. The stale table is
> dead weight, not a live gap.

### The uniform result envelope

Every interface returns the same `ToolResult` (`envelope.py`), which is what lets
the controller route without knowing any chemistry:

| Field | Meaning |
| --- | --- |
| `status` | `success` / `partial` / `failed`. `partial` is a first-class state: "it ran, it produced something usable, and something is missing". |
| `artifacts` | `Artifact(key, path, kind, sha256, n_records, summary)` — what landed on disk. |
| `provenance` | tool, version, input hashes, database snapshots, model versions, parameters, seed, timings, cost. |
| `qc_flags` | `QCFlag(code, severity, message, subject)` at `info` / `warn` / `blocker`. |
| `uncertainty` | `Uncertainty(code, question, affects, resolvable_by)` — **a first-class field, not a free-text note**, so a step that produced an answer it cannot defend says so in a place the controller can branch on. |
| `next_actions` | `NextAction(action, rationale, params, requires_human)` — concrete and executable, not a vague suggestion. |
| `data` | structured payload for the next step. |

`result.ok` means *usable and free of blocking QC problems* — status alone is not
enough, because a `partial` carrying a blocker is not something to build on.

---

## 4. Control flow: stages, back-edges, branches, termination

`harness/controller.py` is an explicit state machine. The stages are an enum, the
legal moves are a table, and `goto()` raises on an undeclared move:

```
start
  → confirm_reaction_spec ──────────────► await_reaction_approval ─┐
                                                                   │ (granted)
  retrieve_evidence → mine_sequences → annotate_family ◄───────────┘
      → prepare_structures → model_complexes → evaluate_catalysis
            ├── input or mapping error ──► repair_inputs ──► (the stage that failed)
            ├── insufficient evidence ───► widen_retrieval ─► (re-run, or carry forward
            │                                                 as exploration probes)
            └── sufficient support ──────► batch_approval ──► select_batch
                  → await_results → ingest_results
                        ├── hit    ──► local_engineering ──► done
                        └── no hit ──► diagnose_no_hit ────► done
```

**Going back is a declared transition, not a recursive call.** The manifest has
to be able to show that the run returned to `prepare_structures` after a repair,
and a stack frame cannot be written to a JSON file. `REPAIR_INPUTS` and
`WIDEN_RETRIEVAL` each declare every stage they may return to.

**Five terminal stages, not one.** `done`, `escalated`, `halted`,
`awaiting_human`, `awaiting_results`. "Waiting for your decision" and "this
failed and needs you" are different messages, and collapsing them into
`finished` loses the one thing the operator needs.

### Failure is classified before it is retried

`classify_failure()` reads only the envelope — status, QC codes and whether the
step itself asked for a human — never the science. Seven kinds, with
deliberately asymmetric policy:

| Kind | Attempts | Action | Why |
| --- | --- | --- | --- |
| `INPUT_ERROR` | 2 | repair and re-run | A malformed or missing input can be repaired once. The same error after a repair means the repair is not addressing it. |
| `INSUFFICIENT_EVIDENCE` | 2 | widen inputs | Widening once is cheap; widening repeatedly is a search for a result that may not be there. |
| `NEEDS_HUMAN` | 1 | await human | Re-running reproduces the same question. |
| `TOOL_UNAVAILABLE` | 1 | escalate | A binary that is not installed will not be installed by running the step again, and reaching for a different model substitutes a guess for a measurement. |
| `POLICY_BLOCKED` | 1 | escalate | A licence or disclosure refusal is a decision, not a transient fault. |
| `INTERNAL_ERROR` | 1 | escalate | An unhandled exception is a defect in this codebase; retrying hides it. |
| `UNCLASSIFIED` | 1 | escalate | An unrecognised code has no known safe recovery, and guessing one is how an agent loops. |

Order matters inside the classifier: a step that is both blocked on a human and
short of evidence routes to the human, because widening the search while the
operator has not confirmed the substrate searches harder for the wrong thing.
Both caps apply — the per-kind cap and `ctx.policy.max_retries` — and the tighter
wins, so `max_retries=50` still does not buy fifty attempts at a missing binary.

**Widening means more seeds, another database, a further family. It never means
a lowered threshold**, which is the same step with the evidence removed. When
widening is exhausted, the candidates that exist are carried forward as
*exploration probes* — a declared use of batch slots — and the carry-forward is
written into the manifest rather than presented as sufficiency.

**Repair and widen are hooks, not behaviours** (`ControllerHooks`). The
controller knows that an input error should be repaired; it does not know what
the right structure index or the right seed set is, and guessing would make it
the author of a scientific choice belonging to the operator. A hook that reports
it changed nothing causes an escalation rather than a loop, because a re-run
after a no-op repair reproduces the identical failure.

**Termination** has three independent guards: the stage table (no undeclared
move), `max_transitions` (an oscillating controller is a control-flow defect and
must surface as an escalation, not as a run that never returns), and
`budget_breaches()` — which reads what was *actually recorded* against
`policy.cost_ceiling` and never estimates what a step will cost.

### Next actions are routed, never executed off the envelope

A `requires_human` action goes to the approval queue even when the harness could
technically do it, because the whole reason it is marked is that a person must
decide. A non-human action naming a registered interface is recorded as a
*suggestion*: the controller still has to put it in the right place in the state
machine, and executing it straight off the envelope would skip the gates in
between. An action that is neither is recorded in `unroutable_actions` for the
operator rather than guessed at.

### Resume

Each executed step records a digest over its interface name, the task input hash
and its rendered arguments. On resume, a step is skipped **only** when the digest
matches, the digest is provably stable (an argument whose `repr` carries a memory
address renders to an `__unstable__` marker and forces a re-run), and the
previous attempt was one the controller itself classified as clean. A record
routed to widening or repair last time is not "done", and a record written before
this field existed cannot prove either way, so it is re-run.

---

## 5. The three approval gates

`harness/approval.py`. The three are the only ones; `ApprovalQueue.request`
raises on a fourth spelling, because a gate nothing ever checks is worse than no
gate. Each `DecisionPoint` carries the question, the fields the operator **must
be shown**, and the consequence if the decision is wrong.

| Gate | Must show | Consequence if wrong |
| --- | --- | --- |
| `reaction_spec_confirmed` | substrate SMILES, product SMILES, target configuration, `creates_new_stereocenter`, the atom-mapped reaction SMILES, the assumption ledger | Every sequence mined, every complex modelled and every gene ordered afterwards is evidence about the wrong molecule |
| `synthesis_authorized` | constructs, candidate wells, control wells, plates, cofactor conditions, replicates, the control list, cost | Money and lab time spent on a round whose composition nobody reviewed |
| `functional_criteria_confirmed` | assay template id, tier, method, whether it identifies the product, the positive criteria, the detection limit, the required controls | The criterion gets fitted to whatever the plate produced, and the round has no endpoint |

Mechanics that make them real rather than decorative:

- **A request states what is being decided.** `_request_id` hashes the gate, the
  payload *and* the request time, so re-asking with a changed batch produces a
  new request instead of inheriting the grant given for the old one. There is no
  way to approve a gate in the abstract.
- **An anonymous decision is refused.** `grant`/`deny` require a non-empty actor.
- **A decision cannot be overwritten.** Re-deciding a finalised request raises;
  the record of what was agreed stands.
- **The queue is persisted**, because a batch authorisation is requested on
  Monday and granted on Thursday, after the run has been stopped and resumed
  twice.
- **`guard_batch_selection` is a hard block** in front of `select_batch`, checked
  by the controller *before* it will route there at all. It is deliberately
  stricter than `RunContext.require_approval`, which also accepts the task-file
  flag: the flag says somebody wrote `true` in a file; the manifest grant says a
  named person said yes to a specific batch at a specific cost.
- **The CLI prints the gap.** `eagent approve` lists the `must_show` fields the
  request does **not** carry and tells the operator to decide only if they have
  those facts from elsewhere.
- **The criteria gate is checked before the plate is read**, not after. The
  controller stops at `ingest_results` with "reading the plate first would let
  the criterion be fitted to the data".

`batch_cost_payload()` deserves a note: prices default to `None` with an explicit
note, and a total is computed **only** when every price it needs is present.
Treating a missing plate price as zero would quote a total that is confidently
too low — the shape of mistake an authorisation request must not make.

---

## 6. The independent verifier

Independent means one specific thing: `harness/verifier.py` does not read the
producing step's verdict. It re-derives what it checks from the primary material
and compares. A verifier that reads `GeometryReport.gating_passed` and agrees
with it has verified nothing; it has restated the claim in a second voice, which
is worse than no check because it reads like corroboration.

The fields it must never read are listed in `IGNORED_PRODUCER_CONCLUSIONS`, with
the recomputation that replaces each, and the list is **asserted in the tests**,
so a later edit that "simplifies" a check by trusting one of them fails.

| Check | The failure it exists for |
| --- | --- |
| sequence ↔ structure pairing | Candidate C paired with a structure of a different protein. Pocket residues, mutation numbering and geometry are then all measured on the wrong molecule and look perfectly self-consistent. |
| ligand identity and chirality | The docked molecule is not the one in the spec, or carries a different configuration; the run optimises for a compound the project is not about. |
| cofactor type and oxidation state | NADP⁺ where the template requires NADPH. An oxidised cofactor has no hydride to donate, so a "competent" complex built around one is a picture of a reaction that cannot occur. |
| residue numbering round-trip | An off-by-a-His-tag mapping. The variant expresses, folds, shows nothing, and the conclusion drawn is "the hypothesis was wrong" rather than "we built the wrong protein". |
| claims backed by artifacts and provenance | A sentence in a report that nothing on disk supports. It cannot be rechecked and it will outlive the run. |
| numeric ee without a calibration source | An ee is a measurement. A predicted one must name what it was calibrated on, or it is a guess formatted as data. |
| disqualification on an uncalibrated window alone | The pool shrinks, the shrinkage looks principled, and the criterion was never evidence. |
| restrained constraints counted as independent evidence | A distance enforced during modelling cannot afterwards corroborate the model that enforced it. |
| an experimental-negative label on a computational failure | A crashed docking run written into the record layer as "no activity detected" is a fabricated experimental result. |

What it could not check goes into `VerificationReport.unverifiable` rather than
being silently omitted, and `eagent verify` refuses to report a pass when there
was nothing to check at all: "an empty verification is not a passed one".

Circularity has a second enforcement point upstream.
`model_complexes.assert_restraints_recorded` raises `CircularEvidenceError`
rather than emitting a pose whose restraints were silently forgotten — a
forgotten restraint is indistinguishable from an honest measurement in every
later artifact — and `science.robustness.CircularityGuard` subtracts the
restrained constraints from the independent-evidence count.

---

## 7. Reproducibility

`RunManifest` (`provenance.py`) is append-only and is written to the run
directory. A candidate that reaches a synthesis order must be traceable back
through it to the sequence record and the evidence that justified it.

| Recorded | Where | Why it is not optional |
| --- | --- | --- |
| **Input hashes** | `task_input_sha256`; `Provenance.inputs_sha256` per step; the controller's own `controller_step_inputs` digest | "Was this run on the spec I approved?" must be answerable without trusting a file's mtime. |
| **Sequence identity** | `sequence_hash()` — SHA-256 of the whitespace-stripped, uppercased sequence, used as the join key everywhere | An accession is not an identity: accessions get re-annotated and isoforms share names. |
| **Database snapshots** | `Provenance.databases` (name → snapshot/version), aggregated into `RunManifest.databases` | A hit list is reproducible only against a named snapshot. |
| **Software and model versions** | `Provenance.tool_version` per step, `Provenance.models` (e.g. `{"rdkit": "absent"}`), `capture_environment()` with the Python version, platform, host, cwd and git commit | A version written from memory in a manifest is worse than no manifest, because it looks like a fact. `__version__` resolves from installed distribution metadata, falling back to the number declared in `pyproject.toml`, and is never guessed. |
| **Seeds** | `RunManifest.global_seed`; `derive_seed(global_seed, step_id)` per step | A per-step derivation means re-running one step does not shift the others' randomness. |
| **Numbering maps** | `residue_atom_mapping.tsv`, candidate index ↔ author numbering, residue by residue, rebuilt by alignment for every structure | The one thing no downstream step can detect is that the residue being measured is not the residue that was meant. |
| **Failure logs** | every attempt appended to the manifest *before* any routing decision; `StepAttempt` per execution; `Escalation` with the reason; unhandled exceptions carry their traceback in `result.data` | A crash between the call and the decision still leaves the evidence that the call happened. |
| **Approvals** | gate, decision, actor, detail, timestamp | See [§5](#5-the-three-approval-gates). |
| **Cost** | `Provenance.cost` per step (cpu\_s, gpu\_s, api\_tokens, currency), summed into `RunManifest.cost_total` | A ceiling with nothing recorded against it prints "nothing recorded", not `0.0`. Those are different facts. |
| **Template calibration state** | `TemplateLibrary.calibration_report()` printed at the head of every run | A report must be able to say "17 of 17 constraints uncalibrated" rather than presenting a geometric verdict as though it were grounded. |

Beyond the manifest, `datalayer/snapshot.py` freezes the *data* a round was
decided on — source files at a checksum, the cleaning rules in order, the records
dropped and by which rule — so that "round 2 has a different hit rate: was that
the method or the data?" is answerable. `deliverables/bundle.py` then assembles
the package and `verify_bundle` re-checks, anywhere, that every declared file is
present, hashes to what the manifest recorded, and that no unrecorded file has
appeared inside a declared directory.

---

## 8. What is a seam

A **seam** is a boundary where the contract, the validation, the provenance, the
artifact layout and the refusal path are implemented and tested, and the external
program is absent. A seam does not degrade into a heuristic. It returns
`ToolUnavailableError` → a `failed` envelope with code `tool_unavailable` → the
controller escalates without retrying.

- `mine_sequences`: `BlastpAdapter`, `MMseqs2Adapter`, `HmmsearchAdapter`
  resolve their executable through an injectable `executable_finder` and run it
  through an injectable `CommandRunner`. No binary is present here.
- `prepare_structures`: `StructurePredictor` is a `Protocol`; the default is
  `UnavailablePredictor`.
- `model_complexes`: `DockingRunner` and `ComplexPredictor` are `Protocol`s;
  the defaults are `UnavailableDockingRunner` and `UnavailableComplexPredictor`.
- `propose_mutations`: `LigandMPNNAdapter` is abstract; the default is
  `MissingLigandMPNN`.
- Every public resource: `connectors/base.OfflineConnector` is cache-first, a
  miss is a structured `MISS` naming the exact file a human must place, and **no
  code path in the package produces database content**.
- `eval/baselines`: `family_function_prediction` and
  `substrate_specificity_model` return `RankedSelection` with
  `unavailable_reason` set and no picks, and `BaselineComparison.render` lists
  them by name — a stand-in baseline that the agent then beats is the most
  flattering possible result and means nothing.

Three policy guards sit on the same boundary. `ExecutionPolicy.allow_network` is
false by default. `connectors.base.looks_like_biological_sequence` plus
`AccessPolicy` raise `UnauthorizedSubmissionError` when a sequence-shaped string
appears in an outbound payload without either a named human authorisation
covering that exact sequence hash or evidence that the sequence is already
public — an unreleased construct sent to a remote service is disclosed
irreversibly, and no later policy decision undoes it. And `check_license` refuses
a commercial run against any tool facet whose terms are unknown, because an
unverified licence is not a permission.

---

## 中文摘要

### 一个控制器 + 结构化工具 + 独立校验器 + 实验反馈接口

整个系统只有**一个**控制器（`harness/controller.py`），它是一台显式状态机，只决定
"下一步跑什么"，不计算任何科学量。十个接口（`tools/*.py`）共用一套契约和一个统一信封。
一个独立校验器（`harness/verifier.py`）从**原始材料**重算。实验反馈接口由
`select_batch`（在上板前写下板图与预注册判据）和 `ingest_results`（只按那一个判据读回）
构成闭环。

语言模型只出现在 `harness/llm.py` 一个地方，只做四件事：理解任务、组织检索、提出可证伪
假设、把证据解读成下一步计划。

### 为什么不是一群扮演角色的智能体在开会

1. **角色扮演之间的一致不是独立佐证。** 同一个模型、同一套训练数据、同一个先验，换三身
   行头回答同一个问题。它们一致时什么也没被检验，输出是一个意见渲染了三遍，却看起来像共识。
   这里的"独立"指**误差来源在物理上不同**：实验结构 vs 预测结构、模板对接 vs 联合折叠、
   测量值 vs 它所比对的模板窗口。
2. **一个"批评家"智能体无法校验它没有算过的数字。** 文字评审检查的是一个说法**听起来**
   站不站得住。这里的校验器检查的是：坐标文件里的序列是不是报告里配对的那条、口袋里的配体
   代码是不是模板要求的还原态辅因子、残基编号能不能在编号映射里往返。这些是重算，不是意见。
3. **辩论没有声明过的终止条件，也没有可审计的路径。** 这里的合法状态转移是一张表
   （`TRANSITIONS`），未声明的转移直接抛错。半年后问"这次运行是怎么从建模走到下单基因的"，
   答案来自这张表和运行记录里的 `path`，而不是重读一段对话。
4. **责任弥散。** 三个智能体一起选错底物时，没有任何产物说得清是谁决定的。这里只有三个
   闸门，每个后面都有**具名的人**，决策、决策人、时间、以及他当时看到的内容都在清单里。

### 模型与确定性代码的分界线，以及它是怎么被强制的

分界线不是写在提示词里的——提示词只是建议，而研究型智能体最典型的失败不是拒答，而是给出
一个读起来像测量值、实际从未被测量的数字。所以强制点在**输出端**：

`NumericGuard` 检查每一行模型输出。凡是带"测量单位"的数量（`%`、Å、nm、kcal/mol、s⁻¹、
mM/µM/nM、pLDDT、ee、kcat、Km、度），同一行必须带 `[artifact:...]` 之类的产物引用，
否则抛 `FabricationGuardError`。"在 `[artifact:catalytic_geometry.tsv]` 里该氢负离子转移
距离是 3.6 Å"可以；"该转移距离大约 3.6 Å"不行。

`ModelTurn` 只允许三种形状：`tool_calls`（只能调已注册接口）、`hypotheses`（必须同时给出
`test` 和 `would_falsify`，否则 `validate_turn` 判为不可证伪）、`questions`（只有人能做的
决定）。控制器本身**一行科学计算都没有**——哪怕一句 `if plddt > 70` 都会在这里种下一个
没有任何模板授权、也没有任何评审者能找到的阈值。

### 控制流：带回边、分支和终止

十个阶段 + 五个终止态。**回退是声明过的转移，不是递归调用**——清单必须能显示"修复后回到了
`prepare_structures`"，而栈帧写不进 JSON 文件。五个终止态而不是一个："在等你拍板"和
"它失败了需要你"不是同一条消息。

失败先分类再决定能不能重试，七类里只有两类可重跑：输入错误（修一次）、证据不足（放宽一次）。
缺二进制**永不重试**——再跑一遍也不会把它装上，而换个模型顶替正是本系统要防的事故。
"放宽"只意味着更多种子、另一个数据库、再加一个家族，**永远不意味着降阈值**——那是同一步
把证据删掉。修复和放宽都是**钩子**，控制器知道该修，但不知道正确的结构索引或种子集是什么，
猜了它就成了本该属于操作者的科学判断的作者。钩子报告"什么都没改"时直接上报，而不是打转。

### 三个审批闸门

请求本身说明"在批什么"：`_request_id` 把闸门、载荷和请求时间一起哈希，所以换了批次重问会
生成**新请求**，而不是继承旧批次拿到的批准。匿名决策被拒；已定的决策不可覆盖；队列落盘
（周一提请、周四批准，中间运行停过两次）。`guard_batch_selection` 查的是**清单**而不是任务
文件里的标志位。判据闸门在**读板之前**检查，而不是之后——先看数据再定判据，判据就变成了对
数据的描述。

### 可复现性

输入哈希、序列哈希（SHA-256 规范化后的序列，而不是登录号——登录号会被重新注释，异构体还共用
名字）、数据库快照版本、软件与模型版本（含 `{"rdkit": "absent"}` 这种诚实记录）、全局种子与
每步派生种子（重跑一步不会移动其它步的随机性）、残基编号映射（逐残基写出，因为"量错了残基"
是下游唯一检测不到的错误）、失败日志（每次尝试在**做路由决策之前**就写入清单）、审批记录、
以及成本。成本上限下没有任何记录时打印"nothing recorded"，不是 `0.0`——这是两个不同的事实。

### 什么叫"接缝"

契约、校验、溯源、产物布局、拒绝路径都实现并测试了，外部程序不在。接缝**不会退化成启发式**：
它返回 `tool_unavailable`，控制器不重试、不换工具。搜索二进制、结构预测器、对接程序、联合
复合物预测器、LigandMPNN、以及所有公共数据库连接器，当前**全部是接缝**。
`eval/baselines` 里的家族功能预测和底物特异性模型也是接缝，并且会在对比表里按名字列出
"不可用"——用一个替身基线让 agent 赢，是最讨人喜欢也最没有意义的结果。
