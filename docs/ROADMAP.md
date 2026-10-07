# Roadmap

Three milestones, then a branch. Each milestone has an acceptance criterion that
can fail, because a milestone nobody can fail is a description rather than a
plan.

**Where the project is today.** The harness is complete and tested: the whole suite
passes (1614 tests plus 415 subtests at the time of writing). **No step has ever run against a live public database, a
real structure predictor, a real docking program, a real inverse-folding model
or a real plate.** Milestone 1 is therefore *built but not demonstrated*, and
milestones 2 and 3 have not started because they require a laboratory, not more
code. §5 is the component-by-component table.

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

- **No data source has been connectivity-tested.** All 49 entries in
  `configs/datasources/` carry `connectivity_verified: false`, 49 carry
  `needs_curation: true`, and **0 record an endpoint**. Twenty-six are
  registered with a network access mode but no endpoint, which means the
  registry describes what each resource is *documented* to offer, not what has
  been proven to work from this machine.
- **No search binary, structure predictor, docking program or inverse-folding
  model is installed.** Every one of those steps is a seam that reports
  `tool_unavailable`.
- **No licence has been read.** All 44 facets in `configs/tool_registry.yaml`
  record `license: null` and `permits_commercial_use: null`, which **blocks** a
  commercial run.
- **No geometry window is calibrated.** 17 of 17.

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
| **Source connectivity** | **not started** | 0 of 49 sources connectivity-tested; 0 record an endpoint; 26 registered in a network mode with no endpoint; 6 are human-import only; 23 record their lineage as admittedly incomplete |
| `connectors/base.py` | implemented | cache-first contract, disclosure guard, evidence ceilings |
| Per-resource connectors (UniProt, PDB, …) | **not started** | all ten resources wired as `OfflineConnector` |

### Harness

| Component | State | Note |
| --- | --- | --- |
| `harness/templates.py` | implemented | strict loading, calibration report, window authority, integrity problems |
| `harness/registry.py` | implemented | strict ten-interface registry, dependency problems, manifest-ready report |
| `harness/llm.py` | implemented | numeric guard (value-to-cell binding, verified against the run's own artifacts), restricted turn shape, `EchoClient` on the tested path. **No production LLM client is wired**; `CallbackClient` is the embedding point |
| `harness/citation.py` | implemented | citation grammar, artifact index from the manifest, per-cell verification |
| `harness/approval.py` | implemented | three gates, named actors, persisted queue, hard batch block |
| `harness/controller.py` | implemented | declared state machine, seven failure kinds, resume by input digest |
| `harness/verifier.py` | implemented | nine checks, re-derived from primary material |

### Evaluation, deliverables, interface

| Component | State | Note |
| --- | --- | --- |
| `eval/splits.py` | implemented | three regimes, grouped splitting, six-category audit |
| `eval/metrics.py` | implemented | pre-registration guard, both denominators, signed ee, round-two report |
| `eval/baselines.py` | implemented, **2 seams** | `homology_multi_seed`, `docking_score_ranking`, `full_agent` run; `family_function_prediction` and `substrate_specificity_model` report unavailable by name |
| `eval/ablations.py` | implemented | four ablations; `active_learning` reports not-evaluable without a `PriorRound` |
| `deliverables/bundle.py` | implemented | 18 declared items, derived-file validation, hash verification |
| `cli.py` | implemented | 10 commands, 5 exit codes |

### Configuration and curation

| Artefact | State | Note |
| --- | --- | --- |
| `configs/tasks/KRED_PILOT_001.yaml` | implemented | ships with all four gate-1 fields null, deliberately |
| Reaction template (1) | implemented, **needs curation** | SMARTS never parsed in this environment; no RHEA id verified |
| Family templates (3: SDR, AKR, MDR/ADH) | implemented, **needs curation** | `seed_accessions` and `interpro_ids` empty; no accession verified |
| Catalytic templates (3) | implemented, **needs curation** | `reference_structures` empty in all three; **17 of 17 windows uncalibrated**, 0 gating |
| Engineering templates (3) | implemented, **needs curation** | SDR, AKR and MDR/ADH all present; windows uncalibrated as above |
| Family numbering schemes | **not started** | `science/family_numbering.py` and `science/pocket.py` ship the machinery; no sourced reference sequence is bundled, so pocket signatures fall back to composition and say so |
| Assay templates (3 tiers) | implemented, **needs curation** | every numeric bar `null`; limits of detection must be measured on site |
| `configs/tool_registry.yaml` | implemented, **needs legal review** | 11 tools × 4 facets = 44 entries; every licence `null`, every commercial permission `null`, which blocks a commercial run |
| `configs/datasources/*.yaml` | implemented, **not connectivity-tested** | see the data-layer table above |
| De novo design | **not started** | — |

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

## 中文摘要

### 现状一句话

框架已完成并通过测试（写作时 1614 个测试 + 415 个子测试），但**没有任何一步跑过真实的公共数据库、
真实的结构预测器、真实的对接程序、真实的反向折叠模型或真实的实验板**。里程碑 1 是
"已建成但未演示"；里程碑 2 和 3 尚未开始，因为它们需要的是实验室，不是更多代码。

### 里程碑 1：证据层与可复现筛选

**含义**：输入一个具名底物，输出一批构建体，背后有被记录的人工批准，每个候选都能追溯到支撑
它的证据，整次运行可以从清单重算。这个里程碑**不要求任何一个酶能工作**，它要求的是
"做出主张的机器"是可靠的。

**具体缺口**：49 个数据源**全部未做连通性测试**、全部 `needs_curation`、**0 个记录了端点**
（其中 26 个登记为网络访问模式却没有端点）；搜索二进制、结构预测器、对接程序、反向折叠模型
**一个都没装**；44 个许可面的 `license` 与 `permits_commercial_use` 全是 `null`，这会**阻断**
商业用途的运行；17 条几何窗口**全部未标定**。

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
- **未开始**：各资源的具体连接器（UniProt、PDB 等，当前全部按 `OfflineConnector` 接线）、
  数据源连通性验证（**0 / 49**）、AKR 与 MDR/ADH 的**改造模板**（因此这两个家族提不出变体）、
  从头设计。
- **已实现但需要策展**：全部 11 个模板（催化模板的 `reference_structures` 全空、
  **17 / 17 窗口未标定、0 条 gating**；检测模板的数值门槛全为 `null`）；工具注册表
  （11 个工具 × 4 个许可面 = 44 条，许可全 `null`）。
- **已知的瑕疵**：`eagent/tools/__init__.py` 里残留着规划期的 `INTERFACE_MODULES` 表，
  列了六个没有任何模块实现的名字。发现机制是权威的，控制器走的路径不报缺失；这是该删掉的
  死重，不是真正的缺口。

### 通往里程碑 1 的最短路径

按"解锁最多"排序：补齐试点任务的四个一号闸门字段 → 按 `evidence_gaps.tsv` 列出的十二条查询
填充连接器缓存（这是策展任务，不是写代码）→ 装一个搜索二进制并指向带快照日期的本地序列库 →
放入 mmCIF 结构或装一个预测器并记录版本与许可 → **标定一条窗口**（选 SDR 的氢负离子转移距离，
给出实验三元复合物，拟合上界，写进 `calibrated_on`，然后才考虑提升 severity）→ 读实际装上的
那些工具的许可并连同版本记录 → 写 AKR 与 MDR/ADH 的改造模板，否则只有 SDR 亲本能提变体。
