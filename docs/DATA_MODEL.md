# Data Model

The typed data model in `src/eagent/schemas/`: what a record is, what each
outcome entitles a claim to, and the distinctions the schema makes structurally
impossible to collapse.

This document covers the **record and claim** model. The six-layer storage
architecture, the joins between layers, the chemical identity ladder, lineage
counting and snapshots are in [`DATA_LAYER.md`](DATA_LAYER.md); the registered
public resources are in [`DATASOURCES.md`](DATASOURCES.md); the project's own
per-substrate database is in [`HOUSE_DATABASE.md`](HOUSE_DATABASE.md).

**Scope and status.** These are pydantic v2 models with validators, exercised by
the test suite. No record described here has been created from a real public
database or a real plate.

---

## 1. The record is a seven-part tuple

```python
Record = (Sequence, Substrate, Reaction, Cofactor, Conditions, Outcome, Evidence)
```

`ExperimentRecord` (`schemas/record.py`) is that tuple made explicit. Every part
is required to mean something before the record means anything.

| Part | Fields | Why it is part of the record and not metadata |
| --- | --- | --- |
| **Sequence** | `sequence`, `sequence_sha256`, `accession`, `database_version`, `is_variant`, `parent_sequence_sha256`, `mutations`, `construct_description`, `construct_sequence` | The hash is the identity. An accession is not: accessions get re-annotated, isoforms share names, and **an engineered enzyme usually has no accession at all**, so a database identifier cannot serve as its identity. `construct_sequence` is what was actually expressed — tags, truncations and fusions included. |
| **Substrate** | `SubstrateSpec`: isomeric SMILES or molfile, InChIKey, reactive atoms, prochirality, purity, origin | A prose name loses tautomer, salt form and stereochemistry. `_no_placeholder` refuses the strings `""`, `"null"`, `"None"` and `"TBD"` — use `None` for an unresolved SMILES, so that "unknown" cannot masquerade as a value. |
| **Reaction** | `reaction_class`, `reaction_direction`, `reaction_id` | See §4. |
| **Cofactor** | `CofactorSpec`: name, **oxidation state**, SMILES, PDB ligand code, transfer atom, stoichiometry, recycling system, `LigandSource` | See §5. NAD(P)⁺ and NAD(P)H are different molecules with different chemistry. |
| **Conditions** | pH, temperature, solvent system, cosolvent fraction, buffer, expression host, substrate concentration, enzyme loading, reaction time, cofactor options | See §2. |
| **Outcome** | `OutcomeClass` plus `Detection`, `conversion_pct`, signed `ee_target_pct`, `specific_activity`, `measurement_type` / `measurement_value` / `measurement_unit`, `kcat_s`, `km_mM`, `soluble_expression` | See §3. |
| **Evidence** | `list[EvidenceRef]` | See §6. |

Three validators fire on construction and are the model's teeth:

```python
if self.outcome.is_positive and not self.detection.confirms_product_identity:
    raise ValueError("confirmed_target_product requires a detection method that "
                     "identifies the product, not an indirect signal")

if self.outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED \
        and self.detection.limit_of_detection is None:
    raise ValueError("a negative record must carry the detection limit it is negative at")

if self.is_variant and not self.parent_sequence_sha256:
    raise ValueError("a variant record must name its parent sequence hash")
```

`measurement_type` exists so that a conversion, an initial rate, a specific
activity, a growth readout and a binding readout are never pooled onto one
numeric scale. They are different endpoints, and `eval/metrics.py` raises
`EndpointMismatchError` rather than subtracting one from another.

---

## 2. The same sequence under different conditions is a different record

`Conditions.key()` is part of the record's identity, not a decoration on it:

```python
(cofactors, pH, temperature_C, solvent_system, cosolvent_fraction, buffer,
 expression_host, substrate_concentration_mM, enzyme_loading, reaction_time_h)
```

and `ExperimentRecord.group_key()` is

```python
(parent_sequence_sha256 or sequence_sha256,
 substrate.inchikey or substrate.isomeric_smiles,
 cofactor.describe(),            # name + oxidation state
 conditions.key())
```

The reason is not pedantry. Collapsing conditions is what makes public enzyme
data unusable for substrate-level prediction, and it does so in four specific
ways:

- **Cofactor.** An NADPH-preferring enzyme tested only with NADH yields a true
  negative that reads as "this enzyme does not do the chemistry". Merge the two
  rows and you have manufactured a contradiction, or worse, averaged it away.
- **Cosolvent.** Most ketones of interest need a cosolvent, and the fraction
  changes both enzyme stability and the measured conversion. Two conversions at
  5 % and 20 % DMSO are two facts.
- **Host.** Soluble expression in *E. coli* and in a yeast host are different
  observations about the same gene, and only one of them may be in hand.
- **Temperature and pH.** Both move kinetics and both move the equilibrium
  position of a reversible carbonyl reduction.

So the model has no "activity" field on a sequence. It has records, and two
records differ when any element of the key differs. The same discipline runs
through the whole system: `eval/metrics.variant_versus_parent` raises
`ConditionMismatchError` — the same error the house database raises — rather
than returning a delta with a footnote, because a variant assayed at a different
pH has not been shown to be better than its parent.

---

## 3. The seven outcome classes

The distinction between "not tested", "tested and not detected" and "the protein
never expressed" carries most of the information in a screening campaign, and
all three collapse to `0` in a binary label. `OutcomeClass` keeps seven apart.
Three properties gate what may be done with each: `is_experimental`,
`is_positive`, and `informs_catalytic_ability`.

| Class | What it entitles you to claim | Experimental? | Informs catalysis? |
| --- | --- | --- | --- |
| `CONFIRMED_TARGET_PRODUCT` | "Target reaction activity exists under the recorded conditions." Requires a detection method that **identifies the product**. | yes | yes |
| `NO_TARGET_PRODUCT_DETECTED` | "No activity detected under these conditions **at this detection limit**." Not "inactive" — the limit is mandatory and is part of the claim. | yes | yes |
| `EXPRESSION_OR_SOLUBILITY_FAILURE` | "This construct did not pass expression; **catalytic ability undetermined**." Says nothing whatsoever about whether the enzyme can catalyse. | yes | **no** |
| `OTHER_PRODUCT_OR_WRONG_CONFIGURATION` | "Turnover occurred but does not meet the target requirement." Arguably the most valuable result a round can produce — a selective enzyme pointing the wrong way is an engineering target — and the first thing a binary label destroys. | yes | yes |
| `NOT_TESTED` | "Unknown." Nothing. | no | no |
| `COMPUTATIONAL_FAILURE` | "Modelling or tooling produced no usable result; **says nothing about the enzyme**." | no | no |
| `COMPUTATIONAL_NEGATIVE` | "Model training label only; **not an experimental negative**." | no | no |

### Separating a computational failure from an experimental negative

This is enforced at three points, not asserted once.

**At the producing step.** `PoseOutcome.as_record_outcome()` in
`evaluate_catalysis` maps every modelling outcome into the non-experimental half
of the enum, and raises `FabricationGuardError` if a mapping ever lands on an
experimental class:

| Pose outcome | Record outcome |
| --- | --- |
| `MECHANISM_SATISFIED` | `NOT_TESTED` — a satisfied pose is a hypothesis, not an observation; nothing was assayed |
| `MECHANISM_VIOLATED` | `COMPUTATIONAL_NEGATIVE` |
| `OUTSIDE_UNCALIBRATED_WINDOW` | `NOT_TESTED` |
| `NOT_MEASURABLE` | `COMPUTATIONAL_FAILURE` |
| `INPUT_ERROR` | `COMPUTATIONAL_FAILURE` |

There is **no code path from the evaluation module to
`NO_TARGET_PRODUCT_DETECTED`.** The counts are also kept in separate fields
(`n_violated` versus `n_not_measurable`) and the robustness denominator counts
only decided poses, so a failed modelling run cannot be averaged into a rate of
failure.

**At ingest.** `normalise_outcome()` refuses to guess. `n.d.` means "not
detected" in one paper and "not determined" in the next, so it resolves to
`NOT_TESTED` with a recorded uncertainty, never to a negative. Negation is read
*structurally*, from the position of a negator relative to the phrase it
governs, because a matcher that consumes phrases out of the text loses the
negator and reads "0 % conversion observed" as a confirmed product.

**At verification.** The verifier checks for an experimental-negative label on a
record whose evidence is a computational failure, and reports it as a finding.

### A positive needs the product, a negative needs a limit

A rising NADPH absorbance at 340 nm says something consumed the cofactor —
lysate does that, and so does slow autoxidation. `Detection` therefore carries
`confirms_product_identity` as an explicit boolean, and the schema ties
`CONFIRMED_TARGET_PRODUCT` to it. `AssayTemplate` enforces the same rule from the
other side: a tier-2 or tier-3 template whose `confirms_product_identity` is
false is rejected at load time.

Symmetrically, "no product" at an unstated sensitivity excludes nothing.
`ingest_results` holds such a row as *unresolved* rather than writing it down as
a negative — and never converts a questionable positive into a negative, which
would be the opposite error.

---

## 4. Reaction direction

`ReactionDirection` travels **with the record** rather than being inferred from a
label, because many curated resources store a reference direction that differs
from the direction actually assayed.

| Member | `supports_target_direction` |
| --- | --- |
| `FORWARD_AS_TARGET` | yes |
| `REVERSE_OF_TARGET` | no |
| `REVERSIBLE_BOTH_SHOWN` | yes |
| `UNSPECIFIED` | **no** — an unrecorded direction is not a forward one |

The concrete trap for this project: EC 1.1.1.– is written in the **oxidation**
direction. An alcohol dehydrogenase assayed on the alcohol, following NAD⁺
reduction, is a superb record of the oxidation and says nothing dependable about
ketone reduction at the target pH, with the target cofactor, at the target
substrate concentration — thermodynamics and kinetics both differ. Those records
arrive from curated resources with the same EC number and the same substrate
name, so without a check they enter a seed set as positives.

`datalayer/intake.direction_check()` uses two independent signals — the declared
`ReactionDirection` and whether the record's reaction class is the chemical
reverse of the target's — and treats a **disagreement between them as itself a
refusal**. The pilot task's `ec_hint: "1.1.1.-"` is documented in the task file
as a search hint that must widen a search and never narrow one: a candidate
without an EC assignment is uncharacterised, not disqualified.

---

## 5. Ligand source provenance

Where a ligand sits is one fact; how it came to sit there is another, and
collapsing them is how a computational placement becomes a reported observation.
`LigandSource` carries six origins, each with a `claim()` string that a report
can quote:

| Source | What it is entitled to say |
| --- | --- |
| `EXPERIMENTAL_OBSERVED` | observed in an experimental structure |
| `HOMOLOGY_TRANSPLANTED` | inferred by transfer from a homologous structure; **not measured here** |
| `DOCKING_PREDICTED` | a docking pose; a hypothesis about placement |
| `JOINT_STRUCTURE_PREDICTION` | a co-folded prediction; a hypothesis about placement |
| `MANUAL_PLACEMENT` | placed by hand |
| `UNKNOWN` | origin not recorded |

Only `EXPERIMENTAL_OBSERVED` returns true from `is_experimental`. A cofactor
transplanted from a homologue is a hypothesis worth checking, not a measurement
of this enzyme. `StructureRecord` and `ComplexPose` both carry the source
(`ComplexPose` separately for the substrate and the cofactor), and the verifier
checks the declared ligand identity and configuration against the spec.

### Cofactor state is part of cofactor identity

`CofactorState` is `REDUCED` / `OXIDIZED` / `NOT_APPLICABLE` / `UNKNOWN`, and
`UNKNOWN` is a real state that blocks rather than defaults.
`cofactor_state_from_ligand_code()` resolves only the four PDB chemical
components that are routinely confused —

| Code | Species | State |
| --- | --- | --- |
| `NAD` | NAD⁺ | oxidised |
| `NAP` | NADP⁺ | oxidised |
| `NAI` | NADH | reduced |
| `NDP` | NADPH | reduced |

— and returns `UNKNOWN` for anything else rather than guessing. `CatalyticTemplate`
refuses to load if it declares a required cofactor without an oxidation state:
*"NAD(P)+ and NAD(P)H are not interchangeable in a hydride-transfer model"*.
`CofactorSpec.is_hydride_donor` is true only when the state is `REDUCED` **and** a
transfer atom is bound.

### Reactive atoms, not centroids

`ReactiveAtoms` names the atoms the reaction touches — electrophile, nucleophile,
leaving group, stabilised atoms, prochiral centre — by **atom-map id**, not by
element name. Distance checks are defined against these. A distance from the
substrate centroid to the protein centroid is not a catalytic criterion, and
`AtomRef` exists so that nothing downstream can accidentally make it one.

---

## 6. The evidence strength ladder

`EvidenceStrength` answers one question: **how tightly does this claim bind to
this exact sequence?**

| Strength | Rank | Means |
| --- | --- | --- |
| `SEQUENCE_LEVEL_EXPERIMENTAL` | 4 | somebody measured this activity on this sequence |
| `HOMOLOG_EXPERIMENTAL` | 3 | measured on a homologue |
| `EC_SPECIES_MAPPED` | 2 | an EC number and an organism, not a sequence |
| `ANNOTATION_ONLY` | 1 | a database annotation, frequently propagated by similarity |
| `COMPUTATIONAL_CONSTRUCT` | 0 | produced by this pipeline |

`ExperimentRecord.max_strength` returns `COMPUTATIONAL_CONSTRUCT` when there is
no evidence at all — the honest floor, rather than an empty list silently reading
as "unqualified".

### An EC-plus-species mapping may not become sequence-level evidence without review

This is the single most common way an enzyme dataset corrupts itself: a BRENDA
row keyed on EC 1.1.1.1 and *Lactobacillus* passes through a pipeline, gets
joined to a sequence by organism, and emerges labelled as a measurement on that
sequence. It is enforced in three places.

**On the record.** `EvidenceRef` has a validator:

```python
if (self.extracted_by or "").startswith("model") and self.strength.is_sequence_level \
        and not self.verified_by:
    raise ValueError("a model-extracted record may not be promoted to sequence-level "
                     "experimental evidence without a human verifier")
```

**At the source.** Every entry in `configs/datasources/*.yaml` carries an
`evidence_strength_ceiling`, enforced on ingest *and* on promotion. BRENDA's is
`ec_species_mapped`; KEGG's, IntEnzyDB's, ESIBank's and CatPred-DB's are the same;
`enzengdb`'s is `homolog_experimental`; AlphaFold DB's and AlphaFill's are
`computational_construct`. A record cannot acquire a stronger label by passing
through a pipeline.

**On promotion.** `datalayer/intake.promote()` is the only upward path. It
demands a named **human** reviewer (a reviewer naming software is refused, which
stops a pipeline promoting its own output), a justification, and the registry.
To exceed a source's ceiling the reviewer must attach `supporting_evidence`: the
primary `EvidenceRef` they actually read. *BRENDA cannot be used to justify a
claim stronger than BRENDA supports; the paper behind it can.* There is
deliberately no `auto_promote`, no confidence threshold that promotes, and **no
promotion out of `MODEL_INFERRED` at all** — reviewing an inference does not turn
it into an observation.

### Tier and strength are different axes

`EvidenceTier` (`expert_verified_primary`, `curated_database`,
`machine_extracted_pending`, `model_inferred`) answers *who produced this row and
did a person check it*. `EvidenceStrength` answers *how tightly does the claim
bind to this sequence*. A machine extraction can concern a specific sequence, and
an expert can verify an EC-level mapping that is still only EC-level. Collapsing
them produces a dataset where "verified" means nothing in particular, so
`IntakeStore` partitions on tier and pooling requires naming the tiers you are
pooling.

### Four rows from four databases are not four pieces of evidence

`EvidenceRef` carries `upstream_sources` (the resources this record was
re-integrated from) and `experiment_activity_id` (the measurement campaign).
One measurement, curated into BRENDA, re-integrated by OED, by SKiD and by
CatPred-DB, retrieves as four agreeing rows. It is one experiment, and the
agreement is an artefact of copying. `datalayer/lineage.py` collapses them, and
`grouping_key` / `leakage_safe_groups` make the same collapse the basis of
evaluation splits — see [`DATA_LAYER.md`](DATA_LAYER.md) §6 and
[`EVALUATION.md`](EVALUATION.md) §4.

`EvidenceRef` also carries `license`, so redistribution terms travel with the
data rather than being looked up later, and `retrieved_at` /
`database_version` / `locator`, so a claim can be re-read at the page, table or
figure it came from.

---

## 7. The rest of the model, in one pass

| Model | Holds | The failure it prevents |
| --- | --- | --- |
| `TaskSpec` | the task, the three gates, the budget, the objectives, **and the `Assumption` ledger** | A filled field with no authority. `resolve()` is the only setter and refuses any source that is not `operator:` / `literature:` / `database:` / `template:` / `experiment:`. |
| `Budget` | the funnel targets | `detailed_complex_target < candidate_slots` is refused: selection would have nothing to select from. |
| `Objectives` | one primary (an **experimental confirmation**, not a score) and named secondary axes | They are never combined into one total. |
| `SequenceRecord` | a mined sequence plus its retrieval provenance | Derives the hash and the length; sets `has_nonstandard_residues` from the alphabet rather than trusting a flag. |
| `FamilyAnnotation` | signals supporting **and conflicting**, with an ordinal confidence | A family called from one motif. `recompute_confidence` returns `CONTRADICTORY` when conflicts outweigh support, which is a different state from "weak". |
| `CatalyticMapping` | role → residue in author numbering **and** role → 0-based index | Mixing the two axes is the off-by-N error that mutates the wrong residue. |
| `StructureRecord` | source, priority rank, missing regions, bound ligands, cofactor state, `mean_plddt` **and** `pocket_plddt`, numbering offset | A chain mean hides a disordered pocket; `pocket_confidence` reads the pocket. |
| `ComplexPose` | method, ligand sources, metals, docking score **with its scoring function**, model confidences, and `restrained_constraints` | A restrained distance re-used as independent evidence. |
| `GeometryReport` | every measurement, every satisfied/unsatisfied verdict, the circular constraints, and `independent_satisfied` / `independent_total` | `gating_passed` is three-valued: `None` means *undecided*, not failed. |
| `StereoCall` | a direction (`favors_target` / `favors_opposite` / `competing_poses` / `insufficient_evidence` / `not_applicable`) and pose counts | A numeric `predicted_ee_pct` is **refused without a named calibration source**. Pose counts are an artefact of sampling, not a Boltzmann population. |
| `ScoreDimension` | one axis: ordinal level, optional value with unit and direction, basis, and whether it is a gate | Nine axes, no total; gates are kept out of `SCORE_DIMENSIONS` so a gate can never be mistaken for a tradeable objective. |
| `Candidate` | everything known, plus `input_errors` and `disqualified` | An input defect (wrong sequence, wrong ligand, wrong chirality) is **repaired, never traded off** against a good docking score. `has_unresolved_gate` keeps "could not evaluate" apart from "failed". |
| `MutationProposal` | mutations in **both** numbering systems, per-class site evidence, `intended_improvement`, `possible_cost`, `axis_expectations`, `decomposition_controls` | Refuses a proposal with no stated cost; refuses a combination with no single-mutant controls; `apply_to()` verifies every wild-type letter before building the variant sequence and raises if the numbering is wrong. `axes_summary()` returns three entries and **never a total**. |
| `BatchPlan` | members by role, controls with their claims, `measurement_units` (wells, not genes) | Refuses a short batch with no `shortfall_reason`; refuses a control needing a new gene that does not occupy a slot. |

### Signed ee lives in the model, not in a report

```python
def ee_target(n_target: float, n_opposite: float) -> float:
    total = n_target + n_opposite
    if total <= 0:
        raise ValueError("no product quantified; ee is undefined")
    return (n_target - n_opposite) / total * 100.0
```

`ExperimentRecord.ee_target_pct` is bounded `-100 … +100` and documented as
*signed toward the target enantiomer; negative means the opposite configuration
dominated*. An undefined ee raises rather than returning zero.

---

## 中文摘要

### 记录是一个七元组

```
记录 = (序列, 底物, 反应, 辅因子, 条件, 结果, 证据)
```

序列的身份是**哈希**，不是登录号：登录号会被重新注释、异构体共用名字，而**改造过的酶往往
根本没有登录号**，所以数据库标识符当不了身份。`construct_sequence` 单独记录"实际表达的那
条序列"，包含标签、截短和融合。底物必须是结构（isomeric SMILES 或 molfile），
`"TBD"`、`"null"` 这类占位字符串会被直接拒绝——未解析就写 `None`，不能让"不知道"伪装成值。

三条构造期校验是这个模型的牙齿：阳性必须有**能鉴定产物**的检测方法；阴性必须带**检测限**；
变体必须写明亲本序列哈希。

### 为什么同一条序列在不同条件下是不同的记录

`Conditions.key()` 进入记录身份：辅因子、pH、温度、溶剂体系、助溶剂比例、缓冲液、表达宿主、
底物浓度、酶量、反应时间。理由很具体：偏好 NADPH 的酶只用 NADH 测，会得到一个**真阴性**，
读起来却像"这个酶不做这个化学"；助溶剂比例同时改变酶稳定性和实测转化率；宿主不同是关于同一
个基因的两个不同观察；温度和 pH 既动动力学也动可逆羰基还原的平衡位置。

所以模型里**没有**"某条序列的活性"这种字段，只有记录；键里任一项不同就是两条记录。
`variant_versus_parent` 在条件不一致时直接抛 `ConditionMismatchError`，而不是给一个带脚注的
差值——换了 pH 测出来的变体，没有被证明比亲本好。

### 七个结果类别

`confirmed_target_product`（在记录条件下存在目标活性）、
`no_target_product_detected`（**在该检测限下**未检出，不等于"无活性"）、
`expression_or_solubility_failure`（没表达，**催化能力未定**，对催化力一个字都没说）、
`other_product_or_wrong_configuration`（有转化但不达标——这常常是一轮里**最值钱**的结果，
方向错了的高选择性酶正是改造靶点，也是二值标签第一个毁掉的东西）、`not_tested`、
`computational_failure`、`computational_negative`。

**计算失败与实验阴性的分离**在三处强制：产生端
（`PoseOutcome.as_record_outcome()` 结构上只能落到非实验类别，否则抛
`FabricationGuardError`；该模块**不存在**通向 `NO_TARGET_PRODUCT_DETECTED` 的代码路径）；
录入端（`normalise_outcome()` 把 `n.d.` 解析成 `NOT_TESTED` 并记录不确定性——它在一篇文章里
是"未检出"，在下一篇里是"未测定"；否定词按**结构**识别，否则"观察到 0% 转化"会被读成确认的
产物）；校验端（校验器会把"给计算失败贴实验阴性标签"报成 finding）。

### 反应方向

EC 1.1.1.- 是按**氧化方向**书写的。一个醇脱氢酶在醇上测、跟踪 NAD⁺ 还原，是一份极好的氧化
记录，对"在目标 pH、目标辅因子、目标底物浓度下的酮还原"**说不出任何可靠的话**——热力学和动
力学都不同。可这些记录会带着同样的 EC 号和同样的底物名从策展库里过来。`direction_check()`
用两个独立信号判断，并且**两者不一致本身就是拒绝**；`UNSPECIFIED` 算不支持——没记录的方向
不是正向。

### 配体来源溯源

配体**在哪儿**和它**怎么到那儿的**是两件事，混为一谈就是"计算摆位变成观测事实"。
六种来源里只有 `experimental_observed` 算实验：从同源物移植过来的辅因子是一个值得检验的假设，
不是对这个酶的测量。

辅因子的**氧化态属于身份的一部分**：`NAD`/`NAP` 是氧化态，`NAI`/`NDP` 是还原态，表外的代码
一律返回 `UNKNOWN` 而不猜。`CatalyticTemplate` 在声明了需要某辅因子却没写氧化态时拒绝加载。
反应性原子按**原子映射号**指定——从底物质心到蛋白质心的距离不是催化判据。

### 证据强度阶梯，以及 EC+物种不得升格

五级：序列级实验 (4) > 同源物实验 (3) > EC+物种映射 (2) > 仅注释 (1) > 计算构造 (0)。

**一条 EC+物种的映射，未经复核不得变成序列级实验证据。** 这是酶数据集自我腐蚀最常见的方式。
三处强制：记录上（模型抽取的记录没有人工复核者时，不允许标为序列级）；数据源上
（每个源有 `evidence_strength_ceiling`，录入和升格时都查——BRENDA 的上限是 `ec_species_mapped`）；
升格时（`promote()` 要求**具名的人类**复核者——复核者写成软件会被拒，这挡住了流水线给自己的
输出升格——要求理由，要超过数据源上限还必须附上他真正读过的那条原始 `EvidenceRef`。
**BRENDA 不能用来支撑比 BRENDA 本身更强的主张，它背后的那篇论文可以。** 没有 `auto_promote`，
没有"置信度到了就升"，并且**完全不允许**从 `model_inferred` 升格——复核一个推断不会把它变成
一次观测）。

层级（tier）与强度（strength）是两个轴：前者问"谁产生的、有没有人看过"，后者问"这个主张跟
这条序列绑得多紧"。混成一个，"已核验"就什么都不意味着了。

四个数据库里的四行不是四份证据：`upstream_sources` 和 `experiment_activity_id` 让
`lineage.py` 把它们折叠回一次实验。

### 有符号 ee 写在模型里

`ee_target()` 朝目标对映体取符号，产物量为零时**抛错**而不是返回 0；
`ExperimentRecord.ee_target_pct` 的取值范围是 −100 到 +100。`StereoCall` 的数值 ee
在没有具名标定来源时被**拒绝**——位姿计数是采样协议的产物，不是 Boltzmann 分布。
