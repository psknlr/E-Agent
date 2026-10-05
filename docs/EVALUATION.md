# Evaluation

How this system is allowed to claim it worked, and — more of this document —
how it is not.

**Scope and status.** `src/eagent/eval/` is implemented and tested: splits,
endpoints, baselines, ablations. Nothing in it has been run on real experimental
data, because no round has been run. Two of the five baselines are seams and
report themselves unavailable rather than improvising. The headline caveat
travels with every ablation table the code renders:

> 96 experiments on a single substrate can establish a first application and
> cannot establish generality across reaction types.

---

## 1. The primary endpoint is experimental; structural metrics are supporting evidence

The objective in `TaskSpec` is `experimentally_confirmed_target_product`. Not a
score, not a ranking, not a pLDDT: **a construct shown, by a method that
identifies the product, to make the target product under the recorded
conditions.** Everything the computational pipeline produces is in service of
spending 96 wells well, and none of it is the result.

This is not modesty. It is the only arrangement under which the computational
part can be *wrong in a detectable way*. A pipeline whose endpoint is its own
ranking cannot fail; a pipeline whose endpoint is a plate can.

---

## 2. What each metric may be used for, and what it cannot support

Every row of this table is a claim somebody has made in a paper, and the right
column is why it does not follow. The left column is the real, legitimate use —
these metrics are not useless, they are *bounded*.

| Metric | Legitimate use | The claim it cannot support |
| --- | --- | --- |
| **Pocket-local pLDDT** (`StructureRecord.pocket_plddt`) | How much to trust coordinates *in the region the measurements are taken from*. Routes a structure to re-prediction or to an experimental alternative. | **Not activity, and not expression.** A confidently predicted active site is a confident picture of a geometry; it says nothing about turnover, and nothing about whether the protein folds and stays soluble in the chosen host. Confidence is about the *prediction*, not about the *protein*. |
| **Mean chain pLDDT** | A coarse screen for a model that failed outright. | **Not a substitute for the pocket figure.** The mean is dominated by the well-predicted core; a model at mean 92 can have a pocket at 55, and the substrate-binding loops — exactly the residues a selectivity argument depends on — are the ones that are disordered. `prepare_structures` records both and `pocket_confidence` reads the pocket. |
| **PAE / relative-position uncertainty** (`ComplexPose.model_confidence`) | Whether two parts of a model are *placed* confidently relative to each other — e.g. whether a domain carrying a catalytic residue is positioned relative to the cofactor site at all. | **Not a guarantee of turnover.** A confidently placed pocket can be confidently placed in the wrong conformational state, around the wrong ligand, or in a protein that never expresses. Low PAE between two regions means the predictor is sure where they are with respect to each other; it is silent on whether chemistry happens between them. |
| **Agreement between two predictions** (`cross_method_agreement`) | Evidence that a result is not an artefact of one method's particular failure mode — worth something precisely because template docking and joint co-folding fail *differently*. A `CONTRADICTORY` verdict is a strong signal that an experiment would be informative. | **Computational consistency, not accuracy.** Two methods trained on overlapping data and sharing the same structural priors can agree and both be wrong. Agreement raises an ordinal confidence level; it never becomes a probability of being right. |
| **A distance inside a template window** (`GeometryReport`) | The pose is *compatible* with the mechanism the template describes: the atoms the chemistry touches are arranged so the step is not excluded. | **A distance inside a window is not a reaction.** It is a static arrangement in one sampled conformer of a model. It carries no barrier, no dynamics, no protonation state and no solvent. And if the window was never calibrated (see §3) it does not even establish compatibility against anything measured. |
| **Pose robustness G** (`science/robustness`) | How *rare* the satisfying arrangement was across the sampled poses — the discriminating question, because enough sampling produces a satisfying pose for almost anything. Read through the Wilson **lower** bound, not the point estimate. | **Not a probability of catalysis.** It is a sampling statistic about a modelling protocol. 1/1 and 40/40 are the same fraction and very different evidence, which is why `classify_robustness` returns `INSUFFICIENT` below a minimum pose count rather than a high level. |
| **A docking score** (`ComplexPose.docking_score` + `docking_score_function`) | Ordering poses **within one scoring function, one receptor preparation and one box**. | **Not a turnover number and not an ee.** It is an unbounded pseudo-energy whose sign convention and dynamic range belong to one function; it ranks pocket volume and ligand heavy-atom count at least as strongly as it ranks catalysis, so comparing it across families ranks the families. `comparable()` in the scorecard refuses cross-function comparison rather than normalising it away. |
| **A model ranking score / ipTM** | Structural plausibility: how confident the model is that these coordinates are a real complex. | **Silent on turnover, selectivity and ee.** A pose that ranks first with wrong catalytic geometry is a confident model of an unproductive binding mode. These values are recorded verbatim, never rescaled or combined. |
| **A substrate-specificity model score** | Nothing yet — **it is a seam.** When one exists and has been calibrated on held-out data in the relevant regime, it becomes the right comparator for the agent. | **An uncalibrated specificity score is not a hit probability.** A number in [0, 1] from an unvalidated model is not a probability of anything; calling it one imports a guarantee the model never made. `StereoCall` enforces the same rule for the stereochemical case: a numeric `predicted_ee_pct` is **refused** without a named `calibration_source`. |
| **Sequence identity to a characterised enzyme** | A retrieval signal and a diversity axis. | **Not substrate specificity.** Two enzymes 85 % identical overall can differ at the three positions that set substrate size; global identity is dominated by the scaffold. This is why batch diversity is measured on the **pocket**, not on global identity. |
| **A family call** | Which mechanistic hypothesis to test, which catalytic template applies, which quota a candidate counts against. | **Not an activity prediction.** A family call cannot distinguish two members of one family with opposite substrate ranges — which is the entire problem a substrate-directed campaign has. The `family_function_prediction` baseline carries this sentence in its `question_answered` field precisely so its number is never read as answering the agent's question. |
| **An EC number** | A search hint that must widen a search. | **Not substrate-specificity evidence, and never a filter.** EC 1.1.1.– is written in the oxidation direction, and a sequence with no EC assignment is uncharacterised, not disqualified. |

Two meta-rules follow and are enforced in code:

**There is no total.** `science/scorecard.refuse_linear_blend` stops any attempt
to fuse these into one number. The axes are not nine measurements of one
quantity on nine scales, and this repository contains no calibration set mapping
a weighted combination onto the probability of turnover. A weighted sum would
manufacture a precise, reproducible, completely unjustified ordering — and
because it is precise and reproducible it would survive review.

**Weak evidence is not negative evidence.** `scorecard.evidence_weaknesses`
(low pocket pLDDT, few poses, routes disagreeing, an uncalibrated window, no
sequence-level literature) lowers the *strength of the evidence* and nothing
else. A candidate with weak evidence is one we have not looked at hard enough,
which is exactly the population an uncertainty-probe slot exists to sample.
Reading weakness as a negative is how a campaign ends up testing only the
proteins the models already understood. This is kept structurally apart from
`scorecard.input_defects` — a wrong sequence, an oxidised cofactor, an
unresolved stereochemical spec — which are **defects to repair**, disqualifying
until fixed and never traded off against a good docking score.

---

## 3. Gates are three-valued, and an uncalibrated window may not reject

`FeasibilityGate` outcomes are `PASS` / `FAIL` / `UNEVALUATED`, and
`UNEVALUATED` is never folded into `FAIL`. "We could not check" is routed to an
`Uncertainty` the controller can act on; folding it into a failure would let a
modelling budget read as chemistry.

The same three-valuedness runs through geometry. A failure against a window that
was never fitted to systems of known activity yields
`PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW`, which leaves `gating_passed` at
`None`, is excluded from the robustness denominator, and raises an explicit
uncertainty. All 17 shipped windows are uncalibrated; see
[`TEMPLATES.md`](TEMPLATES.md) §5 for the full argument, of which the short form
is: a false pass costs one well and the experiment finds it, a false rejection
removes a candidate before any experiment can see it, and the error is
correlated across a whole family.

---

## 4. The primary endpoint, computed against a criterion that cannot move

### The failure this is built around

A campaign spends 96 wells against a bar written down in advance — say 20 %
conversion with the target configuration dominant. The plate comes back with
nothing over 12 %. There is then an entirely reasonable-sounding conversation
about how 10 % is really the meaningful bar for a first round, the endpoint
becomes 10 %, and the paper reports four hits. **Nothing in the results table
records that this happened, and no reader can detect it.**

### How it is stopped

Every primary-endpoint function in `eval/metrics.py` takes a `PreRegistration`
and refuses to run against anything else:

- The criterion and its digest are fixed at registration, together with
  `registered_by` and `registered_at` — an unattributed bar is one nobody can be
  held to, and without a timestamp "before the data" cannot be checked.
- The criterion mapping is **copied** when read, so a caller still holding the
  original dict cannot change the registration by editing it.
- `assert_unchanged()` **re-derives** the digest from the criterion's own
  contents on every endpoint call, so an in-place edit of `criterion.raw` — the
  easy, accidental version of moving the goalposts — raises
  `CriterionChangedError`.
- There is **no parameter anywhere in the module that accepts pre-computed hit
  labels.** The hit definition is applied here, from the registered criterion,
  or the number is not produced. That closes the other bypass.

The criterion object is the same `PositiveCriterion` that `select_batch` wrote
into `experiment_plan.yaml` before the plate ran and that `ingest_results`
classified against, so the registration is checkable against the file on disk
rather than against a copy of it in memory.

### `precision_at_k` — registered hits among new candidates, over slots spent

Two deliberate asymmetries:

- **`known_candidate_ids` stay in the denominator and out of the numerator.**
  Their slots were spent, so they count against the budget; re-confirming a
  known enzyme is not a discovery, so it does not count as one. Erring this way
  means the endpoint can never be raised by stacking the batch with sure things.
- **Scoring more constructs than the registered budget raises.** That is not a
  generous round; it means these rows are not the round that was registered, and
  silently dividing by the larger number would answer a question nobody asked.

---

## 5. Both hit-rate denominators, always

Three hits out of eight expressed constructs is 37 %. The same three out of
twenty-four submitted is 12 %. Both are true, and only one gets quoted.

`HitRateReport` therefore has **no attribute called `hit_rate` or `rate`**. It
returns two:

| Rate | Denominator | Meaning |
| --- | --- | --- |
| `rate_all_submitted` | every construct put into the batch, including the ones that never expressed | what the round cost per hit |
| `rate_expressed_only` | only the constructs that yielded soluble protein; **undefined when none did** | how well the selection chose, given that expression worked |

Alongside them, the counts that make the two comparable:
`n_expression_failed`, `n_expression_unknown`, `n_not_tested`, `n_informative`,
`n_undecidable` (rows the criterion could not decide, e.g. because the
pre-registered bar is still `null`). Every rate carries a **Wilson interval**,
because 3/8 and 30/80 are the same fraction and very different evidence, and a
96-well round produces numbers much closer to the first. Wilson rather than the
textbook normal interval because the latter collapses to zero width at 0 and 1 —
"100 %, ± 0" from five samples is a precise claim nobody is entitled to.

Three further endpoint rules in the same module:

- **`signed_ee_aggregate` never means absolute ee.** It pools peak areas where
  it has them and otherwise reports a median with its range. Averaging absolute
  values turns a batch half of which made the wrong enantiomer into a success.
- **`variant_versus_parent` raises `ConditionMismatchError`** — the same error
  the house database raises — rather than returning a delta with a footnote. A
  variant assayed at a different pH, cofactor state, substrate loading or
  endpoint has not been shown to be better than its parent.
- **`EndpointMismatchError`** stops an initial rate being subtracted from a
  conversion. They are different quantities and the difference has no unit.

---

## 6. Leakage control: the three regimes and grouped splitting

A retrospective enzyme benchmark almost always reports a number that is too
good, by four mechanisms that leave no trace in the results table:

| Mechanism | Why the test rows are not new |
| --- | --- |
| **Re-curation** | One measurement, in one paper, curated into database A, re-integrated by B, re-integrated again by C. "Train on A, test on B" is not a split: the test rows are the training rows wearing a different accession. |
| **Variants of one parent** | A parent and its twelve single mutants differ at one position each. Parent in train, mutant in test, and the model is interpolating inside a sequence it has already seen. |
| **One publication, many rows** | Rows from one paper share a construct, an assay, a calibration and an analyst. Splitting them apart measures memorisation of that paper's idiosyncrasies. |
| **Sequence clusters** | Two sequences at 95 % identity are, for every purpose a substrate-specificity model cares about, the same protein. |

### Grouping comes before splitting, and there is exactly one grouping key

The unit assigned to a fold is **never a record**. It is the leakage-safe group
from `datalayer.lineage.leakage_safe_groups`: the transitive closure over
publication, experiment activity, parent-sequence lineage and sequence cluster.

**The source database is deliberately absent from the grouping key**, because
grouping by source is the re-curation trap written as a feature.

No second grouping key is defined in `eval/splits.py`. A second spelling of
"which rows are the same system" is precisely how an audit comes to disagree with
the split it is auditing.

### The three regimes

| Regime | Test units held out on | The question it answers | Shared on purpose |
| --- | --- | --- | --- |
| `novel_enzyme` | sequence clusters absent from training | does the method transfer to an unseen enzyme, on chemistry it has seen? | substrates — deliberately |
| `novel_substrate` | substrate scaffolds absent from training | does it transfer to an unseen substrate? | — |
| `dual_extrapolation` | both at once | **the regime a discovery campaign is actually in**, and the one on which published numbers are scarcest | — |

For the two substrate regimes the split unit is the closure over *both* the
lineage group and the substrate scaffold, because a scaffold straddling the
boundary is scaffold leakage and a lineage group straddling it is lineage
leakage, and both must hold at once. The price is stated rather than discovered
by surprise: an enzyme measured on a held-out scaffold is pulled into the test
fold along with its other measurements.

`novel_enzyme` requested with no sequence clustering raises
`SplitNotPossibleError` — it is a claim nobody could check. A dataset that merely
cannot be divided is **not** an error: `grouped_split` returns the split short
with the arithmetic in `shortfall_reason`, the way a short batch is reported
short.

Splits are deterministic: units are ordered by `sha256(seed:unit)`, so the same
inputs and seed give the same folds, a different seed gives a genuinely
different draw, and neither the first-listed nor the largest unit is
systematically the test fold.

### The audit runs after the split

`audit_leakage` reports six overlap categories, kept apart because the remedies
differ:

| Category | Remedy |
| --- | --- |
| `shared_sequence_cluster` | re-cluster at a tighter identity |
| `shared_scaffold` | regroup — **except** under `novel_enzyme`, where sharing substrates is the design |
| `shared_publication` | regroup |
| `shared_parent_lineage` | regroup |
| `shared_split_group` | a bug in the split itself |
| `recurated_source` | **cannot be fixed by splitting at all** — one of the two resources has to leave the evaluation |

`is_expected_under(regime)` returns true for exactly one case, so no regime can
be used to excuse an overlap it did not design for. The re-curation check reads
the `derived_from` edges of the source registry, which is what turns "these are
two independent resources" from an assumption into something checkable. Note the
registry's own honesty flag: 23 of the registered sources record their lineage
as **admittedly incomplete**, so a clean re-curation audit today is a statement
about what the registry knows.

Scaffold keys carry the basis they were computed from (`ScaffoldBasis`), because
an rdkit Murcko scaffold and this module's pure-Python ring-and-linker skeleton
agree about which molecules share a core far more often than they agree about
the *string* that names it — and comparing a key from one with a key from the
other reads as "different scaffold" for the same molecule, which is the
direction that leaks. rdkit is absent here, so keys fall back to the
ring-and-linker skeleton or to an InChIKey constitution block, and
`UNRESOLVED` is a real outcome rather than a silent default.

---

## 7. Fair baselines

"The agent found four hits; homology search would not have" is the claim the
project rests on, and the easiest claim in computational biology to make badly.
Three things must be identical across comparators or the comparison measures
those things instead of the method, and all three are enforced in code rather
than asserted in prose:

- **The candidate pool.** `CandidatePool` carries a digest over each member's id
  *and* its sequence hash, so neither swapping a candidate for another with the
  same id nor adding one extra can go unnoticed. `compare_baselines` raises
  `PoolMismatchError` on any selection naming a candidate the pool does not
  contain, or produced against a different pool digest.
- **The budget.** Picking 96 when the comparator picked 20 compares budgets.
- **The hit definition.** Every selection records the `PreRegistration` digest it
  was made under, and the comparison refuses a mixture.

Every comparator has the **same signature** —
`(pool, budget, registration) -> RankedSelection` — and anything a particular
comparator needs beyond that (seed accessions, a trained model, a docking sign
convention) is bound by a factory *before* the comparison starts, where it is
visible, rather than passed at call time where it could differ.

| Comparator | What it is | What it is entitled to answer |
| --- | --- | --- |
| `homology_multi_seed` | search from several seeds, spread the picks so the batch is not one clade | **the honest baseline to beat** — what a competent human does first |
| `family_function_prediction` | a family or EC-class predictor — **a seam** | "which family does this sequence belong to", which is a **different question** from "does this enzyme turn over this substrate under these conditions". A family call cannot distinguish two members of one family with opposite substrate ranges |
| `docking_score_ranking` | rank by the docking number | honest **only** within one scoring function and one pocket class, which the comparator enforces rather than assumes |
| `substrate_specificity_model` | a trained specificity predictor — **a seam** | the right comparator, when one exists |
| `full_agent` | the pipeline's own selection, through `compose_batch` — gates, lexicographic rank, family and clade quotas, diversity | the thing being evaluated is the code that would actually run |

**A seam returns "unavailable"; it does not improvise.** Where no model is
installed the comparator returns a `RankedSelection` with `unavailable_reason`
set and no picks, and `BaselineComparison.render` lists the unavailable
comparators **by name** so the gap is on the page. A stand-in baseline that the
agent then beats is the most flattering possible result and means nothing.

---

## 8. Module ablations

One module removed at a time, over a **fixed** pool, a **fixed** budget and the
**fixed** pre-registered criterion. The same selection code re-runs
(`compose_batch`) and the primary endpoint is recomputed over whichever picked
candidates have recorded outcomes. Nothing is simulated: a candidate a selection
picks that nobody ever tested is counted as **unscorable and named**, because
treating it as a failure would reward whichever variant happened to pick the
candidates someone had already run.

| `AblatedModule` | What is switched off |
| --- | --- |
| `catalytic_geometry` | the `catalytic_geometry` ranking axis **and** the `catalytic_machinery_mappable` gate — everything the template-driven geometry layer contributes to selection |
| `cofactor_constraints` | the `cofactor_compatible` gate; removing it admits candidates whose mechanism needs a cofactor the construct cannot use |
| `family_diversity_quota` | the family quota and the sequence-cluster cap, so an over-sequenced clade can take the whole plate |
| `active_learning` | the between-round update — **not evaluable from a single round**, and it says so rather than producing a number. With no prior round there is nothing for the module to have learned from. Supply a `PriorRound` and the ablation becomes meaningful |

**Removing a gate is not the same as failing it.** An ablation strips the gate
from a *copy* of each candidate's scorecard, so a candidate that would have been
excluded becomes eligible — the question is what the module kept out of the
plate. The originals are never mutated, so the full-system selection and the
ablated one are computed from the same inputs.

**`AblationResult.intervals_overlap`** is reported for every pair. A difference
of one or two hits out of 96 is inside the Wilson interval of almost any
comparison, and the point estimate must not be read as a finding. The scope
statement is repeated into **every rendered report**, not left in a docstring,
because a caveat that does not travel with the table someone copies into a slide
is not a caveat.

---

## 9. What this evaluation cannot establish

Stated here so it does not have to be inferred:

- **Generality across reaction types.** One ketone, one family set, one
  laboratory's conditions. Nothing here speaks to a transaminase or a
  halogenase.
- **That a module "contributes N percent".** See §8.
- **That the geometry layer is correct.** All 17 windows are uncalibrated; the
  layer can be shown to be *useful* (it changes what reaches the plate) long
  before it can be shown to be *right*.
- **Anything about enzymes the retrieval never reached.** The hit-rate
  denominators are over the batch, not over sequence space, and an empty
  evidence base is reported as an empty evidence base — never as evidence that
  no enzyme performs the reaction.
- **Any retrospective number at all, yet.** No data source has been
  connectivity-tested; there is no corpus in this environment to split.

---

## 中文摘要

### 主要终点是实验，结构学指标只是支持性证据

`TaskSpec` 的首要目标是 `experimentally_confirmed_target_product`：**一个构建体被一种
能鉴定产物的方法证明，在记录的条件下生成了目标产物。** 不是分数，不是排名，不是 pLDDT。
这不是谦虚，而是**唯一能让计算部分以可被发现的方式出错**的安排：以自己的排名为终点的流水线
不可能失败，以一块实验板为终点的流水线可以。

### 每个指标能支持什么、不能支持什么

- **口袋局部 pLDDT**：可用来判断"在取测量值的那个区域，坐标值不值得信"。
  **不是活性，也不是表达。** 置信度说的是**预测**，不是**蛋白**。
- **全链平均 pLDDT**：只能粗筛彻底失败的模型。**不能替代口袋数值**——平均值被折叠良好的
  核心主导，平均 92 的模型口袋可能只有 55，而底物结合环恰恰是选择性论证依赖的那些残基。
- **PAE / 相对位置不确定度**：说明模型对"两个部分彼此的相对摆放"有多确定。
  **不是周转的保证**——一个被自信摆放的口袋，可以自信地摆在错误的构象态里、错误的配体周围、
  或者一个永远不表达的蛋白里。
- **两种预测方法的一致**：有价值，正因为模板对接与联合折叠的**失效方式不同**；不一致
  （`CONTRADICTORY`）是"做个实验会很有信息量"的强信号。但它是**计算上的自洽，不是准确性**：
  训练数据重叠、先验相同的两个方法可以一致地错。
- **落在模板窗口内的距离**：说明这个位姿与模板描述的机理**相容**。
  **窗口内的距离不是一个反应**——它是模型某一个采样构象里的静态排布，不含能垒、动力学、
  质子化状态和溶剂；而如果这个窗口从未标定，它连"与被测量过的东西相容"都算不上。
- **位姿鲁棒性 G**：衡量"那个令人满意的排布有多**罕见**"——这才是有鉴别力的问题，因为采样够多
  几乎什么都能摆出一个满意位姿。按 Wilson **下界**读，不按点估计读。
  **不是催化的概率**，它是关于建模协议的采样统计量。
- **对接打分**：只能在**同一个打分函数、同一套受体准备、同一个盒子**内部排序。
  **不是 kcat，也不是 ee**——它对口袋体积和配体重原子数的排序至少和对催化的排序一样强，
  跨家族比较等于在给家族排序。
- **未标定的特异性打分**：**不是命中概率。** 一个未验证模型给出的 [0,1] 数字不是任何东西的
  概率。`StereoCall` 把同一条规则用在立体化学上：没有具名标定来源的数值 ee 被**拒绝**。
- **与已表征酶的序列同一性**：是检索信号和多样性轴，**不是底物特异性**——全局同一性被骨架
  主导，所以批次多样性按**口袋**衡量。
- **家族判定**：决定该测哪个机理假设，**不是活性预测**——它分不开同一家族中底物谱相反的两个
  成员，而这正是底物导向项目的全部难题。
- **EC 号**：只能**放宽**检索的提示。EC 1.1.1.- 按氧化方向书写，没有 EC 注释的序列是
  "未表征"而不是"被淘汰"。

两条元规则写在代码里：**没有总分**（`refuse_linear_blend`）；**证据弱不等于结果为负**
（`evidence_weaknesses` 只降低证据强度，而 `input_defects` 是**要修的缺陷**，在修好之前
取消资格，且永不与一个好看的对接分做权衡）。

### 门是三值的

`PASS` / `FAIL` / `UNEVALUATED`，而 `UNEVALUATED` **永不**并入 `FAIL`。"我们没法查"被路由成
一条控制器能处理的不确定性；把它并成失败，等于把建模预算读成化学。几何层同理：对未标定窗口的
越界让 `gating_passed` 停在 `None`，并被**排除出鲁棒性分母**。

### 判据不能移动

一轮实验按事先写下的 20% 转化率的门槛花掉 96 个孔，回板最高 12%。接下来会有一场听起来非常
合理的讨论：第一轮真正有意义的门槛其实是 10%。终点变成 10%，文章报告四个命中。
**结果表里没有任何东西记录这件事发生过，读者也检测不到。**

所以每个主要终点函数都只接受 `PreRegistration`：判据连同摘要在注册时固定，带注册人和注册
时间（没有署名的门槛没人需要对它负责；没有时间戳就无法核验"在数据之前"）；判据映射在读取时
被**复制**；每次调用都从判据自身内容**重新推导**摘要，所以就地修改 `criterion.raw`——那种
最容易、最不经意的移动球门方式——会抛 `CriterionChangedError`；并且**模块里没有任何参数接受
预先算好的命中标签**，这堵死了另一条绕行路径。

`precision_at_k` 有两条刻意的不对称：**已知能工作的候选留在分母里、不进分子**（槽位花掉了，
所以计入预算；重新确认一个已知的酶不是发现，所以不计入发现）——这个方向的偏差意味着终点
**永远不可能靠往批次里塞稳赢的候选来抬高**；打分的构建体多于注册预算会**抛错**，因为那说明
这些行不是被注册的那一轮。

### 两个命中率分母，总是两个

8 个表达成功里有 3 个命中是 37%；同样这 3 个在 24 个提交构建体里是 12%。两个都对，被引用的
只有一个。`HitRateReport` 因此**没有**叫 `hit_rate` 或 `rate` 的属性，它返回两个，外加
表达失败数、表达未评估数、未测试数、判据无法判定数，以及每个比率的 **Wilson 区间**——
3/8 和 30/80 是同一个分数但证据强度天差地别，而 96 孔的一轮产出的数字更接近前者。
用 Wilson 而不是教科书正态区间，是因为后者在 0 和 1 处宽度塌成零："五个里五个，100%，±0"
是一个谁也没资格做的精确主张。

### 泄漏控制：分组先于划分，三种外推机制

四种泄漏机制在结果表里都不留痕迹：**再策展**（一次测量被 A 收录、B 再整合、C 再整合，
"在 A 上训练、在 B 上测试"根本不是划分）、**同一亲本的变体**、**同一篇文章的多行**、
**序列簇**。

划分的**单位永远不是一行记录**，而是 `leakage_safe_groups` 的传递闭包（出版物 + 实验活动 +
亲本谱系 + 序列簇）。**数据源数据库被刻意排除在分组键之外**——按来源分组正是把再策展陷阱当成
特性来写。`eval/splits.py` 里**只有这一个分组键**：对"哪些行是同一个体系"的第二种写法，正是
审计与它所审计的划分产生分歧的来源。

三种外推：`novel_enzyme`（留出序列簇，**故意**共享底物）、`novel_substrate`（留出底物骨架）、
`dual_extrapolation`（两者同时——**这才是发现型项目真正所处的情形**，也是公开数字最稀缺的）。
后两种的划分单位是谱系组与骨架的**联合闭包**，代价被明说：一个在留出骨架上被测过的酶，会连同
它的其它测量一起被拉进测试折。

划分之后做审计。六类重叠分开报告，因为**补救方式不同**：共享序列簇→收紧同一性重聚类；
共享骨架→重新分组（**除了** `novel_enzyme`，共享底物是它的设计）；共享出版物/亲本谱系→重新
分组；共享划分组→划分本身有 bug；**再策展来源→根本无法通过划分修复**，两个资源必须走掉一个。
注册表自身的诚实标记提醒：49 个源里有 23 个的血缘记为"承认不完整"。

### 公平基线与模块消融

池子、预算、命中定义三者必须跨比较器完全一致，否则比较测的是这三者而不是方法——三者都在代码
里强制（池摘要覆盖成员 id **和**序列哈希；预算一致；每个选择都记录它所依据的预注册摘要）。
每个比较器签名**相同**，额外需要的东西由工厂在比较**开始前**绑定，放在看得见的地方。

**接缝返回"不可用"，不即兴发挥。** 没装模型时比较器返回空选择并设置 `unavailable_reason`，
渲染时**按名字列出**。用一个替身基线让 agent 赢，是最讨人喜欢也最没有意义的结果。

消融在**固定**池、**固定**预算、**固定**预注册判据下逐个移除模块。没有任何模拟：某个选择选中
但从没人测过的候选被计为**不可打分并点名**——把它算作失败，等于奖励"恰好选中了别人已经做过的
候选"的那个变体。移除一个门不等于没通过这个门：消融在评分卡的**副本**上剥掉门，原件不变。
每一对都报告 **Wilson 区间是否重叠**：96 个里差一两个命中，在几乎任何比较的区间内部。
范围声明被复制进**每一份渲染出来的报告**——一个不会跟着表格一起被粘进幻灯片的告诫，不算告诫。

### 这套评估**不能**建立什么

跨反应类型的普适性；某个模块"贡献了 N 个百分点"；几何层是**正确的**（17 条窗口全未标定——
能证明它**有用**远早于能证明它**对**）；关于检索从未触及的那些酶的任何结论；
以及——目前——任何回顾性数字，因为本环境里没有任何数据源做过连通性测试，没有语料可供划分。
