# E-Agent

**[Open E-Agent chat](https://psknlr.github.io/E-Agent/)** ·
[Configure your model and backend](docs/WEB_CHAT.md)

An agent for **enzyme function mining** and **substrate-directed engineering**,
indexed on the reaction rather than on the protein family. The pilot task is
asymmetric ketone reduction to a chiral secondary alcohol by a ketoreductase
(`configs/tasks/KRED_PILOT_001.yaml`).

The system does one thing end to end: it turns a chemical brief into a batch of
constructs that a laboratory can actually order, keeps a traceable record of why
each one was chosen, reads the plate back into typed records, and proposes the
next round. It is a laboratory planning instrument, not an oracle. It is built on
the assumption that the expensive failure in this field is not a bad ranking — it
is a confident, well-formatted number that was never measured.

---

## The design rules

These four rules are enforced in code, not requested in a prompt. They are the
reason most of this repository exists.

**1. A null is never auto-filled.** A field whose value nobody has decided stays
`None`. `TaskSpec.resolve()` is the only way to fill one, and it demands an
authority — `operator:<name>`, `literature:<PMID/DOI>`, `database:<name>@<version>`,
`template:<id>` or `experiment:<id>`. A language model's own guess is not an
authority and `Assumption` refuses it. A gate whose required fields are still
null does not open; the steps behind it do not run.

**2. A negative is never confused with an untested result.**
`OutcomeClass` has seven members, and the four that a binary `hit` column
destroys are kept apart structurally: *no target product detected at this
detection limit*, *the construct never expressed*, *something was turned over but
not the target*, and *that well was never run*. A negative record without a
detection limit is refused by the schema. A positive record without a detection
method that **identifies the product** is refused too — a rising NADPH absorbance
at 340 nm says something consumed the cofactor, and lysate does that.

**3. A modelling failure is never recorded as an experimental negative.**
A crashed docking run, an unresolvable atom role, an absent cofactor in the
coordinate file: these are `COMPUTATIONAL_FAILURE`. A measured violation of a
*calibrated* mechanism window is `COMPUTATIONAL_NEGATIVE`. Neither is
experimental. `PoseOutcome.as_record_outcome()` cannot return an experimental
class and raises `FabricationGuardError` if a future edit makes it try.

**4. There is no unvalidated weighted total score — anywhere, including inside a
sort key.** The nine scorecard axes carry different units and different error
structures, and this repository contains no calibration set mapping any weighted
combination of them onto the probability that an enzyme turns over a substrate.
`eagent.science.scorecard.refuse_linear_blend` exists so that the next
contributor who writes `0.4*pLDDT + 0.3*dock + 0.3*distance` is stopped with an
explanation. Ranking happens through hard feasibility gates, ordinal evidence
levels, within-family comparison, Pareto non-domination and a **stated**
lexicographic priority order that a reviewer can argue with.

A fifth rule follows from the four: **when an external tool is absent, the step
fails loudly.** It never falls back to a heuristic that imitates the tool. See
[Adapter seams](#adapter-seams-what-is-not-installed).

---

## The three input modes

`TaskMode` in `src/eagent/schemas/reaction.py`. They are different questions and
the system answers them differently.

### A — `enzyme_mining`: target enzyme mining

The reaction *and* the substrate are fixed. The question is "which enzymes do
this, to this molecule, under these conditions". This is the pilot mode. The
substrate structure is the index of the entire search, so `normalize_reaction`
raises a **blocker** if it is given only a prose name: a name fixes neither
tautomer, salt form nor stereochemistry, and a silently resolved name redefines
the project.

### B — `reaction_space_exploration`: the reaction is known, the substrate is not

**This mode refuses to name a best enzyme.** Activity and enantioselectivity are
properties of an enzyme–substrate *pair*. With the substrate unspecified there is
nothing to rank against, so a ranking would be a ranking of nothing — and every
downstream number would be computed against a molecule nobody chose.

Instead the step emits `SUBSTRATE_CLASS_SCAFFOLD`: four chemical sub-spaces
(aromatic, aliphatic, cyclic, functionalised ketone), each with the decision the
operator has to take, why it matters, the evidence to gather and the
representative question to answer. The run stops with an explicit
`best_enzyme_claim_refused` flag on the record and a human task
`choose_substrate_sub_space`. See `examples/walkthrough.md` for the real output.

### C — `substrate_directed_engineering`: a known parent, a new substrate

The parent enzyme is known and confirmed; the question is which substitutions
move substrate fit without destroying catalysis or expression. `TaskSpec`
requires `parent_enzymes` to be non-empty in this mode.

`propose_mutations` runs **second, not first**, and starts from
`ExperimentRecord`s that *confirm* a parent. The reason is diagnostic: when a
round-1 mining batch comes back empty, "we never found the right family" and "we
found a workable scaffold with a poor pocket" are different worlds needing
opposite responses, and only a confirmed parent separates them. Running on an
unconfirmed candidate requires a recorded `ParentOverride`, which is attached to
every resulting proposal as `contradicting_evidence` so the weakness travels with
the variant into the plate map.

---

## Install

Python 3.11. Runtime dependencies are `pydantic>=2`, `PyYAML`, `jsonschema` and
`click` — all pure configuration and CLI machinery.

```bash
git clone <this repository> && cd E-Agent
python3 -m pip install -e .
```

Or run straight from the checkout without installing:

```bash
PYTHONPATH=src python3 -m eagent.cli --help
```

`numpy`, `scipy`, `rdkit`, `biopython` and `networkx` are **not** required and
are not installed here. Every numerical routine in `src/eagent/science/` is pure
Python: the mmCIF/PDB reader, the Gotoh affine-gap aligner, the distance, angle
and dihedral maths, the Wilson interval and the submodular batch selection. The
`chem`, `bio` and `numeric` extras in `pyproject.toml` are optional and change
behaviour in exactly one documented way — when `rdkit` is present,
`normalize_reaction` offers a stereocentre perception as a *proposal requiring
confirmation*, never as a resolved value.

Tests:

```bash
PYTHONPATH=src python3 -m pytest tests -q
```

---

## Quickstart

```bash
# 1. Scaffold a task. Every field is null, and every null is a decision.
PYTHONPATH=src python3 -m eagent.cli init MY_TASK_001 \
    --mode enzyme_mining --reaction-class ketone_to_secondary_alcohol

# 2. Ask what is still unresolved, gate by gate. Exit code 3 means "unresolved".
PYTHONPATH=src python3 -m eagent.cli validate MY_TASK_001.yaml --all-gates

# 3. See what would run, without running anything.
PYTHONPATH=src python3 -m eagent.cli run MY_TASK_001.yaml --rundir runs/001 --dry-run

# 4. Run. It stops at the first human decision point. Exit code 4 means "blocked".
PYTHONPATH=src python3 -m eagent.cli run MY_TASK_001.yaml --rundir runs/001

# 5. Record a decision, with a name attached.
PYTHONPATH=src python3 -m eagent.cli approve reaction_spec_confirmed \
    --rundir runs/001 --actor "J. Chemist" --note "structures confirmed against the order"

# 6. Resume. Steps whose input hash is unchanged are skipped, not re-run.
PYTHONPATH=src python3 -m eagent.cli run MY_TASK_001.yaml --rundir runs/001

# 7. Package it, with every missing item named.
PYTHONPATH=src python3 -m eagent.cli bundle runs/001 --out packages/001
PYTHONPATH=src python3 -m eagent.cli bundle-verify packages/001
```

The shipped pilot task stops immediately and on purpose: all four fields the
first gate needs are null. That is the correct behaviour, and
`examples/walkthrough.md` shows the real terminal output.

---

## The CLI

Plain ASCII, no colour, no spinners, no `rich`. The output goes into a lab
notebook, an email and a ticket, and it has to mean the same thing in all three.
A value that is not known prints as `unknown`, never as `0` or an empty cell.

| Command | What it does |
| --- | --- |
| `eagent init <task_id>` | Scaffold a task YAML with its nulls intact and a header explaining each one. |
| `eagent validate <task.yaml> [--gate G \| --all-gates]` | Report unresolved fields per gate, plus the assumptions on record and who authorised them. |
| `eagent run <task.yaml> --rundir D` | Drive the controller, streaming each step's status, QC flags, uncertainties and next actions. `--dry-run`, `--step`, `--from`, `--until`, `--offline`, `--allow-network`, `--arguments`, `--templates`, `--max-retries`, `--seed`, `--llm-model` (needs `--allow-network`; proposals only, audited under `<rundir>/llm/`). |
| `eagent approve <gate> --actor NAME --rundir D` | Record a human decision against a *specific queued request*. `--deny` records a refusal; `--note` records the reason. Prints what the request does **not** carry before you decide. |
| `eagent status <rundir>` | Where the state machine is, the path it took, cost recorded, approvals, pending decisions, open uncertainties, blocking QC flags. |
| `eagent verify <rundir>` | Run the independent verifier over what the run recorded. Refuses to report a pass when there is nothing to check. |
| `eagent bundle <rundir> --out D` | Assemble the deliverables package, recording every one of the 18 standard items as present, partial or missing with what a curator must supply. |
| `eagent bundle-verify <bundle>` | Re-check a package anywhere: every declared file present, every hash as recorded, every unrecorded file reported. |
| `eagent templates list \| show <id> \| lint` | Read the template library — every threshold a run uses comes from there. `lint` reports uncalibrated windows and library integrity gaps. |
| `eagent sources list \| show <id> \| independence \| coverage \| verify` | Read the data-source registry: what each resource may be used to claim, what it may not, and which sources re-integrate which others. `verify --allow-network --write` calls the shipped probes and records what answered. |
| `eagent benchmark sdr \| feedback --data D` | Retrospective benchmarks on the SDR deposit: label recovery with grouped folds and the leakage priced, and the feedback-loop simulation. Both open with the statement that the labels are annotation-derived. |
| `eagent reference verify \| manifest \| fetch-coordinates \| bindings \| audit \| verify-sources \| import-workbook` | The KRED calibration reference set (19 experimental entries, 28 kinetic records): verify its files, numbers and cross-references offline; fetch its coordinates through the probed route and pin them by hash; derive what is measured on each entry; audit which entries could be references for which template and run the project's calibration machinery on them *in memory*; compare its kinetic numbers with their sources. Nothing here edits a template or writes a calibration record. |

Exit codes, so a wrapper script can tell the outcomes apart without parsing
prose: `0` done, `2` bad invocation, `3` something is unresolved, `4` a human
decision point is blocking, `5` the work ran and failed.

---

## The three human decision points

Cleared only by `eagent approve`, which records a named actor, a timestamp and
the payload they were shown into the run manifest. A boolean in a YAML file is
**not** an approval: `guard_batch_selection` checks the manifest, not the task
flag, so there is no path from "somebody wrote `true` in a file while debugging"
to "ninety-six genes were ordered".

| Gate | The question | If it is wrong |
| --- | --- | --- |
| `reaction_spec_confirmed` | Is this the reaction the project is about: this substrate structure, this product structure, this configuration? | Every sequence mined, every complex modelled and every gene ordered afterwards is evidence about the wrong molecule. |
| `synthesis_authorized` | Authorise this batch: these constructs, these wells, this cost? | Money and lab time are spent on a round whose composition nobody reviewed. |
| `functional_criteria_confirmed` | Which experimental result counts as supporting a functional claim — decided **before** the data exist? | The criterion gets fitted to whatever the plate produced, and the round has no endpoint. |

---

## Adapter seams: what is not installed

Several steps are **adapter seams**. The contract, the validation, the
provenance, the artifact layout and the refusal path are all implemented and
tested; the external program is absent from this environment, so the step reports
`tool_unavailable` and produces nothing. The controller never retries it and
never substitutes another tool, because reaching for a different model is how a
missing measurement becomes a fabricated one.

| Seam | Interface | Absent here |
| --- | --- | --- |
| Sequence search | `mine_sequences` | `blastp`, `mmseqs2`, `hmmsearch` binaries |
| Structure prediction | `prepare_structures` | any `StructurePredictor` (`UnavailablePredictor` is the default) |
| Docking | `model_complexes` | any `DockingRunner` (`UnavailableDockingRunner` is the default) |
| Joint complex prediction | `model_complexes` | any `ComplexPredictor` (`UnavailableComplexPredictor` is the default) |
| Inverse folding | `propose_mutations` | LigandMPNN (`MissingLigandMPNN` is the default) |
| Every public database | `retrieve_evidence` and the data layer | cache-first; four sources (UniProtKB, RCSB PDB, Rhea, Zenodo) have been called live and have checked clients, **the other 45 have not been connectivity-tested** |

`configs/tool_registry.yaml` registers each external tool four times — code,
model weights, input databases, outputs — because those four carry different
licence terms, and several structure predictors ship permissive code with
non-commercial weights. Of the 56 facets, 4 were read from the release's own
files, 4 are restrictions or licences *reported* to the project (VenusMine
CC BY-NC-ND 4.0; AP Novo's non-commercial weights and outputs) and not read, and
48 are unknown. An unknown commercial permission **blocks** a commercial run
rather than permitting one; a no-derivatives position blocks modifying and
republishing the artefact (`check_modification`).

---

## Repository map

```
src/eagent/
  errors.py envelope.py provenance.py context.py   typed failures, the uniform
                                                   result envelope, the run
                                                   manifest, the run context
  schemas/        the typed data model: chemistry, reaction and task, the
                  experiment record, candidates and scorecards, variants,
                  batches, the five template types
  science/        measurement, never judgement: mmCIF/PDB io, geometry,
                  residue numbering, cross-subfamily position equivalence,
                  stereochemistry, robustness, scorecard, diversity
  tools/          the ten scientific interfaces behind one contract
  datalayer/      six layers, identity, intake tiers, lineage, snapshots,
                  preconditions, the house database, the staged rollout
  connectors/     cache-first access to ten public resources
  harness/        templates, interface registry, the LLM boundary, the approval
                  queue, the controller state machine, the independent verifier
  eval/           leakage-controlled splits, endpoints, baselines, ablations
  deliverables/   the research package and its verification
  cli.py          the terminal interface
configs/
  templates/      reaction, family, catalytic, engineering, assay templates
  datasources/    49 registered sources across the six layers
  references/     the KRED calibration reference set: workbook, tables, pins,
                  bindings, verification record, and its NOTICE
  tasks/          the pilot task
  tool_registry.yaml
docs/             the documents listed below
```

## Documentation

| Document | What it covers |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | One controller, structured tools, an independent verifier, the experiment feedback interface — and why it is not a committee of role-playing agents. |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | The research protocol as implemented: the three-layer framing, the staged funnel, batch composition, the three experiment tiers, what happens after a round with no hits. |
| [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) | The record as a seven-part tuple, the seven outcome classes, the evidence ladder, reaction direction, ligand provenance. |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | What each structural metric may and may not support, the primary endpoint, both hit-rate denominators, the three leakage-controlled regimes, baselines and ablations. |
| [`docs/TEMPLATES.md`](docs/TEMPLATES.md) | The five template types, their fields, how to add a family, and why an uncalibrated window may never reject a candidate. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | The three milestones, the de novo branch, and a component-by-component status table. |
| [`docs/DATA_LAYER.md`](docs/DATA_LAYER.md) | The six-layer data architecture, joins, identity ladder, lineage, snapshots. *(owned elsewhere)* |
| [`docs/DATASOURCES.md`](docs/DATASOURCES.md) | The 49 registered sources and the specific caution attached to each. *(owned elsewhere)* |
| [`docs/HOUSE_DATABASE.md`](docs/HOUSE_DATABASE.md) | The project's own per-substrate database and its four storage-layer refusals. *(owned elsewhere)* |
| [`examples/walkthrough.md`](examples/walkthrough.md) | The pilot task, end to end, with the real commands and their real output. |

## Status

The repository implements the scientific harness and a web chat interface.
GitHub Pages serves a Browser + API mode that calls your selected model directly
and runs evidence queries and arithmetic tools in the browser. No Python server
is required for this mode. Its data bundle is generated by the actual Python
reference loaders, preserving their withheld quantities and citations. An
optional Python backend runs the original `ToolLoop` and reference readers.
The chat settings include provider presets, API key, model name and editable
API URL for MiniMax, GPT, Claude and compatible custom services. See
[the setup instructions](docs/WEB_CHAT.md). The page's connection status comes
from the selected runtime, and a model completion is marked verified only after
a real request has succeeded. Direct browser access requires an API that permits
cross-origin requests. Repository text does not indicate
whether an agent service is currently online.

Run `PYTHONPATH=src python3 -m pytest tests -q` for the current test results.
Tests using fake provider responses verify integration contracts; they do not
establish a live model completion.

What has been run against the outside world: four public databases (UniProtKB,
RCSB PDB, Rhea, Zenodo) were called live and have clients checked against
recorded responses, and one real dataset (the SDR substrate-class deposit) was
downloaded through a checksum-verified route and benchmarked. What has **not**:
no structure predictor, docking program, search binary or inverse-folding model
has ever been run (the gnina, Boltz and MMseqs2 adapters were tested against
fakes that write output in the documented shape). The included experimental
reference data comes from published sources; this project has not performed a
new wet-lab experiment. Live model calls require a valid key and reachable API.

What the repository can claim is narrow. It has a sound *machinery of claiming*;
a simple sequence-to-annotated-class model that beats a nearest-neighbour
baseline on a few classes and not on most, on **annotation-derived** labels,
with test sequences held out by cluster; and a simulation showing that feeding
labels back into selection helps on that same annotation-derived data, much less
so once the pool's redundancy is removed. It does **not** support the statement
that the agent has learned from experiments or validated new-enzyme discovery:
no experiment has been run. `docs/EVALUATION.md` §9 and `docs/ROADMAP.md` §7 give
the numbers and what each does not establish.

A first real reference set is also in the repository: 19 experimental KRED/SDR
entries and 28 kinetic records, compiled by an AI assistant, stored with hashes,
recomputed, cross-checked and compared with its sources
(`configs/references/kred_calibration/v0.1`). It is smaller than it looks (six
lineages; three behind the "core" records), it qualifies **one** entry for the
shipped NADPH-SDR template with no known inactives, and it cannot calibrate any
window -- 14 independent actives are the least a modest claim needs. The measured
hydride-donor approach angles of the three complexes that could be measured lie
outside the shipped advisory band. `docs/results/kred_reference_audit.txt` has the
numbers.

Everything described as a refusal is a code path that raises or returns a typed
failure; where a refusal has been exercised against real data, the docs say so.
`docs/ROADMAP.md` has the component-by-component table.

---

## 中文摘要

### 这是什么

E-Agent 是一个**以反应为索引**的酶功能挖掘与底物导向改造系统。试点任务是酮不对称还原
成手性仲醇（酮还原酶）。它的产出是一个实验室真能下单的构建体批次、每一个入选理由的可
追溯记录、把回板数据读回成带类型的记录、以及下一轮的改造提案。

它不是一个"给答案"的模型。整个仓库的大部分代码存在的理由只有一条：这个领域里最贵的错
误不是排序排得不好，而是一个**格式漂亮、看起来像测量值、实际上从未被测量过的数字**。

### 四条设计铁律（写在代码里，不是写在提示词里）

1. **null 永不自动填充。** 没人决定过的字段就保持 `None`。只有 `TaskSpec.resolve()`
   能填，而且必须给出权威来源：`operator:` / `literature:` / `database:` /
   `template:` / `experiment:`。语言模型自己的猜测不是权威，`Assumption` 直接拒绝。
   必填字段还是 null 的闸门不会打开，后面的步骤不会跑。
2. **阴性结果与"没测过"永不混同。** `OutcomeClass` 有七个成员，其中四个正是被
   `hit` 这一个二值列毁掉的信息：*在该检测限下未检出产物*、*构建体根本没表达*、
   *有转化但不是目标产物*、*那个孔压根没做*。没有检测限的阴性记录会被 schema 拒绝；
   没有"能鉴定产物"的检测方法的阳性记录同样被拒绝——340 nm 处 NADPH 吸光度下降只能说明
   有东西消耗了辅因子，而裂解液本身就会。
3. **建模失败永不写成实验阴性。** 对接崩了、原子角色解析不出来、坐标文件里没有辅因子，
   这些是 `COMPUTATIONAL_FAILURE`；只有在**已标定**的机理窗口内被测到越界才是
   `COMPUTATIONAL_NEGATIVE`。两者都不是实验结果。
   `PoseOutcome.as_record_outcome()` 在结构上无法返回实验类别，将来谁改坏了它会直接抛
   `FabricationGuardError`。
4. **任何地方都没有未经验证的加权总分——包括藏在排序键里的。** 九个评分轴单位不同、
   误差结构不同，本仓库里也不存在任何把它们的加权组合映射到"该酶能转化该底物的概率"的
   标定集。`refuse_linear_blend` 专门拦下下一个想写
   `0.4*pLDDT + 0.3*dock + 0.3*distance` 的人。排序靠：硬可行性门、序数化证据等级、
   家族内比较、Pareto 非支配，以及一个**写明的、可被反驳的**字典序优先级。

还有第五条是前四条的推论：**外部工具缺失时，步骤要大声失败**，绝不退回到一个模仿该工具
的启发式。

### 三种输入模式

- **A 模式 `enzyme_mining`（目标酶挖掘）**：反应和底物都定了，问"哪些酶能在这些条件下
  把这个分子转化掉"。底物结构是整个检索的索引，所以只给中文名或英文俗名会被判为
  blocker——名字固定不了互变异构体、成盐形式和立体化学。
- **B 模式 `reaction_space_exploration`（反应已知、底物未定）**：**这个模式拒绝给出
  "最好的酶"**。活性与对映选择性是"酶—底物对"的性质，底物没定就没有可排序的对象，排出来
  的名次是"对空气排序"。系统改为给出四个化学子空间（芳香酮、脂肪酮、环酮、带官能团的酮）
  各自需要的决策、为什么重要、要去取哪一层的证据，并在记录里打上
  `best_enzyme_claim_refused`，然后停下来等人选。
- **C 模式 `substrate_directed_engineering`（已知亲本的底物导向改造）**：必须填
  `parent_enzymes`。`propose_mutations` 刻意排在第二步而不是第一步：第一轮挖掘空手而归
  时，"家族就找错了"和"骨架能用但口袋形状不对"是两个完全不同的世界，需要相反的下一步，
  而只有一个**实验确认过的亲本**能把两者分开。要在未确认的候选上跑，必须留下
  `ParentOverride` 记录，并且这条弱点会被挂进每一个产出提案的 `contradicting_evidence`，
  一路跟到板图里。

### 安装与最短路径

Python 3.11，只依赖 pydantic v2 / PyYAML / jsonschema / click。numpy、scipy、rdkit、
biopython、networkx **都不需要**，`src/eagent/science/` 里的数学全是纯 Python：mmCIF
读写、Gotoh 仿射空位比对、距离/角度/二面角、Wilson 区间、子模批次选择。

```bash
PYTHONPATH=src python3 -m eagent.cli init MY_TASK_001
PYTHONPATH=src python3 -m eagent.cli validate MY_TASK_001.yaml --all-gates   # 退出码 3
PYTHONPATH=src python3 -m eagent.cli run MY_TASK_001.yaml --rundir runs/001  # 退出码 4
PYTHONPATH=src python3 -m eagent.cli approve reaction_spec_confirmed \
       --rundir runs/001 --actor "张三"
PYTHONPATH=src python3 -m eagent.cli bundle runs/001 --out packages/001
```

出厂的试点任务一跑就停，这是**正确行为**：第一个闸门需要的四个字段全是 null。
`examples/walkthrough.md` 里是真实的终端输出。

### 三个人工决策点

只有 `eagent approve` 能打开，并会把**决策人姓名、时间戳、当时看到的内容**一起写进运行
清单。YAML 文件里的布尔值不算批准：`guard_batch_selection` 查的是清单而不是任务标志位，
所以不存在"调试时随手写了个 true"通向"九十六个基因已下单"的路径。

### 诚实说明：哪些是"接缝"

搜索二进制（blastp / mmseqs2 / hmmsearch）、结构预测器、对接程序、联合复合物预测器、
LigandMPNN，在本环境里**全部不存在**。它们的契约、校验、溯源、产物布局和拒绝路径都已实现
并有测试，但真程序缺席时步骤只会报 `tool_unavailable` 并且什么都不产出。控制器既不重试，
也不换一个工具顶上——换模型顶替正是"缺失的测量变成捏造的测量"的那条路。

同样地，`configs/datasources/` 里 49 个数据源只有 **4 个**做过连通性测试（其余 45 个没有），
`configs/tool_registry.yaml` 的 56 个许可面中 4 个读过原文、4 个是他人转述、其余 48 个记为未知，
未知的商用许可会**阻断**商业用途的运行，而不是放行。

### 现状

整个测试套件通过（写作时为 2919 个测试 + 696 个子测试；代码树仍在增长，当前数字请跑 `PYTHONPATH=src python3 -m pytest tests -q`）。
**对外部世界**：4 个公共数据库（UniProtKB、RCSB PDB、Rhea、Zenodo）做过真实调用并有按记录响应核对过的客户端，
一份真实数据集（SDR 底物类别）经校验和核对的路径下载并做过基准测试；另有一份 KRED 实验复合物参考集
（19 个条目、28 条动力学记录）被保存、哈希固定、重算并与原始表格比对——但它按现有模板只有 1 个条目合格，
**不能标定任何窗口**。**没有**跑过真实的结构预测器、对接程序、搜索二进制或反向折叠模型（适配器只用"按文档格式写
输出的假程序"测过），没有调用过语言模型，没有读过任何实验板。文档里写"系统会拒绝 X"，意思是**存在一条会抛错或返回带类型
失败的代码路径**；拒绝在真实数据上被触发过的地方，文档会明说。逐组件的状态表在
`docs/ROADMAP.md`。
