# House Database

The project's own per-substrate enzyme sequence, catalytic performance and
mutation lineage database: `src/eagent/datalayer/house_db.py`.

**Status.** The schema, the refusals, the SQL triggers and the reporting types
exist and run. The file is standard-library `sqlite3`, schema-versioned at
`SCHEMA_VERSION = 1` with a migration path. **It contains no data.** Every
number in this document that looks like a result is an illustration computed
from the types, not a measurement. Whether any registered public resource
covers a given substrate is a **per-task coverage audit**, not a claim this
document makes.

---

## 1. Why build this rather than another general sequence repository

Public resources already hold millions of sequences and a thin layer of
re-curated activity labels. Building a forty-ninth general repository would add
nothing. What no public resource holds, and what this project cannot buy, is the
material that actually makes substrate-directed engineering work:

* **Every construct that was submitted in a round**, including the ones that
  never expressed and the ones that were never measured. Without them a hit rate
  has no honest denominator.
* **The prediction that was made before the experiment ran**, frozen against the
  dataset snapshot it was computed from. Without it, "did the agent improve
  discovery efficiency" is a story told afterwards rather than a measurable
  question.
* **Mutation lineage as a first-class relation**, not a string in a notes
  column, so "every descendant of the parent that gained the W110A background"
  is a query.
* **Three performance axes recorded apart**, with no place anywhere in the
  schema to collapse them into one number.
* **The conditions each number was measured under**, as part of the record key,
  so a variant and its parent are only ever compared when they are comparable.

Those five things are exactly what public enzyme resources discard. A published
table reports the variants that worked; a database that keeps only what was
published cannot tell you what a round cost.

The database is **per substrate target**. The same enzyme against a second
ketone is a different project, not a second row in the same campaign, and
`substrate_target` is the unit everything else hangs from.

### What existing resources contribute, and what they do not

The design is informed by resources this project registers and respects:

| Resource | What it contributes to the design |
| --- | --- |
| `enzengdb` | Engineered parents with their variants and reported performance — the record *shape* this project's own batches produce |
| `strenda_db` | What a fully reported enzymology record looks like: protein, conditions and measured quantity present together |
| `skid`, `intenzydb` | The sequence–kinetics–structure join, and the reminder that such a link must be re-checked per record |
| `retrobiocat_db` | Substrate-scope organised by reaction type, which is the shape a substrate-directed search needs |
| `mavedb`, `fireprotdb` | Dense position-tolerance and stability priors — and the reason `measurement_type` must travel on every row |
| `enzymeml` | The reporting vocabulary the export is organised along |

**Whether any of them covers a specific substrate remains an open,
per-task question.** None of these resources is asserted here to contain data
for the pilot ketone, or for any other substrate. `coverage_audit()` is the
method that answers it for a given target, and it answers with an intersection
count, not a headline.

---

## 2. Schema

Twelve tables. Every `*_at` column is UTC to the second.

```
substrate_target ──< experiment_round ──< prediction ──┐
        │                    │                         │
        │                    └──< experiment_record ────┤──< evidence
        │                                   │           │
candidate ──< candidate_accession           └──< lineage_group
    │  │
    │  └──< performance_axes
    └──< lineage (parent_sha256 -> variant_sha256)

prediction_freeze_log      schema_version
```

| Table | Key columns | What it is for |
| --- | --- | --- |
| `substrate_target` | `substrate_target_id` PK; `substrate_ladder_json`, `product_ladder_json`, `target_stereochemistry`, `creates_new_stereocenter`, `reaction_id`, `reaction_class`, `required_cofactor_species`, `required_cofactor_state`, `needs_curation`, `curation_notes` | One precisely defined substrate and target product. The whole `ChemicalIdentityLadder` is stored, so a later reader can see whether the molecule was defined by a structure or only by a name. |
| `candidate` | `sequence_sha256` **PK**; `sequence`, `construct_sequence`, `construct_sha256`, `candidate_id`, `origin`, `family`, `family_basis`, `cluster_id`, `organism`, `discovery_method`, `provenance_json`, `deprecated` | One enzyme, keyed by the hash of its sequence. The **construct** sequence — what was actually expressed, tags and truncations included — is stored beside it. |
| `candidate_accession` | `(sequence_sha256, database, accession, version)` unique | Accessions as versioned **secondary attributes**, never as identity. An unversioned accession keeps `version = NULL` rather than being given a plausible one. |
| `lineage` | `parent_sha256`, `variant_sha256`, `mutations_json`, `numbering_reference`, `generator`, `round_id` | Ancestry as an edge. The numbering reference travels with the edge because `W110A` names two different residues in two numberings. |
| `experiment_round` | `round_id` PK; `substrate_target_id`, `round_number`, `plan_id`, `predictions_frozen_at`, `predictions_snapshot_id`, `results_ingested_at` | The unit a prediction is frozen against and a batch is ingested into. The two timestamps are what make "the prediction came first" auditable. |
| `prediction` | `(round_id, sequence_sha256)` and `(round_id, predicted_rank)` unique; `scorecard_json`, `stereo_call`, `stereo_basis`, `predicted_ee_pct`, `calibration_source`, `robustness_G`, `selection_role`, `selection_reason`, `uncertainty`, `confidence`, `snapshot_id`, `frozen_at`, `content_sha256` | Everything the agent claimed **before** seeing a result. |
| `prediction_freeze_log` | `round_id`, `snapshot_id`, `n_rows`, `n_replaced`, `reason`, `frozen_at` | Every freeze, superseded ones included. |
| `experiment_record` | `record_id` PK; `outcome`, `expression_status`, `soluble_expression`, detection block, product block, measurement block, `reaction_direction`, `reaction_id`, cofactor block, `conditions_json`, `condition_key`, `deprecated`, `deprecation_reason` | One construct's result in one round. Hits, negatives, expression failures and untested constructs all live here. |
| `performance_axes` | `(sequence_sha256, round_id, axis, kind)` unique; `axis` **CHECK**ed against the three names; `kind` CHECKed to `expected`/`observed`; `direction`, `value`, `unit`, `basis`, `evidence_json` | One axis, one row. |
| `evidence` | `record_id` / `sequence_sha256`, `source_type`, `identifier`, `strength`, `source_doi`, `source_record_id`, `license`, `database_version`, `experiment_activity_id`, `upstream_sources_json`, `extracted_by`, `verified_by`, `quote` | The provenance that makes the independent-evidence count possible. `upstream_sources` is kept verbatim. |
| `lineage_group` | `(group_id, record_id)` PK; `key_kind`, `key_values_json`, `representative_record_id`, `strongest_strength`, `n_rows` | Persisted output of `datalayer/lineage.py`, so the independent count is a query rather than a recomputation. |
| `schema_version` | `version` PK, `applied_at`, `description` | Opening a file written by a **newer** schema raises `SchemaVersionError` rather than writing through columns whose meaning has changed. |

Outcome classes, reaction directions and cofactor states are validated **in
Python** against the enums in `src/eagent/schemas/`, not frozen into SQL
`CHECK` constraints, because the enums are the single authority and a duplicated
list in DDL would silently diverge on an existing file. The three performance
axes are the one exception: they are a fixed architectural commitment, so the
`CHECK` is written into the DDL and a fourth "combined" axis cannot be inserted
even with raw SQL.

### Four refusals enforced at the storage layer

The Python API refuses these; the SQL triggers refuse them again, so a direct
`UPDATE` cannot slip past:

```sql
trg_prediction_frozen_insert  -- ABORT if the round's results_ingested_at is set
trg_prediction_frozen_update  -- ABORT, same condition
trg_prediction_frozen_delete  -- ABORT, same condition
trg_experiment_record_no_delete -- ABORT unconditionally
```

The abort messages say why, in the database, where someone reading raw SQL will
find them:

> `experiment records are never deleted: negatives and expression failures are
> the batch; use deprecate_record(record_id, reason)`

### Typed refusals

| Error | Raised when | What it prevents |
| --- | --- | --- |
| `FrozenPredictionError` | a prediction is written, edited or removed after that round's results arrived | a prediction tuned to the result, which makes top-k enrichment unfalsifiable |
| `DeletionRefusedError` | `delete_record()` is called (it always raises) | deleting negatives, which turns a 12 % hit rate into 100 % |
| `RecordOverwriteError` | an ingest changes a stored result on an existing `record_id` | delete-and-replace performed without a delete |
| `ConditionMismatchError` | a variant/parent comparison is demanded across differing conditions | a condition change reported as an engineering result |
| `CollapsedScoreError` | a payload carries a combined "mutation quality" score | the three axes silently becoming one |
| `SchemaVersionError` | the file was written by a different schema generation | writing through a column that means something else now |
| `UnknownRecordError` | a referenced round, candidate, target or record does not exist | a stub row created by a typo becoming an orphan candidate |

---

## 3. The pre-experiment prediction freeze

This is the only honest way to test whether the agent improves discovery
efficiency, and it is the reason the database exists.

`freeze_predictions(round_id, predictions, snapshot_id)` writes, per candidate:
the `predicted_rank`, the scorecard with its dimensions still separate, the
directional `stereo_call` and its `stereo_basis`, `predicted_ee_pct` (refused
without a named `calibration_source`), `robustness_G`, the `selection_role`, the
`selection_reason` — why that slot was spent — the stated `uncertainty`, and the
id of the **frozen dataset snapshot** it was all computed from.

The first `ingest_round()` for a round stamps `results_ingested_at`. From that
moment the round's predictions are permanently read-only:

> If a prediction can be edited afterwards, then top-k enrichment,
> rank-of-first-hit and stereochemical accuracy stop being measurements of the
> agent and become descriptions of the experiment — and there is no way for a
> reader to tell which one they are looking at.

Re-freezing **before** results arrive is legitimate; plans change. It requires a
`replace_reason` and is written to `prediction_freeze_log`, so a ranking
replaced before the results arrived is not invisible. The log is the evidence
that the final ranking was not the third attempt after a peek at the plate.

### What the comparison then reports

`prediction_vs_outcome(round_id)` returns a `PredictionOutcomeReport`:

| Measure | Definition |
| --- | --- |
| `baseline_rate()` | the round's own hit rate over informative rows — the chance level this round actually had |
| `top_k(k)` | hits among the `k` best-ranked informative, submitted constructs, with `n_considered` and `short_of_k` so a top-10 scored over 4 rows says so |
| `enrichment_vs_baseline(k)` | top-k rate ÷ the round's own hit rate; `None` whenever either side is undefined, notably when the round produced no hits at all |
| `rank_of_first_hit()` | predicted rank of the first construct that actually worked |
| `stereo_agreement_counts()` | `agree` / `disagree` / `undetermined` for the frozen stereochemical call against the measured ee |
| `n_submitted_without_prediction` | constructs that entered the batch after the freeze |
| `n_predicted_not_submitted` | ranked candidates that never made it into the batch |

Three guards keep the answer honest.

1. **If the predictions were not frozen before the results arrived**, the report
   comes back with `interpretable = False` and the reason, instead of a number
   that looks like evidence of a working agent.
2. **Rows that cannot inform catalysis are excluded** from the rates and counted
   separately. A top-ranked candidate that never expressed is a cloning result,
   not a wrong chemical prediction. `computational_failure` rows are excluded for
   the same reason: the modelling produced nothing, which says nothing about the
   enzyme.
3. **Constructs submitted without a prediction are counted**, because a batch
   that quietly grew after the ranking was frozen would otherwise flatter the
   top-k.

`stereo_agreement()` returns `undetermined` whenever the call was
non-directional or no ee was measured. Scoring those as agreement would make
stereochemical accuracy rise simply by predicting nothing.

---

## 4. Keeping whole batches, failures included

`ingest_round()` writes **every construct that was submitted**: the ones that
expressed and worked, the ones that expressed and did nothing, the ones that
never expressed, and the ones that were planned and never measured.

`batch_outcomes(round_id)` returns a `BatchOutcomes` whose point is that **there
is no filtered variant of it**. `by_outcome()` reports a count for every class
the enum defines, keeping the zeroes, so a report shows `expression failures: 0`
rather than omitting the line — an omitted line reads as "not measured".
`include_deprecated` defaults to `True`, because the set of constructs that were
submitted is a historical fact. A caller who wants only the hits must drop rows
themselves, in code a reviewer can see.

`deprecate_record(record_id, reason)` is the only way to retire a result. The row
stays in every batch query with `deprecated = True`, so a reader sees that a
measurement was made and later distrusted — which is a different fact from the
measurement never having existed.

### Refusals that stop an impossible row entering the table

`ingest_round()` refuses these combinations outright, because each one would put
a fabricated scientific value where a model will later read it:

* `confirmed_target_product` **without** a detection method that identifies the
  product. A cofactor absorbance change or a bare conversion number does not
  confirm which molecule was made.
* `no_target_product_detected` **without** a `limit_of_detection`. "No product"
  at an unstated limit bounds nothing.
* an ee value from a chiral method recorded as **not validated**. An unvalidated
  separation cannot assign a configuration.
* an ee outside `[-100, 100]`.
* `expression_or_solubility_failure` carrying a catalytic measurement. Protein
  that did not express cannot have been assayed; split the construct failure
  from the assay row.
* `not_tested` carrying any measurement.
* a `measurement_value` with no `measurement_type`.

Four further situations are not refused but are recorded as curation notes on
the row, and surfaced in the `IngestReport` rather than logged quietly: an ee
with `chiral_method_validated` unrecorded; a catalytic measurement with the
cofactor oxidation state unknown; a catalytic measurement with no cofactor
identity; and an experimental row with `reaction_direction = unspecified` —
"an oxidation measurement is not reduction evidence".

`IngestReport` returns `n_written`, `by_outcome`, `n_predicted_not_submitted`,
`n_submitted_without_prediction` and the `curation_notes`, so a harness step can
fail loudly instead of discovering six months later that half the batch has no
detection limit.

---

## 5. The three performance axes

| Axis | What it measures |
| --- | --- |
| `substrate_fit` | whether the substrate is accommodated and oriented |
| `catalytic_function` | whether turnover happens, and how fast |
| `stability_expression_risk` | what the change costs in stability, solubility and yield |

`record_performance_axis()` writes **one axis per row**, with its own
`direction`, `value`, `unit`, `basis` and `evidence`, and a `kind` of `expected`
or `observed` — stored on the same row shape so the two can be compared and
never merged, because an expectation written after the measurement is not an
expectation.

`performance_axes()` returns an `AxisReadout` with per-axis accessors and **no
total**. `THERE_IS_NO_COMBINED_MUTATION_SCORE` is a module constant that reports
quote verbatim:

> There is deliberately no column, view, property or helper in this schema that
> combines substrate fit, catalytic function and stability/expression risk into
> one number. The three are measured by different assays in different units and
> routinely move in opposite directions: a variant that binds better, turns over
> worse and expresses poorly is the normal result of round one. A weighted sum
> of the three erases exactly the information round two needs, and the weights
> would be invented rather than measured.

The commitment is enforced three ways: the SQL `CHECK` admits only the three
axis names; `_reject_combined_score()` runs on every scorecard and axis payload
written and raises `CollapsedScoreError` on any of a list of sixteen forbidden
keys (`overall`, `score`, `fitness`, `composite`, `mutation_quality`, `total`,
…); and `column_names()` exists so a test can assert that no table in the file
has acquired such a column.

### Comparing a variant with its parent

`variant_vs_parent(parent_hash)` compares each variant with its parent **only**
under identical conditions. Identical means every field of
`CONDITION_KEY_FIELDS` — `pH`, `temperature_C`, `buffer`, `solvent_system`,
`cosolvent_fraction`, `substrate_concentration_mM`, `enzyme_loading`,
`reaction_time_h`, `expression_host` — plus the cofactor identity and state,
plus the reaction direction, plus the measured endpoint and its unit. The
cofactor identity is part of the key, not metadata beside it: the same enzyme
with NADH and with NADPH is two experiments, and a comparison that pools them is
measuring cofactor preference while claiming to measure a mutation.

Anything else is **refused and reported as a refusal naming the differing
fields**. Refusals are first-class, because the common real outcome of a second
round is that the variants were run in a different plate format, and a report
showing only the comparisons it managed to make would present that as a clean
result set. A refusal naming `pH` and `cofactor state` tells a scientist exactly
which assay to repeat; a silently absent comparison looks like "no variants were
made".

Deltas are reported **per endpoint** — `delta_measurement`, `delta_ee_pct`,
`delta_conversion_pct`. There is no single "improvement score": a variant that
gains conversion and loses ee has not simply got better.

---

## 6. Both hit-rate denominators

`HitRate` deliberately has **no attribute called `hit_rate`**. Asking for "the"
hit rate of a screening round is the question that produces the number in the
abstract.

```
r1: 3 hits; 12.5% of 24 submitted, 37.5% of 8 expressed
    (11 expression failures, 5 expression unassessed, 0 never tested)
```

Both are true. Returning them in one object, with the expression failures and
the unassessed constructs counted beside them, makes the flattering one
impossible to quote alone. `as_dict()` carries every count needed to recompute
both rates by hand: `n_submitted`, `n_expressed`, `n_expression_failed`,
`n_expression_unknown`, `n_not_tested`, `n_hits`, `n_informative`, plus
`denominators_differ`.

`rate_expressed_only` is `None` when nothing expressed. A rate with a zero
denominator is not 0.0 and not 1.0 — it is undefined, and reporting it as a
number is how a failed round becomes a plotted point.

`ExpressionStatus.expressed_solubly` is tri-state. `None` is a real answer:
treating "not assessed" as expressed inflates the catalysis denominator;
treating it as failed inflates the hit rate. Unassessed constructs are counted
and reported on their own instead.

---

## 7. The coverage audit

The headline row count of an enzyme database answers a question nobody needs
answered. The question that decides whether a model can be trained, or a result
reproduced, is how many records **simultaneously** have:

1. a defined sequence,
2. a defined substrate structure,
3. a cofactor identity **and** oxidation state,
4. a defined product,
5. full reaction conditions (`pH`, `temperature_C`, `buffer`,
   `substrate_concentration_mM`, `reaction_time_h`), and
6. a quantitative result.

That is an **intersection**. Each facet alone is typically satisfied by most
rows; the intersection is typically satisfied by very few. Reporting facet
counts, or their union, is how a dataset of fifty thousand rows turns out to
contain two hundred usable ones.

`coverage_audit(substrate_target_id)` reports the intersection first and keeps
the union beside it only so the gap is visible:

```
coverage audit for t1: 6/50 records complete on all 6 facets (union touching any facet: 50)
  defined_sequence: present 50, missing 0
  defined_substrate_structure: present 41, missing 9
  cofactor_identity_and_state: present 22, missing 28
  defined_product: present 37, missing 13
  full_reaction_conditions: present 18, missing 32
  quantitative_result: present 31, missing 19
```

*(Illustrative numbers, computed from the type to show the shape of the report.
The database holds no data.)*

`worst_facet()` names the facet disqualifying the most records — where to spend
curation effort. A negative result **with a recorded detection limit counts as
quantitative**: it bounds the value, which is exactly what makes a negative
usable.

`_ladder_has_structure()` looks for an actual structural representation anywhere
in the serialised identity ladder rather than trusting a `resolved` flag, and
`register_substrate_target()` refuses a target with no structural rung on either
ladder unless `needs_curation=True` is set with a note saying what is missing.
Searching very efficiently for the wrong molecule, because
"4-chloroacetophenone" was never resolved to a structure, is the failure that
poisons a whole campaign.

`independent_evidence()` wraps `datalayer/lineage.py` so that four rows
re-curated from one paper count once, and persists the grouping to
`lineage_group`. If the lineage module is unavailable it returns
`available = False` with the reason — **never** a fallback to `len(records)`,
because a row count presented as an evidence count is the exact overstatement
the lineage module exists to prevent.

---

## 8. The EnzymeML-oriented export

`export_enzymeml_like(round_id)` arranges a round's content the way the EnzymeML
data model arranges it: proteins, small molecules, the reaction, and
measurements carrying their conditions.

**It is an interoperability export, not a certified EnzymeML document.** No
EnzymeML schema was available to validate against in the environment this module
was written in, so the document states so in its own body:

```json
{
  "format": "enzymeml_like_export",
  "format_version": "eagent.datalayer.house_db/1",
  "is_certified_enzymeml": false,
  "enzymeml_version": null,
  "needs_curation": true,
  "disclaimer": "Organised along the EnzymeML data model's concepts for
     interoperability. It has NOT been validated against any EnzymeML schema
     version and must not be presented as an EnzymeML document."
}
```

Claiming a version here would be a fabricated interoperability guarantee, and a
downstream tool would trust it. `vessels` is an empty list and the curation notes
say why: vessel type, volume and units are not recorded by this database and are
null rather than assumed. `reactions[].reversible` is `null`, with the note that
this database records the direction each assay ran, not a thermodynamic claim
about the reaction. Protein entries carry `ecnumber: null` because this database
does not assign EC numbers, and `sequence_is_expressed_construct` so a reader can
tell whether the sequence given is the construct or the catalytic domain.

Unlike most exports, **the failures travel with it**. Expression failures and
untested constructs appear as measurements with their `outcome_class`, their
`outcome_claim` and a flag `excluded_from_kinetics: true`, because an export that
silently drops them misrepresents the batch. Each measurement carries its
conditions, its cofactor block (name, redox state and **placement source**), its
detection block (method, limit of detection, whether the product identity was
confirmed, whether the chiral method was validated, whether an authentic standard
was used) and its own curation notes.

What a curator gets is a short, mechanical path to a real EnzymeML document
rather than a re-typing job. What they do **not** get is permission to call it
one.

---

## 9. Honest limits

* **No data.** Everything above is schema and policy. The first real test of
  this design is the first round that is frozen, run and ingested.
* **One migration.** `SCHEMA_VERSION = 1`. The migration machinery exists and is
  idempotent, each migration running in one transaction with its
  `schema_version` row, but no second generation has been exercised.
* **The EnzymeML mapping is unvalidated**, as stated above and in the export.
* **The novelty and exploration defaults in `plan.py` are policy, not measured
  quantities.** Nobody has determined an optimal exploration fraction for this
  chemistry.
* **Coverage of any specific substrate by any registered public resource is
  unknown** until `coverage_audit()` has been run on real imported records. The
  resources in §1 informed the design; none of them is claimed here to contain
  data for the pilot ketone.
* The hit-rate, top-k and coverage figures in this document are **illustrations
  computed from the types**, included to show the shape of each report.

---

## 中文摘要

### 为什么要自建，而不是再做一个通用序列库

公共资源已经有上百万条序列和一层很薄的、经过再次策展的活性标签。本项目真正需要、
又买不到的是另外五样东西，而它们恰恰是公共资源丢弃的部分：

1. **一轮里提交过的每一个构建体**，包括没表达出来的和根本没测的——否则命中率没有诚实
   的分母；
2. **实验之前就做出的预测**，并与它所依据的数据快照一起冻结——否则"智能体是否提高了
   发现效率"只是事后讲的故事；
3. **突变谱系作为一等关系**，而不是备注栏里的一句话；
4. **三条性能轴分开记录**，而且模式里**没有任何地方**可以把它们合成一个数；
5. **每个数字是在什么条件下测的**，条件是记录键的一部分。

数据库按**底物目标**组织：同一个酶面对第二个酮，是另一个项目，不是同一场campaign里的
第二行。

### 模式与四条存储层强制规则

共 12 张表。`candidate` 以**序列 SHA256 为主键**，登录号只是带版本的次级属性；
表达用的构建体序列与催化域序列分开存放。SQLite 触发器在数据库层面再次强制四条规则：
该轮结果一旦入库，`prediction` 表**禁止插入、更新和删除**；`experiment_record`
**永远禁止删除**。中止信息直接写在数据库里，读原始 SQL 的人也能看到原因。

三条性能轴写进了 DDL 的 `CHECK` 约束，所以即使用原始 SQL 也插不进第四条"综合"轴。

### 实验前预测冻结

`freeze_predictions()` 记录智能体在看到任何结果之前声称的一切：排名、维度仍然分开的
评分卡、带方向的立体化学判断及其依据、预测 ee（没有具名标定来源则拒绝写入）、稳健性值、
每个名额为什么花在这里、声明的不确定性，以及计算所依据的**冻结数据快照号**。

第一次 `ingest_round()` 会写入 `results_ingested_at`，从那一刻起该轮预测永久只读。
理由就是这个数据库存在的全部意义：**如果预测事后可以改，那么 top-k 富集、首个命中的排名
和立体化学准确率就不再是对智能体的测量，而变成了对实验的描述，而读者无从分辨自己看到的
是哪一种。** 结果到达之前重新冻结是允许的（计划会变），但必须给出 `replace_reason`
并写入 `prediction_freeze_log`。

若预测并非先于结果冻结，报告返回 `interpretable = False` 并附上原因，而不是给出一个
看起来像证据的数字。

### 保留整批，包括失败

命中、阴性、表达失败、从未测试——全部入库。`batch_outcomes()` **没有过滤版本**；
想只看命中的调用方必须自己在审阅者看得见的代码里丢行。退役记录用
`deprecate_record(record_id, reason)`：行保留、标记 `deprecated=True`，因为"测过后来
不再信任"和"从来没测过"是两个不同的事实。

入库时会直接拒绝几类不可能同时成立的组合，例如：没有能确认产物身份的检测方法却标
`confirmed_target_product`；标"未检出"却没有检出限；用未验证的手性方法报了 ee；
"表达失败"却带着催化测量值。

### 两个命中率分母

`HitRate` 类**故意没有名为 `hit_rate` 的属性**。3 个命中在 24 个提交构建体里是 12.5%，
在 8 个成功表达的构建体里是 37.5%——两个都对。二者必须同时返回，并把表达失败数和未评估
数并列在旁，这样讨好人的那个分母就无法单独被引用。没有任何构建体表达时，
`rate_expressed_only` 返回 `None`：零分母下的比率既不是 0.0 也不是 1.0，而是未定义。

### 覆盖度审计看交集，不看表头行数

决定"能不能训练模型、能不能复现结果"的，是有多少条记录**同时**具备：明确的序列、明确的
底物结构、辅因子物种**与**氧化态、明确的产物、完整反应条件、定量结果。这是一个**交集**，
永远远小于任何单项统计。`coverage_audit()` 先报交集，再把并集放在旁边，只为让差距可见；
`worst_facet()` 指出该把策展力气花在哪里。带检出限的阴性结果**计入**定量——它界定了取值
范围，这正是阴性可用的原因。

### EnzymeML 风格导出

`export_enzymeml_like()` 按 EnzymeML 数据模型的概念组织一轮的内容，但文档自身写明
`is_certified_enzymeml: false`、`enzymeml_version: null`，并附免责声明：**本环境没有
可用于校验的 EnzymeML 模式，因此它不是、也不得被当作 EnzymeML 文档。** 在这里编一个
版本号就是在伪造互操作性保证，而下游工具会信以为真。与多数导出不同的是，**失败记录会
一起导出**，标记 `excluded_from_kinetics: true`——悄悄丢掉它们就是在歪曲这一批的实际情况。

### 诚实说明

以上全部是**模式与规则，数据库里还没有数据**。现有的工程与报告类资源（EnzEngDB、
STRENDA DB、SKiD、IntEnzyDB、RetroBioCat、MaveDB、FireProtDB、EnzymeML）启发了这里的
设计，但**它们是否覆盖某个具体底物，仍然是每个任务各自要做的覆盖度审计**，本文档不作
任何此类断言。文中出现的命中率与覆盖度数字均为由类型计算出的示意值，用以说明报告形态。
