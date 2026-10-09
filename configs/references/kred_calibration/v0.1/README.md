# KRED calibration reference set v0.1

Nineteen PDB entries audited for use as experimental enzyme–substrate complexes,
28 kinetic records matched to them, 10 enantioselectivity (`er`) records, the
per-ligand density statistics behind the structure grades, and the source list
with each source's reuse terms. Compiled by an AI research assistant (audit date
2026-10-07), delivered as `source/KRED_Calibration_Reference_v0.1.xlsx`, and
added here with its provenance, its checks, and its limits written down.

**Read `NOTICE.md` first** for who compiled it, what was *not* received (the
~12 MB ZIP the delivery describes), and the per-source reuse terms — they differ,
and one (HBDH 2018 SI) is non-commercial.

## What is in this directory

| path | what | in `data_digest` |
|---|---|---|
| `source/*.xlsx` | the delivered workbook, byte for byte | yes |
| `tables/*.csv`, `formulas.json`, `sheet_notes.json` | a deterministic conversion of the workbook (no interpretation: numbers as stored, formula text kept apart) | yes |
| `coordinates/coordinates.manifest.json` | sha256, size and facts for each of the 19 mmCIF files, and the RCSB's family annotation of each scored entity. **The files themselves are not committed** (16 MB; CC0; fetched on demand) | yes |
| `bindings/bindings.json` | which residue is the substrate, which the cofactor, which atoms are measured, which conformer — derived by rule, **unreviewed** | yes |
| `verification/results.json` | the workbook's kinetic numbers compared with the tables they came from | no |
| `MANIFEST.json` | sha256 of every file here, and the two digests | — |
| `NOTICE.md`, `README.md`, `.gitattributes` | provenance and terms; line endings pinned so the hashes survive a checkout | no |

## Commands

```
eagent reference verify                    # files, numbers, cross-references; offline
eagent reference manifest [--write]        # check, or deliberately regenerate, MANIFEST.json
eagent reference fetch-coordinates --allow-network   # 19 files, each hashed against its pin
eagent reference bindings [--write]        # what is measured on each entry, and why not
eagent reference audit [--out docs/results]          # eligibility + calibration run, in memory
eagent reference verify-sources --docs DIR # kinetic numbers against their tables
eagent reference import-workbook X.xlsx --out DIR    # start a new version of the set
```

`verify` fails on a single edited byte, on a unit conversion the workbook caches
wrongly, on a cross-reference that is not followed both ways, and — once the
coordinates are fetched — on a validation row that names a residue the file does
not contain.

## What the checks established (2026-10-08)

* **The delivery is internally consistent.** Every count its cover note states
  (19, 28, 26, 2, 18, 8, 10, and the 22/3/1 tiers) is what the tables hold; all
  74 cached SI-unit formula cells equal an independent recomputation through the
  project's unit table; every structure↔kinetic link is followed both ways; all
  118 ligand-validation rows name a residue that is in the pinned file, with the
  right chain, number, component and alternate location.
* **The kinetic numbers match their sources.** 25 of the 28 records match the
  paper's own printed table for every quantity compared; 1 matches BRENDA's
  extraction only; 2 (LbADH) match BRENDA for `kcat`/`Km` and have 6 quantities
  that nobody read (a thesis reprint). All 10 `er` values match. Nothing
  mismatched. (`verification/README.md` says what that does not mean.)
* **Nineteen entries are six lineages.** *Lactobacillus brevis* ADH (5 entries)
  and *L. kefir* KRED (4) are 88 % identical over the aligned region (82 %
  coverage-adjusted); the workbook counts them as two enzymes, this project as
  one. The 22 "core" kinetic records are **three** lineages. Only **two**
  lineages have a substrate or product placed in the site (TR-II, HBDH).
* **Not every entry is the family it is used for.** By the RCSB's own annotation
  17 are classical SDR, 1Y1P (Ssal-KRED) is the NAD-dependent
  epimerase/dehydratase family, and the two TeSADH entries are zinc-type
  alcohol dehydrogenases.

## What the audit found, and what it did not

Run against the shipped `cat.sdr.nadph_carbonyl_reduction.v1` with the project's
own calibration machinery, in memory, under two declared policies:

* **One entry of 19 is eligible as shipped** — 1IPF (tropinone reductase II with
  NADPH and tropinone). The other 18 fail on named checks: oxidised or wrong
  cofactor (NAD+ ≠ NADPH), product state, low ligand density, scaffold only,
  racemic site, other family. **There are no known-inactive references**: the set's
  "activity-only" mutants have `kcat > 0` and no structure, and an `ND` is not a zero.
* **No scenario calibrates anything.** Relaxing the oxidation state adds nothing;
  dropping the cofactor requirement adds a second lineage (6ZZO, conformer A);
  accepting every graded pose adds no third. Two independent actives support a
  coverage of at most **0.106 at 80 % confidence**. A modest claim (80 %/80 %)
  needs **14** independent actives and a stricter one (90 %/90 %) **38**; the set
  supplies one or two.
* **What the real complexes measure.** Hydride-donor C4N to carbonyl carbon:
  4.06 Å (1IPF), 3.45 Å (6ZZO A), 3.67/3.25 Å (6ZZP A/B). Approach angle at the
  carbonyl carbon: 72.6°, 81.6°, 76.3°/77.7°. All four angles lie **outside the
  shipped advisory 90–130° band** (centred on a small-molecule trajectory), a
  further reason for it to stay advisory. These are three entries from two
  lineages, one of them at 2.5 Å resolution with ligand geometry outliers and one
  with an oxidised cofactor: a reason to doubt the band, not a window to replace it.
* **No template was edited, no calibration record was written, no citation can
  result.** Protein-residue constraints (catalytic Tyr/Ser/Lys) are unmeasured on
  every reference, because choosing those residues by proximity to the ligand
  would make the later measurement of that distance circular; a curator has to
  supply them from each enzyme's literature.

Full output: `docs/results/kred_reference_audit.txt` (readable) and `.json`.
A test fails if that file goes stale against the data.

## Do not use it for

* a benchmark of the pipeline or of any model — 19 entries, 6 lineages;
* a training set — 26 numeric records, three label types that must not be pooled;
* a reason to raise any window above `advisory`;
* converting kinetics into active/inactive labels. A340 NADH-depletion
  efficiencies do not show the product was formed, and an `ND` is not an inactive.

## 中文摘要

这是一份由 AI 助手整理、经用户上传的 KRED 实验复合物参考集（19 个 PDB 条目、28 条动力学记录、
10 条 er 记录）。本目录把它**原样保存、哈希固定、逐项复核**，但**没有**把它当成已验证的金标准：

- 工作簿原文件、转换后的 CSV、19 个坐标文件的 sha256、绑定记录都在 `MANIFEST.json` 里；改动任何一个
  字节，`eagent reference verify` 都会报错。
- 每个数值都用项目自己的单位表重算；每条交叉引用双向核对；118 条配体验证记录都在对应的坐标文件里找到
  了同链、同编号、同组分的残基。
- 动力学数值已与原始表格逐项比对：25/28 条与论文自己的表一致，1 条只与 BRENDA 一致，2 条（LbADH）
  的 kcat/Km 与 BRENDA 一致、其余 6 个量未读过原文；10 条 er 全部一致，**无不一致项**。
- **19 个条目只是 6 个谱系**（L. brevis ADH 与 L. kefir KRED 序列一致性约 88%，被合并为一个谱系）；
  22 条"核心"记录只对应 **3 个**谱系；真正在位点里放了底物/产物的谱系只有 **2 个**。
- 按现有 SDR 模板审计：19 个条目里只有 **1IPF** 合格（NADPH + 底物）；**没有已知无活性的反例**；
  所有情景（放宽氧化态、放宽辅因子、接受全部分级）都**无法标定任何窗口**——80%/80% 需要 14 个独立阳性，
  90%/90% 需要 38 个，这份数据只有 1–2 个。
- 实测几何：氢负离子供体 C4N 到羰基碳 3.25–4.06 Å，接近角 72.6°–81.6°，**全部落在模板的 90–130° 咨询窗口之外**。
  这只是 3 个条目、2 个谱系的观察，用来怀疑那个窗口，不能用来替换它。
- 没有修改任何模板，没有写入任何标定记录；催化残基（Tyr/Ser/Lys）未绑定，需要熟悉各酶文献的人来补。
