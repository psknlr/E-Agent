# The second delivery: `KRED_Calibration_Reference_v0.1.zip`

The reference set arrived twice. First a spreadsheet, which `../tables/` is the
conversion of. Then this archive of 194 files, sha256
`0b32db1402150ac9bcde2a8c4aac8c4da5b8eae13428f35b24a25f25c9266025`, audit date
2026-10-07. `archive.manifest.json` lists every member with its hash, the kind
the archive declared, the URL it came from where it has one, and what this
repository did with it.

## What the two deliveries agree on

Checked by `eagent reference verify-bundle`, not taken on trust:

| check | result |
|---|---|
| the archive's own `file_manifest.json` against the 194 members | 193 listed, every hash matches, nothing unlisted |
| the archive's spreadsheet against the one under `../source/` | byte-identical |
| the 19 coordinate files against this project's own RCSB pins | **19 of 19 byte-identical** |
| the archive's JSON tables against the spreadsheet's CSV conversion | **1632 shared fields, 0 disagreements** |
| `data/structure_audit_raw.json` against the 19 per-structure audits | identical entry by entry, so the aggregate was dropped |

The coordinate result is the strongest provenance statement available here. The
compiler fetched 19 entries from the RCSB; this project fetched the same 19
independently a day later through its own probed route; the bytes are the same.
Neither retrieval can now be the one that quietly caught a different revision.

The 1632-field agreement is weaker than it sounds and worth saying plainly: both
exports come from one compilation, so agreeing shows the exports are faithful to
each other, not that the values are right. What makes the values independently
checked is `../verification/results.json`, which compares them with the printed
tables of the papers.

Three representation conventions differ between the two exports and are
normalised before comparison, because they are spellings of one value: a JSON
list against a semicolon-joined string, `=` against `eq`, and a JSON boolean
against `1`/`0`.

## What is committed here and what is not

| disposition | files | size | what |
|---|---|---|---|
| `committed` | 54 | 1.63 MB | the curated compilation: the JSON/CSV tables, the full assay conditions, the 19 per-structure audits, three evidence files, the notices |
| `committed_upstream` | 8 | 0.12 MB | the 2026 activity data (MIT and CC BY 4.0, read by `eagent.eval.kred_activity`) |
| `pinned` | 129 | 46.05 MB | coordinates, biological assemblies, validation XML, RCSB entry/entity JSON, wwPDB chemical components — hash and URL kept, files not committed |
| `duplicate_of_source` | 1 | 0.05 MB | the spreadsheet, already under `../source/` |
| `dropped_duplicate` | 1 | 0.79 MB | `data/structure_audit_raw.json`, shown identical to the 19 per-structure files |

The rule is the archive's own `kind` label, with one stated exception: eight
`upstream_original` members are committed anyway because they are small, openly
licensed, and a loader in this package reads them. Pinning those alone would
make the activity loader need the network to do anything, which is the opposite
of the point. The exception is a list in `kred_bundle.COMMITTED_UPSTREAM`, one
path at a time, so it stays visible.

Every pinned file is re-fetchable from the `source_url` in
`archive.manifest.json`, and all four hosts are routes this project has probed.

## What the archive added that the spreadsheet did not have

* **The full assay conditions.** `data/assay_conditions.json` carries the
  concentration ranges, enzyme loadings, pre-incubation, cuvette and replicate
  counts the spreadsheet only summarised — including that the HBDH rows are
  printed as 283 K and called 10 °C by the later paper, with the instruction to
  preserve the original and treat 10 °C as nominal. That is this project's own
  discipline arriving from the other direction.
* **The 2026 ortholog activity data.** 109 constructs, 69 solubly expressed,
  four endpoint substrates. These are the first **measured** labels in this
  repository — see `eagent.eval.kred_activity` and `docs/EVALUATION.md`.
* **The chemical component definitions** for all 15 ligand codes (pinned), and
  the biological assemblies (pinned).
* **Three evidence files** recording the per-claim provenance behind the HBDH,
  ternary-complex and industrial-activity statements.

## Reuse terms

`NOTICE.txt` is the archive's own statement and is committed verbatim beside
this file. In short: PDB files CC0; BRENDA factual labels CC BY 4.0; the 2026
GitHub files MIT; the 2026 Zenodo deposit CC BY 4.0; the 2018 HBDH supporting
information is CC BY-NC 4.0 and **its PDF and extracted text are not bundled**,
only factual extractions and citations. No publication PDF is in the archive.
`../NOTICE.md` is this repository's per-source table and is the one to read
before any commercial use.
