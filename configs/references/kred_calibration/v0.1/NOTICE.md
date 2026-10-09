# Provenance and reuse terms — KRED calibration reference set v0.1

## What was delivered, and by whom

`source/KRED_Calibration_Reference_v0.1.xlsx` is a compilation prepared by an AI
research assistant (audit date 2026-10-07) and supplied to this repository's
owner, who asked for it to be added. The workbook states no licence of its own.
This repository stores it unchanged (sha256 in `MANIFEST.json`) and asserts no
licence for the compilation. Its own usage sheet says: *do not describe the
whole package as CC0* — the terms below are per source, and they differ.

The delivery also described a ~12.2 MB ZIP (raw mmCIF files, validation XML,
reference sequences, `data/assay_conditions.json`, `activity_expansion_2026/`).
**That ZIP was not received.** Only the workbook was. Consequences, all recorded
rather than worked around:

* the 19 coordinate files are not in this directory; they are fetched from the
  RCSB by `eagent reference fetch-coordinates` and pinned by sha256 in
  `coordinates/coordinates.manifest.json`, so the files used are the files
  whose hashes are recorded, not whichever the RCSB serves later;
* the full concentration scans and replicate columns (`assay_conditions.json`),
  the 38 reference-sequence records and the 2026 Ssal-KRED ortholog extension
  (109 sequences, 276 enzyme–substrate endpoint pairs) are **not** in this
  repository. The workbook's auxiliary sheet summarises the extension, and that
  summary is all that is kept; the two substrate identity problems it reports
  (1a, 5a) are carried as reported and nothing is built on them.

## Terms by source

"Read" = this project's build session opened the provider's own page or the
paper's own full text on 2026-10-08 and the sentence is there. "Reported" =
the workbook says so and nothing here has checked it.

| source_id | what is taken from it | terms | status |
|---|---|---|---|
| wwPDB | coordinates, assemblies, validation statistics (RSCC etc.) | CC0 1.0 — "Data files contained in the PDB archive are available under the CC0 1.0 Universal Public Domain Dedication"; attribution to the structure authors is encouraged | **read** (wwpdb.org/about/usage-policies) |
| BRENDA (`LbADH_BRENDA`, `TRII_BRENDA`) | kcat, Km, pH, temperature of three records | CC BY 4.0, Release 2026.1 | **read** (both literature pages carry the statement) |
| HBDH2018_SI (Figshare 10.1021/acs.biochem.8b01099.s001) | four kcat/Km rows at 283 K (factual extraction only; no SI text is stored here) | **CC BY-NC 4.0** — the Figshare record states this. Non-commercial: a reader planning commercial use of anything derived from these four rows must check the SI terms themselves | **read** (Figshare record metadata) |
| HBDH2018 (main paper) | context, assay description | CC BY (open access article) | **read** (full text) |
| HBDH2020 | four mutant kcat/Km rows, structures 6ZZO/P/Q/S | CC BY (open access article) | **read** (full text) |
| Ssal2024 | seven apparent kcat/Km rows (main Table 2) | article CC BY 4.0 | **read** (full text). The authors' GitHub repository was reported to carry no LICENSE file; nothing from it is used here |
| PNAS2015 | eight reported efficiencies, two ND, ten er values (Table 1) | **not established.** The paper is publicly readable on PMC, but the PMC page carries only PMC's own copyright notice and no open licence. Hosting on PMC does not make it CC BY | **read** (absence), consistent with the workbook. The numbers are facts transcribed from a table; no text or figure is reproduced |
| LbADH2005, LbADH table reprint | the two LbADH kinetic rows as corroborated by BRENDA and by a university thesis reprint | original full text not accessed; the thesis is public but nothing from it is stored beyond the numbers | **reported** |
| TRII2003, TRII1999 | the tropinone reductase II structures' descriptions | PDB data CC0; papers not reproduced | **reported** |
| Sm2020 | one approximate specific-activity value read from a bar chart (6XEW) | open-access paper | **reported** |
| TeSADH2022 | the reason 7UTC/7UUT are excluded | PDB data CC0 | **reported** |
| Ssal2005 | 1Y1P description | PDB data CC0 | **reported** |
| Ortholog2026, Ortholog2026_Zenodo, Ortholog2026_GitHub | counts and the two chemical-identity problems only | paper is an *early accepted version* as retrieved 2026-10-07; Zenodo CC BY 4.0; GitHub MIT | **reported** |

## What follows from this table

* **Redistribution.** Everything *stored in this directory* is either factual
  data (numbers transcribed from a table, PDB metadata) or the compiler's own
  annotation. Nothing is a copy of a paper's text, figures or supporting
  information. The one term with a restriction attached is the HBDH 2018 SI
  (CC BY-NC): four numbers, extracted as facts, with their source named.
* **Commercial use.** If the project is used commercially, the HBDH2018_SI rows
  are the ones to take legal advice on, and the PNAS2015 values (no licence
  established) are the others. Neither affects the structures or the BRENDA
  rows. `eagent reference audit` lists the records by source so they can be
  excluded in one line.
* **Attribution.** Cite the original papers and the PDB entries, not the
  workbook. The `sources` sheet (`tables/sources.csv`) has the DOIs.
