# Source verification record

`results.json` is the output of `eagent reference verify-sources`: every number of
the 28 kinetic records and the 10 `er` values, compared with the printed table it
was taken from, by a parser written for that one table.

It records, for each of six documents, the sha256 of the **bytes that were read**.
The documents are not committed (they are other people's papers, and the HBDH 2018
SI is CC BY-NC) and are not fetched by the code. To repeat the check, download them
into one directory under these names and run
`eagent reference verify-sources --docs <dir>`:

| file name | what | where from |
|---|---|---|
| `PMC10902378.xml` | Ssal2024, Table 2 | `https://europepmc.org/api/service/rest/PMC10902378/fullTextXML` |
| `PMC7773212.xml` | HBDH2020, Table 4 | `https://europepmc.org/api/service/rest/PMC7773212/fullTextXML` |
| `PMC4697376.html` | PNAS2015, Table 1 | `https://pmc.ncbi.nlm.nih.gov/articles/PMC4697376/` |
| `hbdh2018_si.txt` | HBDH2018 SI, Tables S1–S4 | the SI PDF (Figshare 10.1021/acs.biochem.8b01099.s001, md5 `8bdb217983edb4e915d65a5f48d406ba`) through `pdftotext -layout` |
| `brenda_654707.html` | TR-II, BRENDA literature 654707 | `https://www.brenda-enzymes.org/literature.php?e=1.1.1.236&r=654707` |
| `brenda_675348.html` | LbADH, BRENDA literature 675348 | `https://brenda-enzymes.info/literature.php?e=1.1.1.2&r=675348` |

A page that has been re-rendered since will hash differently; the values are what is
compared, and the hash says which rendering the committed record is about.

## What the record does and does not say

* **25 of the 28 records** match the paper's own table for every quantity compared
  (Ssal-KRED Table 2 ×7, the HBDH 2020 mutant table ×4, the HBDH 2018 SI 283 K rows ×4,
  the PNAS 2015 efficiencies ×10 including the two `ND`). All ten `er` values match.
* **1 record** (the tropinone reductase II row) matches **BRENDA's extraction** of the
  2003 paper only. BRENDA is a second reader of the paper, not the paper; the paper's
  table was not read.
* **2 records** (the LbADH rows) match BRENDA for `kcat` and `Km`. Their efficiencies
  (16000, 3300 M⁻¹ s⁻¹) and the `±` terms come from a university thesis that reprints
  the paper's table; neither was opened, and those six quantities are `not_checked`.
* **Nothing mismatched.** That is a statement about transcription. It is not a
  statement that the numbers are right: the sources' own caveats (an apparent
  `kcat` in an iPrOH-regeneration assay, a `Km` printed as `<5400` in a column
  labelled mM, an unsaturated M4 fit) travel with the records, in `quality_flags`.
