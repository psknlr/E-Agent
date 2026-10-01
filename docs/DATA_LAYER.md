# Data Layer

How E-Agent organises the data behind a substrate-directed enzyme search, and
which specific failures each part of that organisation prevents.

**Scope and status.** Everything described here exists as code in
`src/eagent/datalayer/` and as registry configuration in `configs/datasources/`.
Nothing described here has been run against a live public database. The layers,
joins, identity ladder, gate chain, lineage grouping and snapshot format are a
**schema and a policy, awaiting real data**. Where this document says "the
system refuses X", it means there is a code path that raises or returns a typed
refusal — not that the refusal has been exercised on a real corpus.

| Module | What it holds |
| --- | --- |
| `datalayer/layers.py` | The six layers; the closed set of permitted joins; coverage accounting |
| `datalayer/registry.py` | The connector capability registry loaded from `configs/datasources/*.yaml` |
| `datalayer/identity.py` | The chemical identity ladder, name resolution, cofactor identity, protein identity, merge policy |
| `datalayer/preconditions.py` | The five-step gate chain that runs before catalytic geometry is ranked |
| `datalayer/lineage.py` | Independent-evidence counting and leakage-safe splitting |
| `datalayer/snapshot.py` | Freezing, verifying and diffing the data a round was decided on |
| `datalayer/intake.py` | Four-tier staged promotion of evidence, outcome normalisation, direction checking |
| `datalayer/house_db.py` | The project's own per-substrate database (see `docs/HOUSE_DATABASE.md`) |
| `datalayer/plan.py` | The staged rollout as typed, checkable work packages |

---

## 1. The six layers

A single "enzyme database" table silently mixes a nomenclature assignment, a
kinetic measurement and a predicted structure. A downstream ranker then treats
all three as the same kind of support. `DataLayer` splits them into six
questions that are deliberately **not** interchangeable. Each member carries the
question it answers, the agent stage it serves, and a statement of what it
cannot substitute for.

| Layer | Question it answers | Agent stage | What it is **not** a substitute for | Registered sources |
| --- | --- | --- | --- | --- |
| `reaction_and_chemistry` | What is the substrate, which bond changes, and what is the product configuration? | task normalisation, reaction search | A reaction record says nothing about whether any particular protein catalyses it. | 7 |
| `enzymology_evidence` | Which enzymes actually catalysed this or a near substrate, and under what conditions? | seed selection, activity evidence | An activity record for one substrate is not evidence for another substrate, another cofactor or another direction. | 6 |
| `sequence_family_evolution` | Which sequences are worth expanding, and which belong to the same mechanism family? | mining, clustering, diversity selection | Family membership is not activity on the target substrate, and cluster co-membership is not equivalence. | 11 |
| `structure_and_mechanism` | Can the substrate, the cofactor and the catalytic residues form a sensible arrangement? | complex modelling, geometry | A plausible geometry is a hypothesis, not a turnover number; a predicted complex is not an observed one. | 6 |
| `mutation_and_performance` | Which positions should change, and what might that change cost? | local design, directed evolution | A stability or binding measurement is not a measurement of activity on the target substrate. | 10 |
| `literature_and_feedback` | How do we obtain new evidence, and how does this project's own data re-enter the next decision? | literature agent, active learning | A machine-extracted relation is not a curated record, and a favourable thermodynamic result is not catalysis. | 10 |

Forty-nine distinct sources are registered; the counts above sum to fifty
because `enzengdb` genuinely serves two layers and is defined once with both
declared.

### Coverage is reported, never implied

`LayerCoverage` counts what each layer actually supplied for one task, and
`report_lines()` always lists **all six**, including the empty ones:

```
layer coverage for task <id> (thin below 3 records)
  ok     reaction_and_chemistry        4 records (0 experimental) from [rhea, pubchem]
  THIN   enzymology_evidence           1 records (1 experimental) from [brenda]
  EMPTY  mutation_and_performance      0 records (0 experimental) from [-]
           gap: no variant-effect data retrieved for this family
```

A ranked candidate table built on one thin layer looks exactly like one built on
six rich layers. `experimental_counts` is tracked apart from `counts` because
twelve annotation-only records and twelve measurements are not the same
coverage, and `is_single_source()` exists so a layer resting on one resource can
be reported as such.

---

## 2. Why layers are joined by identifiers and experiment records, not by similarity

The characteristic way an enzyme project dies is that a name-level match is
treated as an identity. "ADH from *Lactobacillus*" in a kinetics table,
"alcohol dehydrogenase" in a structure paper, and a 92 %-identical metagenome hit
are written into one row. The resulting row describes no real catalytic system,
and nothing downstream can recover, because the error is in the primary key.

So the permitted links are a **closed set** of nine identifier-based keys.
`JoinKey` enumerates them; each is deliberately composite where a bare
identifier is ambiguous.

| Join key | Fields | Establishes | Does **not** establish |
| --- | --- | --- | --- |
| `sequence_sha256` | `sequence_sha256` | the same amino-acid sequence | the same construct: tags, truncations and fusions change the expressed protein without changing the catalytic domain |
| `accession_with_database_version` | `accession`, `database_version` | the same database entry in the same release | the same sequence across releases: entries are merged, demerged and re-annotated |
| `rhea_reaction_id` | `rhea_id` | the same Rhea reaction entry | the same measured direction: an oxidation-direction record is not evidence of reduction activity |
| `chebi_id` | `chebi_id` | the same ChEBI entry | the same compound in the assay: a class entry covers many compounds; protonation and stereochemistry differ from the flask |
| `pubchem_cid` | `pubchem_cid` | the same PubChem compound identifier | the same salt form, charge state or enantiomer |
| `inchikey` | `inchikey` | the same connectivity, stereochemistry and protonation layer | equivalence under a skeleton-only (first-block) comparison, which discards exactly the stereochemistry this project optimises |
| `pdb_chain_sifts_residue` | `pdb_id`, `chain_id`, `sifts_residue` | the same residue of the same chain of the same entry, through an explicit SIFTS mapping | agreement of residue numbering: author, label and UniProt numbering diverge |
| `ccd_component_atom` | `ccd_component_id`, `atom_name` | the same atom of the same chemical component entry | the same cofactor oxidation state: `NAD`/`NAI` and `NAP`/`NDP` are different component ids for what prose calls one cofactor |
| `doi_with_experiment_activity_id` | `doi`, `experiment_activity_id` | the same measurement campaign in the same publication, and therefore **not** independent | independent corroboration — this key exists mainly to *collapse* duplicates |

`permitted_keys(a, b)` lets a planner discover that two layers share no
identifier and must stay unjoined, rather than reaching for a resemblance score
to bridge the gap.

### The refusal that gives the architecture its value

`refuse_similarity_join()` and `SimilarityJoinRefusedError` are the explicit,
greppable refusal path. `is_similarity_pseudo_key()` trips on the *shape* of the
request rather than on a list of tool names, matching stems including
`similar`, `identity`, `embed`, `cosine`, `tanimoto`, `rmsd`, `tmscore`,
`blast`, `mmseqs`, `knn`, `vector`, `nearest`, `homolog`, `cluster_member`,
`name_match` and `alignment_score`. A similarity tool invented tomorrow still
trips it as long as it is honest about what it computes.

The rule is not that resemblance is banned. Resemblance may **rank** candidates,
choose templates and order a shortlist. It may never **merge** two rows, because
once merged the error is in the primary key.

### Three outcomes, not two

`validate_join()` returns or raises three distinct things, kept apart on purpose:

* `joined=True` — every component present on both sides and equal after
  normalisation;
* `joined=False` — the records genuinely disagree, with `partial=True` when the
  primary identifier matched but a qualifier (release, chain, residue, atom) did
  not. That is exactly the case where a human reads "same accession" and merges
  anyway, so it is flagged rather than reported as a plain mismatch;
* `JoinFieldMissingError` — a record does not carry the identifier at all. An
  absent key is a data gap to resolve, not a negative answer. Collapsing this
  into `joined=False` is how "we could not check" becomes "we checked and they
  differ".

`validate_join` also attaches the key's `does_not_establish` caveat to the
result, so the limitation travels to the point of use rather than living in a
design document.

### Experiment records as the other join

Identifiers link *entries*. What links an enzyme to a claim about chemistry is
an **experiment record**: a construct, a substrate, a direction, conditions, a
detection method and an outcome class. `ExperimentRecord` and `EvidenceRef` in
`src/eagent/schemas/` are that unit, and the house database (`house_db.py`)
stores them. A candidate that is joined to a family, a structure and a reaction
but to no experiment record is a well-connected hypothesis, and the coverage
report says so.

### What may merge two records

`EntityMergePolicy` answers "may these two records be merged?" as a documented,
testable object. Admissible grounds are identifier-based. Inadmissible grounds —
name similarity, structural resemblance, substrate-name similarity — are
**enumerated rather than absent**, each with its reason, because they are the
grounds people actually use and a policy that silently lacks them cannot explain
a refusal. When no admissible ground holds, `evaluate()` additionally lists the
resemblance-based grounds that *would* have fired on those two records: that
list names the trap the caller was about to walk into.

`same_protein()` answers on sequence-hash equality and nothing else, returning
`IDENTICAL_SEQUENCE`, `RELATED_NOT_IDENTICAL` or `UNDETERMINED`. The case it
exists for is two records carrying the same accession and two different
sequences — a re-annotated entry, an isoform, a variant deposited under the
wild-type accession. Reading the shared accession as identity merges a wild type
with a mutant and then attributes the mutant's activity to both.

---

## 3. The chemical identity ladder

A prose substrate name becomes a stereo-defined, protonated, docked structure,
and nobody can say which step introduced which assumption. The ladder keeps five
rungs **separate**, each answering a different question with a different
authority.

| Rung | Question | What it is | What it is not |
| --- | --- | --- | --- |
| 0 `as_written` | What did the author actually write? | The author's description verbatim. The only rung that is evidence rather than interpretation. | A name is not a structure; it fixes neither stereochemistry, salt form nor protonation. |
| 1 `normalised_structure` | Which connectivity does that denote? | Connectivity resolved through a registered source. Stereochemistry may still be absent. | Not the assayed material; does not distinguish enantiomers. |
| 2 `stereo_defined_structure` | Which stereoisomer is meant? | Stereochemistry stated. For an asymmetric reduction this rung carries the entire objective. | Does not state the charge or protonation used in modelling. |
| 3 `charge_and_protonation_state` | Which microspecies, at which pH? | Net charge and protonation. Docking runs on one microspecies. | A modelling choice, not a measurement of the compound in the flask. |
| 4 `modelled_structure` | What was actually fed to the tool? | The structure after tool-side edits: added hydrogens, chosen tautomer, stripped counter-ion. | Not the author's compound. Differences from the rungs below are assumptions, not data. |

### Why the rungs stay separate

There is deliberately **no `.structure` property** on `ChemicalIdentityLadder`.
A caller must name the rung it wants, because "the substrate" means the author's
name to a curator, the stereo-defined structure to a chemist, and the
protonated, hydrogen-added ligand to a docking program. Silently returning the
last when the first was meant is how an enantiomeric-excess target gets attached
to the wrong molecule. `structure_for_modelling()` returns only
`MODELLED_STRUCTURE`, or `None` — never the nearest structural rung — so a
connectivity-only SMILES cannot be docked as though stereochemistry and
protonation had been decided.

`is_consistent()` walks the populated rungs and reports every place where a rung
states something the rung below it did not, classified as `structure_from_text`,
`stereochemistry`, `charge_state` or `modelling_edit`. An addition that cites no
source is reported as an **assumption**, so a reviewer can ask "who decided the
substrate was the (S) enantiomer?" and get an answer.

One flag matters more than the others:
`stereocentre_introduced_during_normalisation`. Normalisation is supposed to
canonicalise what the author wrote, not decide which enantiomer they meant, and
that flag is the specific thing the method exists to catch.

Structure strings are compared **textually**. No cheminformatics toolkit is
available in this environment, so two different writings of one molecule are
reported as a change needing curation rather than quietly accepted. That is the
safe direction for the error. `stereo_descriptors()` and `net_charge_of()`
return `None` — not a zero and not an empty result — whenever the representation
cannot be read, because "no stereochemistry stated" and "we cannot see the
stereochemistry" lead to different actions and only the second needs a curator.

`NameResolver` maps a name to an identifier **only** through registered sources,
with exact matching on the case- and whitespace-normalised string.
`normalise_query_name()` deliberately leaves `(R)-`, `(S)-`, `rac-`, salt
suffixes and numbering intact: the usual "normalisation" that strips them merges
enantiomers and salt forms, which for this project destroys the objective
itself. Unresolved names come back as a value carrying a question for the
operator (`UNRESOLVED_NOT_FOUND`, `UNRESOLVED_AMBIGUOUS`,
`UNRESOLVED_CONFLICTING_SOURCES`) rather than as an exception, so a pipeline can
put one question list in front of a human. `NameResolver.guess()` exists solely
to raise `NameGuessRefusedError`, so code tempted to fall back on a fuzzy match
gets a named error naming the compound.

---

## 4. The cofactor species-and-state rule

> A cofactor is recorded as a **specific species in a specific oxidation state**,
> or as `UNKNOWN` with a question. A family label is never promoted to a
> molecule.

`NADH`, `NADPH`, `NAD+` and `NADP+` are four different molecules differing in
phosphorylation and in redox state. A single "NAD-type cofactor" label loses
both distinctions at once: the specificity that decides whether an enzyme works
in a whole-cell system, and the oxidation state that decides whether the
reaction can run at all.

| Species | Backbone | State | Documented PDB component ids |
| --- | --- | --- | --- |
| `NADH` | NAD | reduced | `NAI` |
| `NAD+` | NAD | oxidized | `NAD` |
| `NADPH` | NADP | reduced | `NDP` |
| `NADP+` | NADP | oxidized | `NAP` |
| `FAD` / `FADH2` | FAD | oxidized / reduced | *(none recorded — needs curation)* |
| `FMN` / `FMNH2` | FMN | oxidized / reduced | *(none recorded — needs curation)* |
| `UNKNOWN` | — | unknown | — |

Species whose component ids are not documented in this repository's chemistry
module carry an **empty tuple rather than a guessed code**, because a guessed
code would be read back as a fact by the structure layer.

`CofactorIdentity.from_label()` refuses the family labels listed in
`LOOSE_COFACTOR_LABELS` — including `"NAD(P)H"`, `"nicotinamide cofactor"`,
`"pyridine nucleotide"` and a bare `"NAD"` — returning `UNKNOWN` with the
question attached. `from_ligand_code()` recognises only documented component
ids; an unrecognised code produces `UNKNOWN`, never a plausible species,
because the component dictionary is large and a wrong expansion silently flips
an oxidation state.

The redox question is answered in two forms:

* `hydride_donor_answer()` returns a tri-state `HydrideDonorAnswer` where
  `answered=False` is a **refusal, not a negative**, so a gate can record "state
  unknown, cannot evaluate";
* `is_hydride_donor()` raises `CofactorStateUnknownError` rather than returning
  `False`. A silent `False` rejects a correct candidate; a silent `True`
  validates a hydride-transfer geometry measured against NAD+.

`satisfies()` compares an observed cofactor against a mechanism requirement and
returns `satisfied=None` whenever either side is unknown, so the gate reports
"not evaluated" and the candidate goes back for repair instead of being failed
on missing information. `allow_backbone_substitution=True` permits NADH where
NADPH is required — a legitimate specificity question for an engineering
campaign — but **never** permits an oxidation-state substitution, which is a
mechanism error rather than a preference.

---

## 5. The precondition gate chain

A final candidate ranking is a list of numbers: hydride-transfer distances,
attack angles, clash counts. Those numbers are produced by measuring a model,
and a model can be measured perfectly while describing the wrong thing. Four
mapping errors account for most of it, and each produces a *plausible number*.
Nothing about a 3.1 Å distance says which atoms it was measured between, so
these errors do not show up as noise — they show up as signal, and they reorder
the ranking.

| # | Gate | Adds scientific insight? | The failure it removes |
| --- | --- | --- | --- |
| 1 | `sequence_consistency` | **no** | A homologue, a point mutant or a tagged construct is measured and reported as the candidate; every later number describes a different protein. |
| 2 | `residue_numbering` | **no** | A template residue lands on whichever residue carries that number in this file, so the "catalytic tyrosine" is a different residue and the mutation order names the wrong position. |
| 3 | `ligand_atom_mapping` | **no** | Two files listing one ligand's atoms in different orders are mapped positionally, so the hydride-donor carbon is silently some other atom. |
| 4 | `cofactor_state` | **no** | A hydride-transfer geometry is measured against NAD(P)+, which cannot donate a hydride: the geometry is fine and the chemistry is impossible. |
| 5 | `catalytic_geometry` | **yes** | The only step that measures chemistry, and only meaningful once the four above have passed. |

### These gates add no insight by themselves

This is the point, and `MAPPING_GATES_RATIONALE` states it in one text that
report writers and the CLI quote verbatim rather than paraphrase:

> Steps 1 to 4 (sequence consistency, residue numbering, ligand atom mapping,
> cofactor state) add no scientific insight by themselves. Passing them says
> nothing about whether the enzyme catalyses the reaction. Their entire value is
> removing mapping errors that would otherwise change the final ranking while
> looking like real signal: a distance measured to the wrong atom, a residue
> identified by the wrong number, or a hydride transfer modelled from an
> oxidised cofactor all produce plausible numbers that no downstream statistic
> can detect.

The order is not arbitrary — each gate presupposes the previous one. A
residue-numbering map built against the wrong sequence is a correct map of the
wrong protein, and an atom mapping validated against the wrong ligand is worse
than no mapping at all, because it looks checked.

### Three outcomes per gate, and no discard

`GateOutcome` is `PASSED`, `FAILED` or `NOT_EVALUATED`. Folding `NOT_EVALUATED`
into `FAILED` discards candidates for missing inputs; folding it into `PASSED`
lets an unchecked candidate into the ranking. Both have happened in real
pipelines and both are invisible afterwards.

`PreconditionChainReport.discard_allowed` is **always `False`**. A gate failure
is a statement about the *model*, not about the enzyme, so the candidate is
neither ranked nor dropped: every failure carries a machine-readable
`RepairAction` and a `rerun_from` step. A shortlist can then say "nine ranked,
four held for repair" instead of quietly presenting nine. Steps after the first
failure are recorded as `not_evaluated` with `not_reached=True`, so the report
never implies a skipped gate was checked.

Two further refusals live in gate 5. A constraint with no measurement is
`not_evaluated`, never "not satisfied". And a constraint that was **restrained**
while the pose was built is marked circular and excluded from the independent
count: a distance held at 3.0 Å by the modelling protocol is not evidence that
the enzyme holds it there.

`check_atom_mapping()` deserves a note of its own. It maps by atom name, refuses
any mapping declared on file order (`POSITIONAL_MAPPING_BASES`), and reports
`positional_would_mismatch` — the actual pairs a positional read would have
produced, so a reviewer sees "C4 would have been paired with N1" rather than an
abstract warning about atom order.

---

## 6. Data lineage: counting measurements, not rows

One measurement, in one paper, is curated into BRENDA. OED re-integrates BRENDA.
SKiD re-integrates both. CatPred-DB re-integrates all three. A naive pipeline
retrieves four rows that agree and reads the claim as well supported by four
records. It is **one experiment**. The agreement is an artefact of copying.

### Identity tokens, in three tiers

`identity_tokens()` assigns each row the highest tier it can fill:

1. `experiment_activity` — the measurement campaign id, the only identifier that
   survives re-curation intact;
2. `publication` — a normalised DOI (`normalise_doi()` folds
   `https://doi.org/`, `doi:` and case variations into one string) or another
   publication identifier;
3. `assay_fingerprint` — (sequence, substrate, conditions, outcome, measured
   value), the last resort for bulk dumps carrying no citation. Returns `None`
   when the row has no outcome and no number, because there is then nothing that
   could show it to be a re-report.

`link_across_tiers=True` (the default) lets a row carrying both an activity id
and a DOI bridge to a re-curated copy that kept only the DOI — the common real
case, since the activity id is the first field a re-integrating resource loses.
The cost is that two distinct activities in one paper collapse into one group.
That direction is safe: they share the paper's selection and analysis decisions
and are not independent corroboration anyway. The fingerprint tier never runs
alongside a publication tier, because two genuine replications in different
papers can easily report the same conversion under the same conditions.

Rows that fill no tier come back as `unlinkable` one-row groups. They are
**never merged on suspicion and never counted as corroboration**, and they are
named in the report so a curator can resolve them. Under-counting independence
is the safe direction; every ambiguous case resolves toward "fewer independent
measurements".

### Worked example: one measurement in four databases

Four `ExperimentRecord`s describing one campaign, as BRENDA, OED, SKiD and
CatPred-DB would each present it. Output below is the actual rendering of
`LineageReport.build(...)` on those four rows:

```
claim: this ADH reduces 4-chloroacetophenone to the (S)-alcohol
rows retrieved:           4
independent measurements: 1
corroboration:            weak
upstream resources:       activity:campaign-7, brenda, sabio_rk
groups:
  grp:8a83f09d07c7  [experiment_activity]  rows=4  strength=homolog_experimental  upstream=activity:campaign-7
discounted rows:
  r_oed (grp:8a83f09d07c7): re-report of the same measurement (matched on experiment_activity); counted once through r_brenda
  r_skid (grp:8a83f09d07c7): re-report of the same measurement (matched on experiment_activity); counted once through r_brenda
  r_catpred (grp:8a83f09d07c7): re-report of the same measurement (matched on experiment_activity); counted once through r_brenda
```

Three things are worth reading carefully. The row count and the independent
count sit **side by side**, so "supported by four records" cannot be written
without "which are one independent measurement". Every discounted row names
*which* row it was counted through, so the arithmetic is checkable months later.
And the corroboration level is `weak`, not `strong`: a single homolog-level
group cannot reach `strong` however many copies of it exist.

`corroboration_level()` rises only with genuinely independent groups:

| Level | Condition |
| --- | --- |
| `STRONG` | two or more independent sequence-level experimental groups |
| `MODERATE` | one sequence-level group, or two or more homolog-level-or-better groups |
| `WEAK` | one homolog-level group, or any EC-species-mapped group |
| `INSUFFICIENT` | nothing contributing |
| `CONTRADICTORY` | independent groups disagree, or a group's own rows disagree |

`CONTRADICTORY` is returned instead of a majority vote. A conflict between
independent experiments is information for a human, not noise to be averaged.
With `require_experimental=True` (the default), rows whose outcome is
`NOT_TESTED`, `COMPUTATIONAL_FAILURE`, `COMPUTATIONAL_NEGATIVE` or
`EXPRESSION_OR_SOLUBILITY_FAILURE` do not contribute at all: a modelling failure
and an unexpressed construct are statements about the pipeline, not about the
enzyme.

### The same defect ruins model evaluation

"Train on database A, test on database B" is not an independent test when B
re-curated A: the test rows are the training rows wearing a different accession.
`grouping_key()` therefore returns facets over the **original measurement** —
publication, experiment activity, parent sequence lineage, sequence cluster —
and the **source database is deliberately absent**. An unresolved cluster becomes
`cluster:unresolved:<sequence id>` rather than a shared `cluster:unknown`,
because a shared placeholder would fuse every unclustered sequence into one
giant group that looks conservative and is meaningless.

`leakage_safe_groups()` takes the transitive closure over those facets (A shares
a paper with B, B shares a cluster with C, so all three stay on one side), and
`split_leakage()` reports the rows that cross a proposed boundary. Comparing
`grouping_key` tuples for equality alone still leaks when rows overlap on one
facet only.

`ProvenanceGraph` is built only from fields that actually travel with the data —
`EvidenceRef.source_doi`, `experiment_activity_id`, `upstream_sources` and the
database record id — so it never asserts a lineage nobody recorded. Its
traversal is explicitly cycle-safe, because re-integrating resources routinely
cite each other; cycles are reported through `cycles()`, not followed.

### Lineage between *resources*, as opposed to between rows

`SourceRegistry.independent_source_groups()` does the same job one level up, over
the registry's `derived_from` declarations. On the current registry it collapses
49 sources into 28 independent groups. The notable collapse is
`brenda, catpred_db, enzymemap, oed, sabio_rk` into one group — four of those
re-integrate one or both of the first two. `derived_from_complete=False` marks a
source whose upstream list is known to be incomplete; 23 entries carry that flag
today, and the report names each one as "must not be counted as independent
corroboration".

---

## 7. Snapshot freezing

Every formal screening round runs against a fixed snapshot. Months later, "why
was this enzyme chosen and that one not" has an answer only if the inputs are
pinned. Two failures motivate the module:

* **Unattributable change.** Round 2 has a different hit rate from round 1. Was
  that the method or the data? Without a snapshot per round and a `diff()`
  between them, the question cannot be answered and the round teaches nothing.
* **Silent filtering.** A cleaning step quietly drops 40 % of the rows — all the
  negatives, say, or every entry from one organism — and the surviving dataset
  looks clean and is biased beyond repair.

### What a frozen round records

`DatasetSnapshot` holds:

| Field | Why it is there |
| --- | --- |
| `snapshot_id`, `created_at`, `round_label` | Identity of the round |
| `sources[]` → `SourceSnapshot` | Per source: `version`, `retrieved_at`, pinned `files[]` (path, **sha256**, size, mtime), the ordered `cleaning_rules[]`, `n_records_before` / `n_records_after`, `exclusions` by reason, `exclusions_by_rule`, `license`, `needs_curation`, `curation_notes[]` |
| `model_versions` | Which models produced the predictions |
| `random_seed` | So a stochastic selection is reproducible |
| `selection_policy_id` | Which policy spent the batch's slots |
| `evidence_chains[]` | Why each finally selected candidate was selected, pinned to the same frozen data as the decision |
| `content_sha256` | Hash of the *content* payload, excluding `created_at` and the id, so two freezes of identical data hash identically |
| `needs_curation`, `curation_notes[]`, `notes[]` | What a human still has to confirm |

The path alone is worthless as provenance — files are overwritten in place by
the next download — so `FileRef` carries the checksum, which is what makes "the
decision was made on this data" a checkable statement.

Cleaning rules are **versioned**: "we removed duplicates" is not a reproducible
statement, and `CleaningRule.label()` is `name@version` so that a diff can say
"the data is the same but `drop_fragments` went from v1 to v2".

### Dropping a record without logging it is not an available code path

`CleaningPipeline` has no entry point that takes a plain boolean predicate. A
rule reports *why* it rejects a record, and `run()` writes that reason to the
`ExclusionLog` before the record disappears. `ExclusionLog.record_exclusion()`
rejects an empty reason rather than storing it, because a log full of blank
reasons is indistinguishable from no log. `freeze(strict=True)` then checks the
identity `n_before − n_excluded == n_after` and **raises** on a discrepancy:
records that left without passing through the log are the silent filtering the
module forbids. `strict=False` records the discrepancy as a curation note, for
the case where a legacy source is being ingested and the gap itself is the
finding.

An unknown `version` or `retrieved_at` stays `None` with `needs_curation` set
and a note naming what must be confirmed. It is never replaced by "latest", by
today's date, or by anything else invented in the freeze.

### Verify and diff

`verify()` re-hashes every pinned file and returns a `VerificationReport` rather
than raising — the caller decides whether a changed input invalidates the round
or is an expected re-download. What it must not do is proceed as though the data
were the one that was frozen. `DatasetSnapshot.load()` is stricter: it recomputes
`content_sha256` and refuses a hand-edited file, because such a file carries the
authority of a record while describing data that was never used.

`diff()` splits every change into **data changes** and **method changes**:

* data: sources added or removed, version strings, per-file checksums, record
  counts, exclusion-reason counts, and the ordered cleaning-rule list — a
  reordered pipeline counts, because order changes which rule is credited with
  each drop;
* method: model versions, the random seed, the selection policy.

`SnapshotDiff.attribution()` then states plainly what a difference in results may
be attributed to. If the data side is empty, the method moved the number; if the
method side is empty, the data did; if both changed, the comparison does not
support a causal claim and the diff says so.

---

## 8. The expanded no-result taxonomy

A binary active/inactive label collapses most of the information in a screening
campaign. `OutcomeClass` has seven members: one confirmation and **six distinct
ways of not getting the target product**.

| Class | What the record is entitled to claim | Experimental? | Informs catalytic ability? |
| --- | --- | --- | --- |
| `confirmed_target_product` | target reaction activity exists under the recorded conditions | yes | **yes** |
| `no_target_product_detected` | no activity detected under these conditions at this detection limit | yes | **yes** |
| `other_product_or_wrong_configuration` | turnover occurred but does not meet the target reaction requirement | yes | **yes** |
| `expression_or_solubility_failure` | this construct did not pass expression; catalytic ability undetermined | yes | no |
| `not_tested` | unknown | no | no |
| `computational_failure` | modelling or tooling produced no usable result; says nothing about the enzyme | no | no |
| `computational_negative` | model training label only; not an experimental negative | no | no |

Three properties carry the logic. `is_experimental` is false for `not_tested`,
`computational_failure` and `computational_negative`.
`informs_catalytic_ability` is true for exactly the three classes where an assay
ran on soluble protein and produced a verdict about chemistry — notably **not**
for `expression_or_solubility_failure`, because a construct that never expressed
tells you nothing about the enzyme. `is_positive` is true only for
`confirmed_target_product`.

### Computational failure versus experimental negative

This is the distinction the taxonomy was expanded for.

* `computational_failure` — docking produced no pose, a structure prediction did
  not converge, a job timed out, a tool errored. **The modelling produced
  nothing. It says nothing whatever about the enzyme.** A candidate with this
  outcome has not been shown to be bad; it has been shown that the pipeline
  failed on it. Treating it as a negative label trains a model to predict tool
  failures.
* `computational_negative` — a model predicted inactivity. This is a statement
  about the model, usable as a training label and as triage, and it is **not** an
  experimental negative. No number of these can establish that an enzyme does not
  work.
* `no_target_product_detected` — the only genuine experimental negative, and the
  house database refuses to store one **without a recorded limit of detection**,
  because "no product" at an unstated limit bounds nothing.

`EXPRESSION_OR_SOLUBILITY_FAILURE` sits apart from all three: the assay never
got a chance to run, so the row belongs in the submitted-constructs denominator
and not in the catalysis denominator.

Where these classes bite:

* `lineage.corroboration_level()` excludes `not_tested`,
  `computational_failure`, `computational_negative` and
  `expression_or_solubility_failure` from the evidence count;
* `house_db.prediction_vs_outcome()` excludes rows that cannot inform catalysis
  from top-k rates and counts them separately — a top-ranked candidate that never
  expressed is a cloning result, not a wrong chemical prediction;
* `house_db.hit_rate()` reports expression failures and unassessed constructs
  beside both denominators;
* `house_db` refuses an `expression_or_solubility_failure` row that also carries
  a catalytic measurement, and refuses a `not_tested` row that carries any
  measurement at all. Protein that did not express cannot have been assayed.

### Normalising messy source statements

`intake.normalise_outcome()` maps source prose onto these classes and **refuses
to guess**. The known-ambiguous list is explicit, with a reason recorded per
phrase, and resolves to `NOT_TESTED` with the uncertainty attached rather than to
a negative. `"n.d."` means "not detected" in one paper and "not determined" in
the next — a negative and an absence of a test. `"trace"`, `"low"`, `"weak"`,
`"racemic"`, `"yes"`, `"no"` and `"negative"` are all on the refusal list, each
with the specific reason it cannot be resolved without more context.

`intake.direction_check()` flags records measured in the reverse of the target
direction. An alcohol dehydrogenase assayed on the alcohol, following NAD+
reduction, is a superb record of the oxidation and says nothing dependable about
the ketone reduction at the target pH with the target cofactor. Such records
arrive from curated resources with the same EC number and the same substrate
name, so without this check they enter a seed set as positives. Two signals are
used — the declared `ReactionDirection` and whether the reaction class is the
chemical reverse of the target — and a disagreement between them is itself a
refusal. `UNSPECIFIED` is non-supporting, because an unrecorded direction is not
a forward one.

### Four intake tiers, and no automatic promotion

Tier answers "who produced this row and did a person check it"; the
`EvidenceStrength` ladder answers "how tightly does the claim bind to this exact
sequence". They are different axes and are stored apart.

| Tier | Storage partition | Strongest strength an *automated* ingest may stamp | Human-checked? | Evidence about the world? |
| --- | --- | --- | --- | --- |
| `expert_verified_primary` | `tier/expert_verified_primary` | `sequence_level_experimental` | yes | yes |
| `curated_database` | `tier/curated_database` | `homolog_experimental` | no | yes |
| `machine_extracted_pending` | `tier/machine_extracted_pending` | `ec_species_mapped` | no | yes |
| `model_inferred` | `tier/model_inferred` | `computational_construct` | no | **no** |

> evidence is never promoted automatically: no confidence score, no source
> agreement count and no re-curation raises a record's tier. The only path upward
> is `promote()`, which records a named human reviewer and a justification.

`promote()` refuses a reviewer string that names software (`model`, `agent`,
`pipeline`, `auto`, `bot`, `script`, …), refuses a no-op promotion that would
write an audit entry recording a decision nobody made, and refuses to exceed the
registered source's `evidence_strength_ceiling` unless the reviewer attaches the
primary `EvidenceRef` they actually read. BRENDA cannot justify a claim stronger
than BRENDA supports; the paper behind it can. Promotion out of `model_inferred`
is refused **unconditionally**: reviewing an inference confirms that the model
said so, which is a different claim, and the correct action is to run the
experiment and ingest the result as a new record.

`IntakeStore` partitions by tier, so merging is an explicit act: `pool()` demands
the tiers by name plus a justification, and `allow_model_inferred=True` on top of
that before predictions share a bag with measurements.

---

## 9. What this layer does not do

* Nothing here has been connectivity-tested. Every registry entry carries
  `connectivity_verified=false`; see `docs/DATASOURCES.md`.
* No chemistry is computed. Structure comparison is textual, charge is read from
  bracket atoms, stereo markers are counted from the string. There is no RDKit in
  this environment, and the module reports "this changed, a curator must say why"
  rather than deciding that two differently written SMILES are one molecule.
* No measurement is produced. `preconditions.check_catalytic_geometry()`
  *consumes* measurements and judges them; the numbers come from elsewhere.
* The staged rollout in `plan.py` is typed and checkable, but `readiness()`
  treats nothing as reachable until an operator asserts it. Passing
  `available_sources=None` reports every source-dependent package as blocked.
  That is the honest starting state of a fresh checkout.

---

## 中文摘要

### 为什么要分成六层

一个"酶数据库"表格会把命名法标注、动力学测量和预测结构混在一起，下游排序器于是把三者
当成同一种支持证据。`DataLayer` 把它们拆成六个**不可互相替代**的问题：反应与化学、
酶学证据、序列家族与进化、结构与机制、突变与性能、文献与反馈。每一层都显式记录"它不能
替代什么"——例如家族归属不等于对目标底物有活性，一条活性记录也不能转用到另一个底物、
另一个辅因子或另一个反应方向。覆盖度报告（`LayerCoverage`）永远列出全部六层，包括为空
的层，因为建立在一个薄弱层上的候选排名，看起来和建立在六个充实层上的完全一样。

### 为什么只能按标识符和实验记录连接

项目失败的典型方式是把"名字相同"当成"同一个东西"。因此允许的连接是**封闭的九个标识符
键**（序列 SHA256、带版本的登录号、Rhea 反应号、ChEBI、PubChem CID、完整 InChIKey、
PDB 链 + SIFTS 残基、CCD 组分 + 原子名、DOI + 实验活动号）。每个键都附带"它能证明什么"
与"它不能证明什么"，并随连接结果一起传递。任何基于相似度、序列一致性、嵌入向量、RMSD、
BLAST 命中或名称距离的"连接"都会抛出 `SimilarityJoinRefusedError`。

规则不是禁止相似度，而是：相似度可以用来**排序**候选、挑选模板，但永远不能用来**合并**
两行记录。一旦合并，错误就落在主键上，下游无法挽回。

`validate_join` 返回三种结果而不是两种：匹配、确实不匹配（当主标识符相同而限定符不同
时标记 `partial`）、以及**字段缺失抛错**。把后两者合成一个布尔值，就是把"我们没能检查"
变成"我们检查过且不一致"。

### 化学身份阶梯

小分子身份分成五级并**永不折叠**：作者原文 → 规范化连接 → 立体定义结构 → 电荷与质子化
状态 → 实际送入建模的结构。`ChemicalIdentityLadder` 故意**没有** `.structure` 属性，
调用方必须指明要哪一级。`is_consistent()` 报告每一处"上一级比下一级多说了什么"，其中
`stereocentre_introduced_during_normalisation` 这一标志最关键：规范化只应把作者写的
东西标准化，而不应替作者决定是哪个对映体——这正是不对称还原项目的全部目标所在。

### 辅因子必须记物种与氧化态

NADH、NADPH、NAD+、NADP+ 是四种不同分子，"NAD 型辅因子"这一个标签同时丢掉了磷酸化
specificity 和氧化态。本项目只接受具体物种加具体氧化态，否则记为 `UNKNOWN` 并附上必须
向人提出的问题。只有仓库化学模块中已记载的 PDB 组分号被识别（NAI / NAD / NDP / NAP）；
未记载的物种留空元组而非猜一个代码。`is_hydride_donor()` 在状态未知时**抛错而不是返回
False**：静默的 False 会错杀正确候选，静默的 True 会让针对 NAD+ 测得的氢负离子转移几何
通过验证。允许 NADH 替代 NADPH（这是一个正当的特异性工程问题），但**绝不允许氧化态替代**。

### 前置门链

催化几何排名之前必须依次通过四道门：序列一致性、残基编号映射、配体原子级映射、辅因子
状态。**这四道门本身不提供任何科学洞见**——通过它们并不说明酶能不能催化反应。它们的全部
价值是负向的：排除那些本来会伪装成信号、却会改变最终排名的映射错误。一个 3.1 Å 的距离
本身不会告诉你它是在哪两个原子之间量的，所以这类错误不表现为噪声，而表现为信号。

每道门有三种结果（通过 / 失败 / 未评估），而不是两种。失败的候选**不允许被丢弃**
（`discard_allowed` 恒为 `False`）：门失败是对模型的判断，不是对酶的判断，候选带着可执行
的修复动作和重跑步骤返回。

### 数据血缘

一次测量被 BRENDA 收录，OED 再整合 BRENDA，SKiD 再整合两者，CatPred-DB 再整合三者——
检索到四行一致的记录，实际上只有一次实验。`LineageReport` 把"检索到的行数"和"独立测量
数"并排输出（上文实例：4 行 → 1 次独立测量，证据级别 weak），并逐行写明每一行是通过哪
一行被计入的。模型评估同理："在 A 库训练、在 B 库测试"在 B 重新收录了 A 的情况下不是独立
测试集，所以切分必须按原始测量（文献、实验活动、亲本谱系、序列簇）进行，**数据库来源被
刻意排除在切分键之外**。

### 快照冻结

每一轮正式筛选对着一个冻结快照运行。快照记录：每个来源的版本与获取时间、每个文件的
**sha256 校验和**、有序且带版本号的清洗规则、清洗前后的记录数、按原因与按规则统计的排除
计数、模型版本、随机种子、选择策略号，以及每个入选候选的证据链。删除记录而不写入排除
日志**在代码里不存在这条路径**；`freeze(strict=True)` 会校验
`清洗前 − 排除数 = 清洗后` 并在不符时抛错。`diff()` 把两轮之间的差异分成"数据变化"和
"方法变化"，并直接给出一句归因结论：若只有数据变了，结果差异不能归功于方法。

### 无结果分类法

`OutcomeClass` 有七个取值：一个确认，加上**六种"没有得到目标产物"的不同情形**。其中最
关键的区分是：

* `computational_failure`（计算失败）——对接没出构象、结构预测没收敛、任务超时。
  **建模什么也没产出，这对酶本身一无所言。** 把它当作阴性标签，等于训练模型去预测工具故障。
* `computational_negative`（计算阴性）——模型预测无活性。这是关于模型的陈述，可作训练
  标签和分流依据，**不是实验阴性**。
* `no_target_product_detected`（未检出目标产物）——唯一真正的实验阴性，而且本项目的
  自建数据库**拒绝在没有检出限的情况下存储它**：没有检出限的"未检出"界定不了任何东西。
* `expression_or_solubility_failure`（表达或可溶性失败）——测定根本没机会进行，所以它
  进入"提交构建体"分母，而不进入"催化能力"分母。

### 诚实说明

以上全部是**模式与规则，尚无真实数据**。注册表中没有任何一条做过连通性测试；本环境中没有
RDKit，结构比较是文本比较；`plan.readiness()` 在操作者未明确声明某来源可达之前，一律报告
为阻塞——这是一个全新检出副本应有的诚实初始状态。
