# Data Sources

The registered public resources behind E-Agent: what each is documented to
offer, what it must never be used to claim, and what nobody has checked yet.

---

## Read this first

> **Three entries of 49 have been connectivity-tested; the rest have not.**
> UniProtKB, RCSB PDB and Rhea were called by `eagent sources verify` and carry
> the record of those calls — URL, timestamp, status, response digest, and the
> strings that had to appear in the body. Every other `DataSource` carries
> `connectivity_verified: false`.
>
> `true` is admissible only alongside such a record. A bare boolean asserting
> "this works" is worth nothing, which is why the model used to refuse `true`
> outright; a pass says *this capability, at this URL, answered correctly at
> this time, from this environment*, and nothing more. Nothing below has been
> downloaded in bulk, authenticated against or benchmarked here.
>
> **Capability flags describe documentation, not verified behaviour.** A flag
> that says `bulk_snapshot: supported` means a bulk route is documented
> somewhere, not that it works, not that it is current, and not that it is
> reachable from this network.
>
> **Every uncertain value is `null` with `needs_curation: true`.** Endpoint
> URLs, release strings, record counts and licence terms are left null rather
> than guessed. All 49 entries carry `needs_curation: true`, and each one lists
> in `curation_notes` exactly what a curator must confirm — the model refuses an
> unexplained curation flag.
>
> **An endpoint is recorded only where a call reached it.** Five entries once
> carried a base URL recalled from general knowledge rather than read from
> documentation; all five were nulled and the recalled URL kept in that
> entry's `curation_notes` as a starting hint that had never been called. Three
> endpoints are recorded now, and each was dialled: the hint for `uniprotkb`
> stays beside the verified route so a reader can see which it was.
>
> A registry that admits uncertainty is useful. One with a plausible remembered
> endpoint is dangerous, because code will call it.

### What the registry does and does not assert

| Field | What a value means | What `null` / `unknown` means |
| --- | --- | --- |
| `endpoint` | A URL read from the provider's documentation and recorded with a citation. No entry in this registry has one today. | Nobody established the route — the current state of **every** entry. A URL somebody remembers is not an endpoint; it belongs in `curation_notes` as a hint. The model also refuses an endpoint without a citation, and refuses one on a source with no network access mode. |
| `version` | A release string somebody recorded. | No release was recorded. Never "latest", never today's date. |
| `approximate_record_count` | A number somebody counted. | Nobody counted. Never estimated from memory. |
| `license` | A statement, with `license_source` naming who made it. | The licence has not been read. `needs_legal_review` cannot be false while `license` is null. |
| `redistribution_allowed: unknown` | — | The licence has not been read. **It does not mean redistribution is permitted.** |
| capability `unknown` | — | Nobody checked. It is **not** `not_supported`; a planner can prefer a verified route and schedule the unknown one for curation. |
| `connectivity_verified` | `true` only with a recorded `ConnectivityCheck`. | A pass is about one capability at one URL at one time, from one environment. It is not a licence, not permission to redistribute, and says nothing about the other capabilities. |

**Three entries record an endpoint; 46 do not.** Five once did —
`europe_pmc`, `ncbi_protein`, `pubchem`, `pubmed`, `uniprotkb` — but those URLs
were recalled from general knowledge, not read from the providers'
documentation, and the citations backing them were bare URLs recalled the same
way and never opened from here. A registry consumed by code must not carry a
URL nobody has called, so all five were nulled and each recalled URL survives
in that entry's `curation_notes` as a starting hint, stated as never having
been called.

`uniprotkb` has since been dialled, along with `rcsb_pdb` and `rhea`.
`eagent sources verify` asked each for one known record, checked that the
response contained what only that record carries, fetched the provider's own
documentation page in the same sweep, and wrote the result to
`connectivity.observed.yaml` — machine-written, merged at load, and separate
from these curated files because a reachable URL goes stale on its own while a
curated judgement does not. The citation records when the documentation was
opened, so it is an observation rather than an assertion that documentation
exists.

Twenty-three entries still declare a network access mode and record **no**
endpoint; they are exactly what `SourceRegistry.without_endpoint()` returns, and
every one must be resolved by a curator before any connector dials anything.

**A verified base URL still licenses nothing.** The probe shows that one
request, spelled one way, returned the record it asked for. It says nothing
about the URL a connector would build: the generic `<base>/<key>` shape is not
Rhea's query-parameter API, and calling it would produce a 404 that the
resolver reports as a miss — a route silently not working, which is worse than
a refusal. So a connector declares the probed capability its client was
written against, and `uniprotkb` is the only one that does.

Exactly **one** licence string is recorded anywhere in the registry: `brenda`
carries `CC BY 4.0`, and its `license_source` says:

> Stated by the project owner as their current information in the brief for this
> registry. NOT independently verified against the resource's own terms, which
> is why `needs_legal_review` stays true and `redistribution_allowed` stays
> unknown.

Every other entry has `license: null`, `needs_legal_review: true`, and a
curation note demanding the terms be established before use.

---

## The registry at a glance

| Measure | Value |
| --- | --- |
| Registered sources | 49 (defined once each; `enzengdb` declares two layers) |
| Entries with `connectivity_verified: true` | **3** (uniprotkb, rcsb_pdb, rhea), each with the call recorded |
| Entries with `needs_curation: true` | 49 |
| Entries with a recorded endpoint | **3**, each reached by a recorded call |
| Connectors whose client was checked against a probe | **1** (uniprotkb) |
| Network-mode entries with no endpoint | 26 |
| Entries with a recorded licence | 1 (`brenda`, unverified) |
| Entries with `needs_legal_review: true` | 49 |
| Entries requiring human review **per record** | 9 |
| Entries reachable only by a human import route | 6 |
| Entries not programmatically reachable at all | 18 |
| Source groups after lineage collapse | 28 |
| Groups countable as independent (every member fully traced) | **13** |
| Entries whose upstream list is admittedly incomplete | 23 |
| Rollout stage 1 / 2 / 3 | 16 / 24 / 9 |

### Capability flag legend

The flag column in the tables below reads
`kw fetch seq chem bulk ver redist`, in that fixed order:

| Flag | Meaning |
| --- | --- |
| `kw` `keyword_query` | Free-text or field-scoped search returning candidate records |
| `fetch` `exact_record_fetch` | Retrieval of one record by its stable identifier |
| `seq` `sequence_query` | Search *by sequence* (not by accession) |
| `chem` `chemical_structure_query` | Search by SMILES, InChI or substructure |
| `bulk` `bulk_snapshot` | A downloadable whole-resource dump that can be pinned |
| `ver` `version_information` | A release identifier recordable in run provenance |
| `redist` `redistribution_allowed` | Whether derived records may be redistributed |

`+` supported (documented), `-` not supported, `?` **unknown — nobody checked**.
`?` is the default and the honest majority state: of the 343 flag slots across 49
entries, 220 are `?`, 72 are `-` and only 51 are `+`.

### Access modes

`AccessMode` records what a resource actually is, so a valuable resource with no
live API is registered honestly rather than dressed up as a service:

| Mode | Meaning | Agent can call it unattended? |
| --- | --- | --- |
| `rest_api` | Documented HTTP API | yes |
| `sparql` | SPARQL endpoint | yes |
| `soap` | SOAP/WSDL service, usually with credentials | yes |
| `local_package` | Installed library or local dataset, no network call | yes |
| `bulk_download` | Whole-resource archive, pinned as a snapshot | no (pinned first) |
| `offline_import` | Author CSV or archive imported by hand | no — human step |
| `manual_review_import` | Web interface only; a human curates each record | no — human step |
| `unknown` | Route not established; a curator must determine it | no |

Two entries — `oed` and `protabank` — are registered with `access_modes:
[unknown]` on purpose. Both are described in the literature as offering
programmatic access; whether that is a hosted API, an installable package or a
download **has not been established here and must not be guessed as a REST
endpoint**.

### Evidence strength ceiling

`evidence_strength_ceiling` is the strongest claim an *automated* ingest may
stamp on a record from that source; anything stronger needs a human reviewer and
the primary reference they read (`intake.promote()`). It measures how tightly a
claim binds to a specific sequence and says **nothing** about whether the
measured endpoint is relevant to the target reaction — that is what
`not_good_for` is for. A source can be `sequence_level_experimental` and still
be irrelevant evidence; `mavedb` and `fireprotdb` are exactly that case.

---

## Registered sources, by layer

#### reaction_and_chemistry

| id | stage | access modes | caps `kw fetch seq chem bulk ver redist` | strength ceiling | licence | endpoint | good for (headline) | **not** good for (headline) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `chebi` | 1 | rest_api, soap, bulk_download | `+ + - ? + + ?` | annotation_only | null / review | null | Stable chemical identifiers for reaction participants, used as a join key with Rhea and with literature annotation corpora. | A class entry (for example a generic "ketone" or "secondary alcohol" class) covers many compounds and must never be used as the assay substrate; resolving a task substrate to a class entry silently widens the task. |
| `enzyme_explorenz` | 1 | bulk_download, manual_review_import | `? ? - - + ? ?` | ec_species_mapped | null / review | null | Authoritative EC nomenclature -- accepted and systematic names, the official reaction, and the class hierarchy. | An EC number is a reaction-class label and is not sufficient evidence of specificity for a given substrate; two enzymes sharing an EC number may have no overlap in substrate scope. |
| `enzymemap` | 2 | bulk_download, offline_import | `? ? - ? + ? ?` | annotation_only | null / review | null | Curated and atom-mapped enzymatic reactions, which make the bond that changes explicit instead of leaving it to be inferred from two SMILES. | Reaction-level resource. An entry is NOT a claim that each associated sequence was experimentally validated on that substrate, and must not be ingested as sequence-level activity evidence. |
| `metanetx` | 2 | bulk_download | `? ? - ? + ? ?` | annotation_only | null / review | null | Reconciling compound and reaction identifiers across several chemical and metabolic resources through one namespace, so that two resources can be recognised as describing the same transformation. | A mapping edge is a reconciliation decision, not a statement of chemical identity. Mapped entities may differ from the upstream ones in protonation, stereochemistry or cofactor specificity, so one mapping edge does not license merging all chemical states of a compound. |
| `pubchem` | 1 | rest_api, bulk_download | `+ + ? + + ? ?` | annotation_only | null / review | null (hint in notes) | Normalising a trivial, catalogue or paper-prose compound name to a CID and then to SMILES/InChI, at a scale no curated resource matches. | Name resolution routinely returns the wrong stereoisomer, a salt, or a different charge state. Chirality, salt form and protonation must be re-verified after every lookup; for an asymmetric reduction task this is the difference between the right and the wrong experiment. |
| `retrorules` | 3 | bulk_download, offline_import | `? ? - ? + ? ?` | computational_construct | null / review | null | Reaction rules at several reaction-centre diameters, usable to enumerate candidate transformations for a substrate that has no exact precedent. | A rule firing is an enumeration, not a demonstrated enzymatic reaction, and must never be recorded as an observed transformation. |
| `rhea` | 1 | rest_api, sparql, bulk_download | `+ + - ? + + ?` | annotation_only | null / review | null | Reaction-level definition of a transformation with participants given as ChEBI entries, so the substrate and product are structures rather than names. | Not evidence that any specific sequence catalyses the reaction; a Rhea identifier attached to a protein entry is an annotation, not a measurement. |

#### enzymology_evidence

| id | stage | access modes | caps `kw fetch seq chem bulk ver redist` | strength ceiling | licence | endpoint | good for (headline) | **not** good for (headline) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `brenda` | 1 | soap, bulk_download, manual_review_import | `+ ? ? ? ? ? ?` | ec_species_mapped | CC BY 4.0 / review | null | The broadest curated coverage of enzyme activity, substrate ranges, kinetic parameters, inhibitors, pH and temperature behaviour, with literature references attached. | A record is frequently tied to an EC number and an organism rather than to a specific sequence, so it must not be ingested as sequence-level evidence; each record needs checking that it maps to one specific sequence before any sequence-level claim is made. |
| `enzengdb` | 2 | bulk_download, offline_import | `? ? ? ? ? ? ?` | homolog_experimental | null / review | null | Engineered parents and their variants with the reported performance of each, including yields, total turnover numbers, selectivity and screening outcomes, which is the record shape this project's own batches produce. | Coverage is limited to certain scaffolds and non-natural reactions; absence of an enzyme or reaction here means nothing about the literature. |
| `oed` | 2 | unknown | `? ? ? ? ? ? ?` | ec_species_mapped | null / review | null | Programmatic integration of enzymology records that otherwise have to be assembled by hand from several resources. | NOT an independent source. It re-integrates BRENDA and SABIO-RK, so a record found both here and in either upstream is one piece of evidence, not two, and counting hits across them overstates support. |
| `retrobiocat_db` | 2 | manual_review_import, offline_import | `? ? ? ? ? ? ?` | ec_species_mapped | null / review | null | Biocatalytic transformations organised by reaction type, including the enzyme classes used for ketone reduction, which maps directly onto the pilot task. | The open-source code ships only example specificity data. Having the code is not having the database, and a pipeline built against the example data must not be described as covering the published substrate scope. |
| `sabio_rk` | 1 | rest_api, manual_review_import | `+ ? - ? ? ? ?` | ec_species_mapped | null / review | null | Kinetic parameters recorded with their assay context -- buffer, pH, temperature and the measured entity -- which is what makes a kinetic number interpretable at all. | Coverage is narrower than a breadth-first enzymology resource; absence here is not evidence that an activity was never measured. |
| `strenda_db` | 2 | manual_review_import | `? ? ? ? ? ? ?` | sequence_level_experimental | null / review | null | Enzymology records captured against a reporting standard, so the protein, the assay conditions and the measured quantity are present together rather than scattered across a paper's methods section. | Do not assume sequence-database-scale coverage. The number of fully contextualised records is small compared with a breadth-first enzymology resource, and absence here is not evidence of anything. |

#### sequence_family_evolution

| id | stage | access modes | caps `kw fetch seq chem bulk ver redist` | strength ceiling | licence | endpoint | good for (headline) | **not** good for (headline) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `akr_superfamily` | 2 | manual_review_import, offline_import | `? ? ? - ? ? ?` | annotation_only | null / review | null | Authoritative nomenclature for aldo-keto reductases, the second major family relevant to asymmetric ketone reduction, where naming in the literature is otherwise inconsistent. | Nomenclature and phylogeny are not activity data; no substrate scope, kinetics or enantioselectivity is implied by membership. |
| `cath_funfam` | 2 | rest_api, bulk_download | `? ? ? - ? ? ?` | annotation_only | null / review | null | Structural classification of domains, which groups candidates by fold rather than by sequence similarity alone. | A shared fold is compatible with completely different chemistry; fold is not mechanism and certainly not substrate scope. |
| `eggnog` | 3 | bulk_download, manual_review_import | `? ? ? - ? ? ?` | annotation_only | null / review | null | Orthology groups and taxonomic context, which help decide whether a candidate is a plausible functional counterpart in another organism. | Orthology is an evolutionary relationship, not an activity label, and orthologues routinely differ in substrate preference. |
| `interpro` | 1 | rest_api, bulk_download | `+ + ? - + + ?` | annotation_only | null / review | null | Integrated family, domain and site annotation that constrains which candidates share an architecture with the seeds. | A single short motif or site match does not establish activity on a given substrate; motifs are shared across enzymes with different scopes. |
| `mgnify_proteins` | 3 | rest_api, bulk_download | `? ? ? - ? ? ?` | computational_construct | null / review | null | Metagenome-derived sequences that are absent from cultured-organism databases, which is where genuinely novel candidates come from. | Sequences are predicted from assemblies, so a record may be a fragment, a chimera or an assembly artefact rather than a real protein. |
| `ncbi_protein` | 2 | rest_api, bulk_download | `+ + ? - + + ?` | annotation_only | null / review | null (hint in notes) | Expanding a candidate set beyond curated entries, including sequences that never reach a reviewed database. | Annotation quality varies widely, and a protein name is frequently propagated from a distant homologue; a name is not a function. |
| `pfam` | 2 | rest_api, bulk_download | `+ + ? - + + ?` | annotation_only | null / review | null | Domain definitions and curated alignments, which give the position numbering a family-wide design discussion needs. | A domain assignment is not a function assignment, and a family can contain members with opposite stereopreference. |
| `sdred` | 2 | manual_review_import, offline_import | `? ? ? - ? ? ?` | annotation_only | null / review | null | Subfamily grouping within the short-chain dehydrogenase/reductase superfamily, which is directly the family of the pilot ketone-reduction task. | Family classification is not activity; membership says nothing about the target substrate or the product configuration. |
| `uniparc` | 2 | rest_api, bulk_download | `? + + - + + ?` | annotation_only | null / review | null | A non-redundant archive of sequences with their history, which answers whether two differently named records are literally the same sequence. | No annotation and no function; it answers identity questions only. |
| `uniprotkb` | 1 | rest_api, sparql, bulk_download | `+ + + - + + ?` | annotation_only | null / review | null (hint in notes) | The reference sequence record for most candidates, with cross-references that let a sequence be joined to structures, families and reactions. | Not every functional statement is an experimental result for that sequence. The evidence code must be kept and surfaced, because an inferred catalytic activity annotation reads identically to a measured one. |
| `uniref` | 1 | rest_api, bulk_download | `? + ? - + + ?` | computational_construct | null / review | null | Ready-made clustering at fixed identity thresholds, used to deduplicate a candidate set and to control redundancy in a selection. | Cluster co-membership is a computed grouping, not an identity and not a shared function; two members of one cluster can differ in substrate scope and in stereopreference. |

#### structure_and_mechanism

| id | stage | access modes | caps `kw fetch seq chem bulk ver redist` | strength ceiling | licence | endpoint | good for (headline) | **not** good for (headline) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `alphafill` | 2 | rest_api, bulk_download | `? ? - - ? ? ?` | computational_construct | null / review | null | Transferring cofactors, ligands and ions into predicted structures from homologous experimental structures, which turns an unusable apo model into a testable hypothesis about the holo active site. | Every placement is computational inference from a homologue and must be recorded with the homology-transplanted ligand source, never as an observed ligand; this is the single most likely place for a predicted arrangement to be reported as a fact. |
| `alphafold_db` | 1 | rest_api, bulk_download | `? ? - - ? ? ?` | computational_construct | null / review | null | Reusable predicted monomer structures for sequences with no experimental entry, which is most mined candidates. | Retrieving a stored model and running a complex prediction are different capabilities; this resource provides the first and not the second, and a plan that assumes it supplies holo complexes is wrong. |
| `mcsa` | 2 | rest_api, bulk_download, manual_review_import | `? ? - - ? ? ?` | ec_species_mapped | null / review | null | Literature-sourced catalytic residues with their assigned roles, which is what a catalytic template needs and what cannot be inferred from a fold. | Coverage is partial. Most enzymes have no entry, and absence is not evidence that catalytic residues are unknown. |
| `rcsb_pdb` | 1 | rest_api, bulk_download | `+ + ? ? + ? ?` | annotation_only | null / review | null | Experimentally determined structures, including enzyme complexes with cofactor and substrate analogues, which are the only observed evidence of an active-site arrangement. | Search results can include computed structure models alongside experimental entries; the two must be separated explicitly, because a predicted model returned by a structure search looks like a measurement. |
| `sifts` | 1 | bulk_download, rest_api | `- ? - - ? ? ?` | annotation_only | null / review | null | Residue-level mapping between a sequence database entry and a structure chain, which is the only correct way to carry a position between the sequence and structure layers. | It is a mapping, not evidence; it says nothing about function, binding or activity. |
| `wwpdb_ccd` | 1 | bulk_download | `? ? - ? + ? ?` | annotation_only | null / review | null | Canonical atom names, bonding and chemical identity for every ligand and cofactor that appears in a structure, which is what makes an atom-level geometric criterion reproducible. | It defines chemistry, not occupancy or geometry; a component entry says nothing about whether that ligand is present in any particular structure. |

#### mutation_and_performance

| id | stage | access modes | caps `kw fetch seq chem bulk ver redist` | strength ceiling | licence | endpoint | good for (headline) | **not** good for (headline) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `catpred_db` | 2 | bulk_download, offline_import | `- ? - - ? ? ?` | ec_species_mapped | null / review | null | Assembled kinetic parameter data in a form suitable for training and evaluating kinetic prediction, with consistent field definitions. | It is a dataset, not the prediction model that shares its name, and the two must not be conflated in provenance or in a results narrative. |
| `enzengdb` | 2 | bulk_download, offline_import | `? ? ? ? ? ? ?` | homolog_experimental | null / review | null | Engineered parents and their variants with the reported performance of each, including yields, total turnover numbers, selectivity and screening outcomes, which is the record shape this project's own batches produce. | Coverage is limited to certain scaffolds and non-natural reactions; absence of an enzyme or reaction here means nothing about the literature. |
| `esibank` | 2 | bulk_download, offline_import | `? ? ? ? ? ? ?` | ec_species_mapped | null / review | null | Assembled enzyme-substrate interaction data at a scale useful for training and for prioritising which pairs are worth testing. | It is the data resource associated with a specificity model, not the model itself; keeping that distinction is required in provenance, and a model prediction must never be stored as a database record. |
| `fireprotdb` | 2 | rest_api, bulk_download | `? ? - - ? ? ?` | sequence_level_experimental | null / review | null | Experimental stability measurements for single-point mutants, which is the cost side of any local design decision. | A stability gain is not a specificity gain and not an activity gain; stabilising substitutions frequently reduce turnover, and the two objectives trade off. |
| `intenzydb` | 2 | bulk_download, offline_import | `? ? ? - ? ? ?` | ec_species_mapped | null / review | null | Structure and kinetics associated in one place, usable to look for structural correlates of kinetic differences within a family. | Each record requires per-record alignment between the structure, the sequence and the measurement before it can be used; the association in the table is not itself a verified correspondence. |
| `mavedb` | 2 | rest_api, bulk_download | `? ? - - ? ? ?` | sequence_level_experimental | null / review | null | Multiplexed variant-effect datasets covering many positions in one protein, which is the densest available prior on which positions tolerate substitution. | The measured endpoint is frequently growth, abundance, display or binding rather than catalysis. The measurement type must be preserved on every record, and such a record is not evidence about ketone-reduction activity however strong the signal is. |
| `protabank` | 3 | unknown | `? ? ? - ? ? ?` | homolog_experimental | null / review | null | Supplementary protein engineering datasets spanning several measured properties, useful for widening a sparse prior. | Heterogeneous property definitions, so two datasets reporting activity may be measuring different things and must not be combined. |
| `proteingym` | 3 | bulk_download, local_package | `- ? - - ? ? ?` | annotation_only | null / review | null | A standardised benchmark for comparing variant-effect predictors, useful for deciding which predictor this project should trust at all. | It is a benchmark across many proteins, not a ketone-reduction activity set; strong benchmark performance says nothing about this task's family or endpoint. |
| `reactzyme` | 3 | bulk_download, local_package | `- ? ? ? ? ? ?` | annotation_only | null / review | null | Evaluating enzyme-reaction retrieval, which is the shape of this project's own search problem, so it can measure whether the retrieval step works at all. | It contains positives. An absent enzyme-reaction pair is not an experimental negative, and training or evaluating as though absence means inactivity will manufacture false confidence. |
| `skid` | 2 | bulk_download, offline_import | `? ? ? ? ? ? ?` | ec_species_mapped | null / review | null | Linking a sequence, a kinetic measurement and a structure in one record, which is the join this project otherwise has to make by hand. | It contains modelled complexes alongside experimental ones. A modelled complex must be recorded with its computational ligand source and must never be treated as an observed structure. |

#### literature_and_feedback

| id | stage | access modes | caps `kw fetch seq chem bulk ver redist` | strength ceiling | licence | endpoint | good for (headline) | **not** good for (headline) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bacdive` | 2 | rest_api, manual_review_import | `? ? - - ? ? ?` | annotation_only | null / review | null | Strain origin, isolation source and cultivation conditions for the organism a candidate sequence came from, which informs expression-host choice and plausible operating temperature. | A strain growth temperature is not an enzyme's optimum and not its thermostability; the two are routinely conflated and the inference does not hold in either direction. |
| `enzchemred` | 2 | bulk_download, offline_import | `- ? - - ? ? ?` | annotation_only | null / review | null | An expert-annotated corpus linking enzymes and chemical transformations in text to sequence-database and chemical-ontology identifiers, which is exactly the relation this project's literature module must extract. | It is an evaluation corpus, not an evidence store; a relation in it is not a measurement of activity and carries no conditions. |
| `enzymeml` | 1 | local_package, offline_import | `- - - - - ? ?` | sequence_level_experimental | null / review | null | A structured format for this project's own experimental output, holding the protein, the reaction, the conditions, the measured time courses and the analysis together in one document. | It is a data standard, not another database; it supplies no external records and must never be counted as a source of evidence about other people's enzymes. |
| `equilibrator` | 2 | local_package | `- ? - ? ? ? ?` | computational_construct | null / review | null | Estimated reaction thermodynamics under stated conditions, which tells the project whether a target direction is feasible at all and what cofactor ratio or coupled system it would need. | Accessed as a local package rather than as a simple remote endpoint, so a plan that assumes a REST call is wrong about how this works. |
| `europe_pmc` | 1 | rest_api, bulk_download | `+ + - - ? - ?` | annotation_only | null / review | null (hint in notes) | Programmatic search with access to full text where the licence permits it, which is the only way an extraction pipeline can reach a methods section legitimately. | Full text is available only within open licences; subscription content must not be fetched or stored, and a pipeline that silently falls back to the abstract will produce records that look full-text derived. |
| `kegg` | 3 | rest_api, manual_review_import | `? ? - - ? ? ?` | ec_species_mapped | null / review | null | Pathway context for a reaction, showing what else consumes or produces the substrate in a living system. | Not unconditionally open. Bulk access and some uses are subject to subscription or licence conditions that must be confirmed before any automated retrieval or any redistribution of derived records. |
| `machine_literature_extraction` | 3 | local_package, offline_import | `- - - - - ? ?` | annotation_only | null / review | null | Turning kinetic and condition tables in papers into structured candidate records at a volume no curator can match. | Always a pending-review source. No extracted record may enter the evidence store as confirmed, and none may be promoted to sequence-level experimental strength without a named human verifier. |
| `metacyc_biocyc` | 3 | rest_api, bulk_download, manual_review_import | `? ? - - ? ? ?` | ec_species_mapped | null / review | null | Literature-curated metabolic reactions and pathways with citations attached, which makes a pathway claim auditable. | Not unconditionally open. Access and redistribution are subject to subscription or licence conditions that must be confirmed first. |
| `pubmed` | 1 | rest_api | `+ + - - ? - ?` | annotation_only | null / review | null (hint in notes) | Programmatic search across the biomedical literature, with stable identifiers that can be recorded in an evidence reference. | Abstract-level only; the numbers this project needs are almost always in the methods, the tables or the supplementary material. |
| `zenodo` | 2 | rest_api, offline_import | `? ? - - - + ?` | annotation_only | null / review | null | Fetching an author's deposited dataset by DOI, which is often the only machine-readable form of a paper's supplementary tables. | An archive does not guarantee complete experimental conditions; deposits routinely omit buffer, temperature, detection method and construct. |

---

## The specific cautions

Each of these is recorded in `not_good_for` on the entries named, which is a
required, non-empty field precisely so that a planner cannot read a registry of
what things are *good for* and use BRENDA as a sequence database or a benchmark
set as an activity set.

### 1. An EC number is not substrate-specificity evidence

Two enzymes sharing an EC number may have no overlap in substrate scope. EC
assignments in sequence databases are frequently propagated by similarity, so an
EC match is not an experimental result. `enzyme_explorenz` exists to normalise
EC strings and to detect transferred and deleted EC numbers — **not** to supply
evidence. `brenda`, `sabio_rk`, `oed`, `kegg`, `metacyc_biocyc`, `mcsa`,
`intenzydb`, `skid`, `catpred_db`, `esibank` and `retrobiocat_db` are all capped
at `ec_species_mapped` for this reason: their records are typically attached to
an EC number and an organism rather than to a specific sequence, and promotion
above that ceiling requires a human to establish the sequence per record.

### 2. Reaction direction must be preserved

An alcohol dehydrogenase assayed on the alcohol, following NAD+ reduction, is a
superb record of the **oxidation** and says nothing dependable about the ketone
reduction at the target pH, with the target cofactor, at the target substrate
loading. Such records arrive from curated resources with the same EC number and
the same substrate name.

`rhea`'s reference direction is a curatorial convention: a ketone/alcohol pair is
one reaction entry read in two directions, and citing a Rhea id does not record
which direction was assayed. The `rhea_reaction_id` join key therefore declares
explicitly that it does **not** establish the measured direction. The
`ReactionDirection` field travels on every `ExperimentRecord`, and
`intake.direction_check()` refuses to treat `UNSPECIFIED` as forward.

### 3. A chemical class entry is not a specific stereodefined compound

A ChEBI class entry (a generic "ketone", a "secondary alcohol") covers many
compounds. Resolving a task substrate to a class entry silently widens the task,
and for an asymmetric reduction that is fatal. `chebi` is registered separately
from `pubchem` *because* of the class/instance distinction, and its curation
notes require the project to decide and document the rule that a class entry
must **fail** task normalisation rather than be accepted.

`pubchem` has the mirror-image problem: name resolution routinely returns the
wrong stereoisomer, a salt, or a different charge state. It is registered as a
name-resolution service **with a mandatory verification step**, not as a source
of chemical truth. The `inchikey` join key correspondingly refuses a
first-block-only comparison, which discards exactly the stereochemistry this
project optimises.

### 4. Resources that re-integrate others are not independent evidence

`oed` re-integrates BRENDA and SABIO-RK. `catpred_db` re-integrates BRENDA and
SABIO-RK. `enzymemap` re-curates BRENDA. `proteingym` overlaps MaveDB.
`interpro` integrates Pfam and other member databases. `uniref` is built on
UniProtKB and UniParc. `alphafill` transfers ligands from AlphaFold DB and the
PDB. `sifts` maps between the PDB and UniProtKB. `retrorules` generalises from
MetaNetX and Rhea. `reactzyme` derives from Rhea and UniProtKB. `enzchemred`
links PubMed, UniProtKB and ChEBI. `europe_pmc` includes PubMed's records.
`machine_literature_extraction` reads both literature indices.

`SourceRegistry.independent_source_groups()` collapses 49 sources into **28**
groups. The largest enzymology collapse is:

```
brenda, catpred_db, enzymemap, oed, sabio_rk     -> one group
mavedb, proteingym                               -> one group (shared upstream: mavedb)
interpro, pfam                                   -> one group (shared upstream: pfam)
```

Counting *groups*, not hits, is what stops a re-published measurement being read
as corroboration — but 28 groups is **not** 28 independent sources. Twenty-three
entries carry `derived_from_complete: false`: their upstream list is known to be
incomplete. Among them are `oed`, `catpred_db`, `enzymemap`, `skid`,
`intenzydb`, `esibank`, `fireprotdb`, `interpro`, `metanetx`, `retrorules`,
`reactzyme`, `proteingym` and `europe_pmc`. A source that cannot say what it
re-integrates cannot be shown to be separate from any other group, so
`SourceRegistry.independence_report()` counts only groups whose **every** member
declares a complete lineage: `n_independent` is **13** of the 28 today. The
other 15 groups are still returned and still listed, each printed with "NOT
counted, lineage incomplete" and the member ids responsible, so an untraced
source stays visible instead of being folded into a confident number.

Running that report over the pilot's enzymology and kinetics sources gives,
verbatim:

```
0 of 3 source group(s) countable as independent after lineage collapse
  - brenda, catpred_db, oed, sabio_rk  (shared upstream: brenda, sabio_rk; NOT counted, lineage incomplete: catpred_db, oed)
  - intenzydb  (NOT counted, lineage incomplete: intenzydb)
  - skid  (NOT counted, lineage incomplete: skid)
  ! catpred_db: lineage is known to be incomplete; it must not be counted as independent corroboration
  ! intenzydb: lineage is known to be incomplete; it must not be counted as independent corroboration
  ! oed: lineage is known to be incomplete; it must not be counted as independent corroboration
  ! skid: lineage is known to be incomplete; it must not be counted as independent corroboration
```

Six resources, and **nothing** that may be quoted as independent corroboration:
four of them admit they do not know their own upstreams, and the two that do —
`brenda` and `sabio_rk` — are in one group with the two resources that
re-integrate them. The `shared upstream` note names what caused that collapse,
which the group's own membership does not show: `brenda` and `sabio_rk` share
nothing with each other, and the group exists only because `oed` and
`catpred_db` re-integrate both.

The row-level counterpart is `datalayer/lineage.py`, which groups the retrieved
*records* by experiment activity, publication and assay fingerprint — see
`docs/DATA_LAYER.md` §6 for the four-databases-one-measurement worked example.

### 5. A positives-only retrieval benchmark yields no experimental negatives

`reactzyme` contains positives. An absent enzyme–reaction pair is **not** an
experimental negative, and training or evaluating as though absence means
inactivity manufactures false confidence. The same applies to `esibank`
(absence of a pair is not a tested negative), `retrobiocat_db` (absence of a
substrate in a scope table is not a tested negative) and `enzengdb` (absence of
an enzyme means nothing about the literature). A real experimental negative is
`no_target_product_detected` **with a recorded limit of detection**, and no
public retrieval set supplies one.

### 6. A variant-effect endpoint may be growth or binding rather than catalysis

`mavedb`'s measured endpoint is frequently growth, abundance, display or binding.
Its ceiling is `sequence_level_experimental` because the score genuinely binds
to a specific variant sequence — and its curation notes say in so many words
that this is **not a licence to treat the score as activity evidence**: the
ingest must refuse any record whose measurement type is not catalytic when the
claim is about catalysis. `ExperimentRecord.measurement_type` exists for exactly
this, and the mutation-layer file's own notes call endpoint substitution "the
recurring failure in this layer". Scores are also normalised within an
experiment and are not comparable between experiments or proteins.

### 7. A stability dataset does not speak to specificity

`fireprotdb` holds experimental stability measurements for single-point mutants.
A stability gain is not a specificity gain and not an activity gain; stabilising
substitutions frequently reduce turnover, and the two objectives trade off.
Stability values are condition-dependent and method-dependent, so they must not
be pooled onto a single scale. It is registered as a **cost model** for proposed
mutations, never as support for a selectivity claim. This is also why the house
database keeps `stability_expression_risk` as a separate performance axis with
no combined score (`docs/HOUSE_DATABASE.md` §5).

### 8. A strain's growth temperature is not an enzyme's optimum

`bacdive` gives strain origin, isolation source and cultivation conditions. A
strain growth temperature is not an enzyme's temperature optimum and not its
thermostability; **the two are routinely conflated and the inference does not
hold in either direction**. Organism-level metadata says nothing about a
specific protein's expression, solubility or activity. It is registered as
context only, so that organism metadata has somewhere to live that is not the
enzyme record.

### 9. Favourable thermodynamics does not show that a sequence catalyses a reaction

`equilibrator` estimates reaction thermodynamics under stated conditions, which
tells the project whether a target direction is feasible at all. A favourable
equilibrium is a **necessary condition, never evidence**. Its estimates carry
uncertainty that must be propagated, group-contribution estimates can be poor
for unusual chemistry, and standard conditions are not assay conditions. Note
also that it is registered as `local_package`: a plan that assumes a simple REST
call is wrong about how this works. Its ceiling is `computational_construct`.

### 10. Code being open source does not mean the full dataset ships with it

`retrobiocat_db` is the sharpest case, and the one most directly on the pilot
task. The open-source code ships only **example** specificity data. Having the
code is not having the database, and a pipeline built against the example data
must not be described as covering the published substrate scope. The caveat
lives in `not_good_for` rather than in a source comment for exactly that reason.

Its access modes are `manual_review_import, offline_import` — both human steps,
which makes it human-import only. Registering the published package as
`local_package` would be a second version of the same error one level down:
`local_package` is *programmatic* in this taxonomy, so it would mark the source
callable inside a run and suppress the `plan.readiness()` warning that a person
has to obtain and check the real records first. Installing the code gives you
the examples, not the data.

Related confusions the registry separates by hand:

* `catpred_db` is a **dataset**, not the prediction model that shares its name;
* `esibank` is the **data resource** associated with a specificity model, not
  the model;
* `proteingym` is a **benchmark**, and using it to select candidates rather than
  to evaluate models is a category error;
* `alphafold_db` is a **model store**, explicitly not a prediction service:
  retrieving a stored model and running a complex prediction are different
  capabilities, and its models are apo — no cofactor, no substrate, no ions;
* `enzymeml` is a **data standard**, not another database. It supplies no
  external records and must never be counted as a source of evidence about other
  people's enzymes.

### 11. Some pathway resources carry subscription or licence conditions

`kegg` and `metacyc_biocyc` are **not unconditionally open**. Bulk access and
some uses are subject to subscription or licence conditions that must be
confirmed **before any automated retrieval** and before any redistribution of
derived records. Both are staged at 3 and gated on the licence question rather
than on engineering work; `kegg`'s curation notes additionally require the
project to determine whether its use is academic or commercial, because the
answer changes what is permitted.

More broadly: `redistribution_allowed` is `unknown` on all 49 entries. That means
the licence has not been read, not that redistribution is permitted.
`europe_pmc`'s notes require the per-article licence refusal path to be
implemented **before** any extraction runs; `zenodo`'s require deposit licences
to be read per record.

### 12. Two further cautions the registry records that are easy to miss

* **A predicted model returned by a structure search looks like a measurement.**
  `rcsb_pdb` search results can include computed structure models alongside
  experimental entries, and the two must be separated explicitly. `skid`
  likewise contains modelled complexes alongside experimental ones. Every
  computational placement must carry its `LigandSource`
  (`homology_transplanted`, `docking_predicted`, `joint_structure_prediction`)
  and must never collapse into "binding site confirmed". `alphafill` is the
  single most likely place in the whole registry for a predicted arrangement to
  be reported as a fact.
* **Machine extraction is always a pending-review source.** No record from
  `machine_literature_extraction` may enter the evidence store as confirmed, and
  none may reach sequence-level strength without a named human verifier. Its
  curation note requires the review queue to exist **before** the first
  extraction run, not after, because a pending-review store that does not exist
  becomes an accepted store.

---

## Rollout stages

Staging exists because wiring forty-nine resources at once produces a system
nobody can debug. `SourceRegistry.stage_plan(n)` returns the sources assigned to
each stage, and `datalayer/plan.py` holds the typed work packages, their
deliverables and their preconditions.

| Stage | Intent | Count | Sources |
| --- | --- | --- | --- |
| **1** | Close the natural-enzyme discovery loop: get from a substrate name to a ranked, mechanistically checkable candidate set with a real evidence trail, using the smallest set of resources that makes the pilot task answerable. | 16 | `chebi`, `enzyme_explorenz`, `pubchem`, `rhea`, `brenda`, `sabio_rk`, `interpro`, `uniprotkb`, `uniref`, `alphafold_db`, `rcsb_pdb`, `sifts`, `wwpdb_ccd`, `enzymeml`, `europe_pmc`, `pubmed` |
| **2** | Support mutation optimisation and model evaluation, on top of a parent family stage 1 has already identified. | 24 | `enzymemap`, `metanetx`, `enzengdb`, `oed`, `retrobiocat_db`, `strenda_db`, `akr_superfamily`, `cath_funfam`, `ncbi_protein`, `pfam`, `sdred`, `uniparc`, `alphafill`, `mcsa`, `catpred_db`, `esibank`, `fireprotdb`, `intenzydb`, `mavedb`, `skid`, `bacdive`, `enzchemred`, `equilibrator`, `zenodo` |
| **3** | Novelty and generality, under an explicit budget, once stages 1 and 2 have produced something to compare against. | 9 | `retrorules`, `eggnog`, `mgnify_proteins`, `protabank`, `proteingym`, `reactzyme`, `kegg`, `machine_literature_extraction`, `metacyc_biocyc` |

Two stage decisions are worth stating explicitly.

`sifts` and `wwpdb_ccd` are **stage 1 despite producing no evidence of their
own**. They are base dependencies: without the residue-level mapping and the
chemical component dictionary, every geometric criterion in the project is
measured between atoms nobody can name reproducibly, and gates 2 and 3 of the
precondition chain cannot run at all.

`mgnify_proteins` is **stage 3 on purpose**. `plan.MGNIFY_EARLY_ENTRY_RULE`
states the reason:

> metagenomic candidates may enter an earlier round in small numbers, as a
> deliberately bounded exploration slot, but a whole experimental round must not
> be spent on distant sequences with no functional evidence: a round that returns
> nothing teaches nothing, because a negative on an unexpressed or untestable
> protein does not even tell you the sequence was wrong.

`plan.check_novelty_budget()` enforces this against a proposed batch. Its
defaults — at most 10 % of a round, at most 8 slots, and only in rounds of 24 or
more — are **project policy, not measured quantities**. Nobody has determined an
optimal exploration fraction for this chemistry, and the module says so rather
than pretending otherwise.

### Readiness is asserted, never inferred

`plan.readiness(stage, registry, available_sources=...)` reports what can
actually run. `available_sources` is what an **operator asserts is reachable in
this environment, right now**. It is not defaulted to "everything registered"
and it is not derived from the registry's access modes, because an access mode
is documentation and deriving availability from it would manufacture exactly
the confidence the registry refuses to state. It is not derived from
`connectivity_verified` either: a route that answered once, from one
environment, at one moment, is not a route that is up now — and reachability
is still not a licence, a rate limit or a bulk route. Passing `None` therefore
reports every source-dependent package as blocked. That is the honest starting
state of a fresh checkout.

---

## 中文摘要

### 最重要的一句话

**49 条里只有 3 条做过连通性测试。** UniProtKB、RCSB PDB、Rhea 由
`eagent sources verify` 在本容器里真实调用过，并把调用记录（URL、时间、状态码、
响应摘要，以及响应体中必须出现的字符串）写进了注册表；其余每一条仍带
`connectivity_verified: false`。

`true` 只有在带着这样一条记录时才被接受。一个光秃秃的布尔值说"这条路能走"毫无价值——
这正是模式层原先直接拒绝 `true` 的原因；一次通过只说明**这个能力、这个 URL、在这个时刻、
从这个环境**答对了，仅此而已。能力标志描述的仍是"文档上声称提供什么"，不是"已验证可用"。
所有不确定的值一律写 `null` 并置 `needs_curation: true`，同时在 `curation_notes` 里写清
策展人必须确认什么——模型拒绝接受一个没有说明的 curation 标志。

**而且"基址已验证"并不等于"请求已验证"。** 探针证明的是某一条写法的请求拿回了它要的那条
记录；连接器自己拼出来的 `<base>/<key>` 并不是 Rhea 的查询式 API，调用它会得到一个 404，
而解析层会把它报告成一次 miss——一条悄无声息失效的路线，比直接拒绝更糟。所以连接器必须
声明自己的客户端是照着哪一条探过的请求写的，目前只有 UniProtKB 声明了。

一个承认自己不确定的注册表是有用的；一个编造了看起来合理的 endpoint 的注册表是危险的，
因为代码真的会去调用它。

### 现状数字

共注册 49 个来源（`enzengdb` 跨两层，定义一次）。其中：
记录了 endpoint 的有 **0 个**。曾经有 5 个（`europe_pmc`、`ncbi_protein`、
`pubchem`、`pubmed`、`uniprotkb`），但那些 URL 来自一般知识回忆，并非读自官方文档，
其引用也是同样未曾打开过的裸 URL；代码会真的去调用 endpoint，因此这 5 个 endpoint
与这 5 条引用都已清空，回忆到的 URL 改写进各自的 `curation_notes`，明确标注为"仅供
策展人起步参考、从未在本环境调用过"。声明了网络访问方式却**没有** endpoint 的有
26 个；记录了许可证的只有 1 个（`brenda` 的 `CC BY 4.0`，来源是项目方的陈述，
**未经独立核实**）；需要法务审查的 49 个；需要逐条人工复核的 9 个；只能靠人工导入的
6 个；完全无法程序化访问的 18 个。343 个能力标志槽位中，220 个是"未知"，72 个是
"不支持"，只有 51 个是"支持（有文档）"。

`unknown` ≠ `not_supported`。前者是一项待办的核查任务，后者是一条可用于路由的事实。
`redistribution_allowed: unknown` 的意思是**许可证还没被读过**，绝不表示允许再分发。

### 必须记住的具体告诫

1. **EC 号不是底物特异性证据。** 同一 EC 号下两个酶的底物谱可以毫无交集；序列库里的
   EC 标注常由相似性传播而来。因此 `brenda`、`sabio_rk`、`oed`、`kegg` 等的证据上限
   被设为 `ec_species_mapped`。
2. **反应方向必须保留。** 用醇作底物、跟踪 NAD+ 还原测出的数据，是一条优秀的**氧化**
   记录，对目标条件下的酮还原没有可靠说明力。这类记录带着相同的 EC 号和相同的底物名
   从策展库里出来，不检查就会被当成阳性种子。
3. **化学类目条目不是具体的立体定义化合物。** 把任务底物解析到 ChEBI 的一个类目条目
   （如泛指的"酮"）会悄悄把任务范围放大；对不对称还原而言这是致命的。
4. **再整合型资源不构成独立证据。** `oed`、`catpred_db` 再整合 BRENDA/SABIO-RK，
   `enzymemap` 再策展 BRENDA，`proteingym` 与 MaveDB 重叠。49 个来源经血缘合并后剩
   **28 组**——但 28 组并不等于 28 个独立来源：有 23 个来源的上游清单被明确标记为不
   完整，一个说不清自己再整合了谁的来源，也就无法证明它与别的组相互独立。因此
   `independence_report()` 只统计"全体成员血缘均已交代清楚"的组，当前为 **13 组**；
   其余 15 组照常列出，并逐条标注"NOT counted, lineage incomplete"，绝不混入那个
   看起来可信的数字里。
5. **只含阳性的检索基准给不出实验阴性。** `reactzyme` 里缺失的酶—反应配对不是实验
   阴性；按"缺失即无活性"去训练或评估，只会制造虚假信心。
6. **变异效应数据的测量终点可能是生长或结合，而不是催化。** `mavedb` 的上限虽是
   `sequence_level_experimental`（因为分数确实绑定到具体变体序列），但这**不等于**
   可以把分数当作活性证据；`measurement_type` 必须逐条保留。
7. **稳定性数据不说明特异性。** 稳定化突变常常降低周转数；`fireprotdb` 被注册为
   提议突变的**代价模型**，而不是选择性主张的支撑。
8. **菌株生长温度不是酶的最适温度**，也不是其热稳定性；这个推断在两个方向上都不成立。
9. **热力学有利不等于某条序列能催化该反应。** `equilibrator` 给出的是必要条件而非证据，
   而且它是本地包而不是 REST 服务。
10. **代码开源不等于完整数据随代码发布。** `retrobiocat_db` 的开源代码只附带**示例**
    特异性数据；拿到代码不等于拿到数据库。它的访问方式因此登记为
    `manual_review_import, offline_import`（两者都需要人工）：若登记成
    `local_package`，在本注册表的分类里那属于"可程序化访问"，会把"必须先由人取得并
    核对真实记录"的告警一并抹掉。同理：`catpred_db` 是数据集不是同名模型，
    `esibank` 是数据资源不是模型，`proteingym` 是基准不是候选来源，`alphafold_db` 是
    模型仓库不是预测服务（且模型是 apo，没有辅因子），`enzymeml` 是数据标准不是数据库。
11. **部分通路资源带订阅或许可条件。** `kegg` 与 `metacyc_biocyc` 并非无条件开放，
    在任何自动检索和任何衍生记录再分发之前必须先确认条款。

### 分阶段上线

阶段 1（16 个来源）闭合天然酶发现回路；阶段 2（24 个）支撑突变优化与模型评估；
阶段 3（9 个）才是新颖性与普适性。`sifts` 与 `wwpdb_ccd` 本身不产生任何证据，却被放在
阶段 1，因为没有残基级映射和化学组分字典，项目里所有几何判据都是在无法可重复命名的原子
之间测量的。`mgnify_proteins` 被刻意放在阶段 3：宏基因组候选可以作为**有限额度的探索
位**提前少量进入，但不能把一整轮实验花在没有任何功能证据的远缘序列上——全阴性的一轮
什么也教不会，因为"没表达""不可溶""没活性"这三种结果，便宜的检测根本区分不开。

`plan.readiness()` 不会从注册表推断可达性：只有操作者明确声明某来源在当前环境可达，
它才算可用。不声明就全部报告为阻塞——这是一个全新检出副本应有的诚实初始状态。
