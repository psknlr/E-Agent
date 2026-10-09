Enzyme Substrate Classification Dataset for SDRs and SAM-MTases
===============================================================

Adrian Jinich, Weill Cornell Medicine, ORCID: https://orcid.org/0000-0001-8616-9250
Dmitrij Rappoport, University of California, Irvine, ORCID: https://orcid.org/0000-0002-5024-7998

3 October 2022

Overview
--------
This dataset contains sequence information, three-dimensional structures (from AlphaFold2 model), and substrate classification labels for 358 short-chain dehydrogenase/reductases (SDRs) and 953 S-adenosylmethionine dependent methyltransferases (SAM-MTases).

The aminoacid sequences of these enzymes were obtained from the UniProt Knowledgebase (https://www.uniprot.org). The sets of proteins were obtained by querying using InterPro protein family/domain identifiers corresponding to each family: IPR002347 (SDRs) and IPR029063 (SAM-MTases). The query results were filtered by UniProt annotation score, keeping only those with score above 4-out-of-5, and deduplicated by exact sequence matches.

The structures were submitted to the publicly available AlphaFold2 protein structure predictor (J. Jumper et al., Nature, 2021, 596, 583) using the ColabFold notebook (https://colab.research.google.com/github/sokrypton/ColabFold/blob/v1.1-premultimer/batch/AlphaFold2_batch.ipynb, M. Mirdita, S. Ovchinnikov, M. Steinegger, Nature Meth., 2022, 19, 679, https://github.com/sokrypton/ColabFold). The model settings used were  msa_model = MMSeq2(Uniref+Environmental), num_models = 1, use_amber = False, use_templates = True, do_not_overwrite_results = True. The resulting PDB structures are included as ZIP archives

The classification labels were obtained from the substrate and product annotations of the enzyme UniProtKB records. Two approaches were used: substrate clustering based on molecular fingerprints and manual substrate type classification. For the substate clustering, Morgan fingerprints were generated for all enzymatic substrates and products with known structures (excluding cofactors) with radius = 3 using RDKit (https://rdkit.org). The fingerprints were projected onto two-dimensional space using the UMAP algorithm (L. McInnes, J. Healy, 2018, arXiv 1802.03426) and Jaccard metric and clustered using k-means. This procedure generated 9 clusters for SDR substrates and 13 clusters for SAM-MTases. The SMILES representations of the substrates are listed in the SDR_substrates_to_cluster_map_2DIMUMAP.csv and SAM_substrates_to_13clusters_map_2DIMUMAP.csv files.


The following manually defined classification tasks are included for SDRs: NADP/NAD cofactor classification; phenol substrate, sterol substrate, coenzyme A (CoA) substrate. For SAM-MTases, the manually defined classification tasks are: biopolymer (protein/RNA/DNA) vs. small molecule substrate, phenol subsrates, sterol substrates, nitrogen heterocycle substrates. The SMARTS strings used to define the substrate classes are listed in substructure_search_SMARTS.docx.


References
-----------
This dataset was prepared for the following papers:

D. Rappoport, A. Jinich, Protein Function Prediction from Three-Dimensional Feature Representations Using Space-Filling Curves, 2022, biorXiv 2022.06.14.496158, DOI: 10.1101/2022.06.14.496158.

A. Jinich,  S. Z. Nazia, A. V. Tellez,  D. Rappoport, M. AlQuraishi, K. Y. Rhee, Predicting enzyme substrate chemical structure with protein language models, 2022, 2022.09.28.509940, DOI: 10.1101/2022.09.28.509940.

If this dataset is useful to you, please consider citing these works.


File listing
------------
README.txt -- this file
substructure_search_SMARTS.docx -- SMARTS strings for manually defined substrate classes
SDR_sequences.fasta
SDR_AlphaFold2_PDBs.zip
SDR_cofactor_classifications.csv -- NAD/NADP cofactor classification (1 = NADP, 0 = NAD)
SDR_substructure_classifications.csv -- enzyme labels for phenol, sterol, and CoA substrate classes
SDR_cluster_classifications_2DIMUMAP.csv -- enzyme labels for substrate clusters
SDR_substrates_to_cluster_map_2DIMUMAP.csv -- substrate lists for substrate clusters
SAM_sequences.fasta
SAM_AlphaFold2_PDBs.zip
SAM_protRNADNA_vs_compound.csv -- biopolymer vs small-molecule classification (1 = biopolymer, 0 = small molecule)
SAM_cluster_classifications_2DIMUMAP.csv -- enzyme labels for substrate clusters
SAM_substrates_to_13clusters_map_2DIMUMAP.csv -- substrate lists for substrate clusters

