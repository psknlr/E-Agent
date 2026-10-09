# SDR substrate-classification dataset (test fixture copy)

These five files are verbatim copies of files from the Zenodo deposit
**Enzyme Substrate Classification Dataset for SDRs and SAM-MTases**
(Adrian Jinich and Dmitrij Rappoport, 3 October 2022),
version DOI `10.5281/zenodo.7141435`, licensed **CC BY 4.0**.

They were downloaded through `ZenodoConnector.download_file`, which compares
each file with the md5 the Zenodo record states before writing it. The loader
(`eagent.eval.retrospective.load_sdr`) re-checks the same md5 on every load, so
an edited copy fails loudly instead of changing a benchmark.

The labels were derived by the dataset's authors from the substrate and
product annotations of UniProtKB records; they are annotation-level evidence,
not measurements of activity. The authors ask that users of the dataset cite:

* D. Rappoport, A. Jinich, *Protein Function Prediction from Three-Dimensional
  Feature Representations Using Space-Filling Curves*, bioRxiv 2022,
  doi:10.1101/2022.06.14.496158
* A. Jinich, S. Z. Nazia, A. V. Tellez, D. Rappoport, M. AlQuraishi, K. Y. Rhee,
  *Predicting enzyme substrate chemical structure with protein language
  models*, bioRxiv 2022, doi:10.1101/2022.09.28.509940

Only the four files the benchmark reads, and the deposit's own README, are
copied. The AlphaFold2 structure archives and the SAM-MTase files are not.
