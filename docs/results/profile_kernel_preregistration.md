# Pre-registration: giving the head the profile similarity

**Written and committed before the benchmark was run with this kernel.** The
point of committing it separately is that the commit order is checkable: the
decision rule below cannot have been chosen after seeing the numbers it
decides on.

## What is being decided

`EnzymeSubstrateModel` scores a sequence by kernel ridge regression over a
k-mer cosine. `docs/EVALUATION.md` §9.1 reports two things about that
similarity:

1. a label-free diagnostic (`similarity_contrast`) measuring it at **1.07x its
   own composition background**, so about 93 % of what it scores is
   amino-acid composition; and
2. a benchmark in which a profile-search nearest neighbour (macro AUROC 0.787)
   beats the k-mer nearest neighbour (0.710), which beats the model (0.667).

The question this run answers is the one that leaves: **does the head add
anything over homology transfer when both read the same, better similarity?**

## The criterion the representation was chosen on, which is label-free

`profile_contrast` on the same 309 sequences and the same seeded pairs:

| | k-mer cosine | profile search |
| --- | --- | --- |
| median real pair | 0.3173 | 0.0451 |
| median composition-matched pair | 0.2971 | **-0.0013** |
| ratio | 1.068 | undefined: the background is not positive |
| composition-matched pairs scoring nonzero | 399 of 399 | 249 of 399 |

The profile search has **no composition floor at all**: its
composition-matched pairs sit at or below its own null model. By the rule
already applied to `ks = (3, 4)` in §9.1 -- adopt the representation the
label-free criterion picks unless the benchmark shows it worse -- this is the
representation to try. No label was read to decide that.

## The decision rule, fixed now

1. The profile kernel is added as **its own comparator** (`profile_krr`,
   `profile_krr_calibrated`), not as a replacement. Both heads are scored on
   identical folds, rows and budget, so the comparison is paired.
2. **The default stays the k-mer kernel whatever the result**, because a
   default that requires an optional extra is not a default. The profile
   kernel is reachable only with the `search` extra and only when the caller
   asks for it.
3. **Positive definiteness is checked at the protocol's fixed ridge (0.1) and
   the ridge is not tuned to rescue a failure.** A Cholesky refusal is
   reported as a refusal, with `min_eigenvalue`'s estimate of how far from
   positive definite the matrix was and over the existing ridge sweep, as a
   sweep. Nothing clips an eigenvalue; no kernel is silently repaired.
4. **One run, reported whichever way it goes.** If the profile-kernel head
   loses to the profile-search baseline, or to the k-mer head, or to both,
   that is the recorded result and §9.1 says so. No hyperparameter is tuned
   afterwards, and the losing run is not re-run with a different seed.
5. The claim this can support, at most: "the head does / does not improve on
   homology transfer when both use an alignment, on annotation-derived labels,
   with test sequences held out by cluster at 40 % identity". It remains no
   evidence about ketoreductase activity.

## What would make this run invalid

Changing the folds, the seed, the ridge, the clustering threshold, the
calibration gate or the scoring between writing this and reading the result.
None of them is touched by the commit that adds the kernel; `git diff` against
this commit is the check.
