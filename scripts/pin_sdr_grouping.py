#!/usr/bin/env python3
"""Compute the SDR benchmark's fold grouping completely, and pin the result.

WHY THIS IS A SCRIPT AND NOT PART OF THE BENCHMARK RUN
======================================================
The grouping the benchmark's folds are built from has to satisfy one property:
every pair of sequences in two different clusters is below the identity
threshold. Establishing that for 309 sequences means 47,586 global alignments,
about 35 minutes of pure Python, because there is no cheap *sound* screen -- the
k-mer screen is a heuristic (on this pool, a screen at 0.10 missed half the
over-threshold pairs that a screen at 0.05 found), and the only sound bound
available, the residue-composition overlap, prunes 0.1 % of pairs because amino
acid composition is near-universal.

So the complete computation is run once, here, and its result is checked in with
a digest over its input. ``run_sdr_suite`` loads the pinned grouping and
verifies that digest against the dataset it just read, which costs milliseconds;
a mismatch is a refusal rather than a silent recomputation at a screen. That is
the same arrangement the hash-pinned reference set uses, for the same reason:
the expensive, exactly reproducible thing is computed once and verified every
time.

Usage::

    python scripts/pin_sdr_grouping.py --data tests/fixtures/sdr

Re-run it whenever the dataset or the threshold changes; the benchmark will
refuse to run against a stale pin, so forgetting is loud rather than quiet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eagent.eval.retrospective import (             # noqa: E402
    PINNED_GROUPING, BenchmarkProtocol, grouping_input_digest, load_sdr,
    sdr_examples,
)
from eagent.tools.mine_sequences import single_linkage_clusters   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path,
                    help="Directory holding the four checked SDR files.")
    ap.add_argument("--out", type=Path, default=PINNED_GROUPING,
                    help=f"Where to write it.  [default: {PINNED_GROUPING}]")
    ap.add_argument("--threshold", type=float,
                    default=BenchmarkProtocol().identity_threshold)
    args = ap.parse_args()

    examples = sdr_examples(load_sdr(args.data))
    items = sorted({(e.example_id, e.sequence) for e in examples})
    print(f"{len(items)} sequences; {len(items) * (len(items) - 1) // 2} pairs "
          f"to align at threshold {args.threshold}. No screen: this is the "
          f"complete computation and it takes about half an hour.",
          flush=True)
    started = time.time()
    report = single_linkage_clusters(items, args.threshold, kmer_prefilter=None)
    elapsed = time.time() - started
    assert report.n_prefiltered_pairs == 0, "a screen ran; the pin would not " \
                                            "be complete"
    if report.budget_exhausted:
        print("REFUSING TO PIN: the alignment budget ran out, so two "
              "sequences over the threshold may sit in different clusters.",
              file=sys.stderr)
        return 1

    payload = {
        "schema": "eagent.sdr_grouping.v1",
        "dataset": "SDR substrate-classification deposit (Zenodo 7141435)",
        "input_digest": grouping_input_digest(items, args.threshold),
        "n_sequences": len(items),
        "identity_threshold": args.threshold,
        "method": report.method,
        "identity_definition": report.identity_definition,
        "screen": None,
        "complete": True,
        "n_alignments": report.n_alignments,
        "n_prefiltered_pairs": report.n_prefiltered_pairs,
        "n_clusters": len(report.clusters),
        "seconds": round(elapsed, 1),
        "assignment": dict(sorted(report.assignment.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    sizes = sorted((len(c.members) for c in report.clusters), reverse=True)
    print(f"{len(report.clusters)} clusters from {report.n_alignments} "
          f"alignments in {elapsed / 60:.1f} min; largest {sizes[:5]}, "
          f"{sizes.count(1)} singleton(s)")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
