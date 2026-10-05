"""Retrospective evaluation: splits, endpoints, comparators, ablations.

This package answers one question -- *did the agent actually help?* -- and it
is written on the assumption that the usual way of answering it is wrong.

The four failures it is built against
-------------------------------------
``leakage``
    A retrospective enzyme benchmark almost always reports a number that is
    too good, because the "unseen" test rows are re-curations of training rows,
    variants of the same parent, or near-identical sequences from the same
    cluster. :mod:`eagent.eval.splits` groups before it splits and then
    *audits* the result, so a split can be shown clean instead of assumed
    clean.

``a moving endpoint``
    A campaign that misses its pre-registered bar can be rescued by lowering
    it, and nothing in a results table shows that this happened.
    :mod:`eagent.eval.metrics` refuses to compute the primary endpoint against
    any criterion but the registered one.

``an unfair comparator``
    A baseline evaluated on a different candidate pool, a different budget or
    a different hit definition is not a baseline, it is a strawman.
    :mod:`eagent.eval.baselines` gives every comparator one signature and
    enforces the shared pool in code.

``a module credited for nothing``
    Every component feels necessary until it is removed.
    :mod:`eagent.eval.ablations` removes one at a time over a fixed pool and
    reports the change in the primary endpoint, with the interval that says
    how little a 96-well round can resolve.

Nothing is re-exported here on purpose, following
:mod:`eagent.science`: importing ``eagent.eval`` must not drag the ingest
interface and the whole tool layer in as a side effect. Import the submodule
you need::

    from eagent.eval.splits import grouped_split, audit_leakage
    from eagent.eval.metrics import PreRegistration, precision_at_k
    from eagent.eval.baselines import CandidatePool, compare_baselines
    from eagent.eval.ablations import run_ablations
"""

from __future__ import annotations

__all__: list[str] = []
