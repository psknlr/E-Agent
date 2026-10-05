"""Regression tests: the bar that was registered is the bar that is applied.

From an external audit, reproduced before being fixed. A template carrying the
same id but a looser bar could be handed to the result reader, which rebuilt
its criterion from whatever template it was given. A round registered at 50%
conversion was scored at 5%, a 10% result came back a hit, and nothing was
recorded as a protocol deviation.

The point of a pre-registered endpoint is that "a criterion exists in a file"
and "the result was judged by that criterion" are the same statement. Without
a check they are two statements, and only the first was ever true.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from eagent.context import ExecutionPolicy, RunContext
from eagent.provenance import RunManifest, sha256_obj
from eagent.schemas import TaskSpec
from eagent.schemas.templates import (
    AssayTemplate, TemplateProvenance, TemplateSourceType,
)
from eagent.tools.ingest_results import IngestResults, PositiveCriterion

PROV = TemplateProvenance(source_type=TemplateSourceType.CURATED_DATABASE,
                          identifiers=["internal:test-assay"])


def template(bar: float) -> AssayTemplate:
    """Two templates sharing an id and differing only in the bar."""
    return AssayTemplate(
        template_id="assay.tier2.product_confirmation.v1", tier=2,
        method="GC-MS against an authentic standard",
        confirms_product_identity=True, chiral_capable=True,
        positive_criteria={"min_conversion_pct": bar},
        limit_of_detection=0.1, limit_unit="%", provenance=PROV)


STRICT = template(50.0)
LOOSE = template(5.0)

ROWS = [{
    "plan_id": "plan-1", "slot": 1, "plate": "P1", "well": "A1",
    "candidate_id": "c1", "construct_id": "c1", "kind": "candidate",
    "role": "high_evidence", "cofactor": "NADPH", "cofactor_state": "reduced",
    "replicate": 1, "tested": "yes", "expressed_soluble": "yes",
    "detection_method": "GC-MS", "confirms_product_identity": "yes",
    "authentic_standard": "yes", "limit_of_detection": 0.1, "limit_unit": "%",
    "measurement_type": "conversion", "measurement_value": 10.0,
    "measurement_unit": "%", "conversion_pct": 10.0,
    "product_identity_observed": "target",
}]


def context(tmp: pathlib.Path) -> RunContext:
    """A task resolved far enough for the ingest step's own preconditions.

    The product structure and the functional-criteria gate are required by
    the step itself; they are satisfied here so the test exercises the
    criterion check rather than stopping at an earlier gate.
    """
    task = TaskSpec(task_id="T")
    task.resolve("reaction.product.isomeric_smiles", "C[C@H](O)c1ccccc1",
                 source="operator:test", justification="fixture")
    task.approval.functional_criteria_confirmed = True
    return RunContext(
        task=task, workdir=tmp,
        manifest=RunManifest(run_id="r", task_id="T"),
        policy=ExecutionPolicy(allow_network=False))


class DigestAgreementTests(unittest.TestCase):
    def test_selection_and_ingest_compute_the_same_digest(self) -> None:
        """If the two disagreed, the check could never pass on an honest run."""
        self.assertEqual(sha256_obj(dict(STRICT.positive_criteria)),
                         PositiveCriterion.from_template(STRICT).digest())

    def test_the_same_id_with_a_different_bar_has_a_different_digest(self) -> None:
        self.assertNotEqual(PositiveCriterion.from_template(STRICT).digest(),
                            PositiveCriterion.from_template(LOOSE).digest())
        self.assertEqual(STRICT.template_id, LOOSE.template_id,
                         "the id is what made this invisible")


class SwapRefusedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)
        self.registered = sha256_obj(dict(STRICT.positive_criteria))

    def _run(self, applied: AssayTemplate, **kw):
        return IngestResults().run(
            context(self.tmp), rows=ROWS, assay_template=applied,
            assay_run_id="run-1", **kw)

    def test_a_swapped_looser_bar_is_refused(self) -> None:
        result = self._run(LOOSE, registered_criterion_sha256=self.registered)
        self.assertFalse(result.ok, "a 10% result must not pass a 50% bar")
        codes = {f.code for f in result.qc_flags}
        self.assertIn("criterion_does_not_match_registration", codes)

    def test_the_refusal_names_both_digests_and_the_applied_bar(self) -> None:
        result = self._run(LOOSE, registered_criterion_sha256=self.registered)
        self.assertIn(self.registered[:12], result.message)
        self.assertIn("min_conversion_pct", result.message)

    def test_the_registered_bar_itself_is_accepted(self) -> None:
        result = self._run(STRICT, registered_criterion_sha256=self.registered)
        self.assertTrue(result.status.usable)
        codes = {f.code for f in result.qc_flags}
        self.assertNotIn("criterion_does_not_match_registration", codes)

    def test_without_a_registered_digest_the_gap_is_declared(self) -> None:
        """Not checkable is not the same as checked and clean."""
        result = self._run(LOOSE)
        codes = {u.code for u in result.uncertainty}
        self.assertIn("criterion_registration_unchecked", codes)

    def test_an_argument_the_step_does_not_implement_is_reported(self) -> None:
        """A silently dropped argument is how a run believes it checked something."""
        result = self._run(STRICT, registered_criterion_sha256=self.registered,
                           criterion_digest="typo-for-the-real-parameter")
        flags = [f for f in result.qc_flags if f.code == "unknown_argument"]
        self.assertTrue(flags)
        self.assertIn("criterion_digest", flags[0].message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
