"""Tests for turning a documented capability into a checked one.

The registry records what a resource is documented to offer. That is the right
thing to record and it is not a route: a documented REST API can have moved, be
behind a login, or answer with a cheerful HTML page. The model used to refuse
``connectivity_verified=True`` outright for exactly that reason -- nobody had
called anything.

A probe is what makes the flag earnable, and the markers are what make the
probe worth anything. From this environment, SABIO-RK's documented REST path
answers 302 and then 200 -- on the provider's own 404 page. A status check
passes it. So does a login wall, and so does a cached CDN error page.

No test here reaches the network: every one supplies its own fetcher.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from eagent.datalayer.probe import (
    PROBES, CapabilityProbe, ProbeError, Response, load_observed, run_probe,
    run_probes, write_checks,
)
from eagent.datalayer.registry import (
    OBSERVED_CONNECTIVITY_FILE, DataSource, SourceRegistry,
)


def probe(**kw) -> CapabilityProbe:
    base = dict(source_id="example", capability="exact_record_fetch",
                url="https://example.org/api/P07846",
                markers=("P07846",), description="fetch one record",
                documentation="https://example.org/docs")
    base.update(kw)
    return CapabilityProbe(**base)


def answering(body: str, status: int = 200, docs_status: int = 200):
    """A fetcher that answers the record URL and the documentation URL."""
    def fetch(url: str, timeout: float) -> Response:
        if url.endswith("/docs"):
            return Response(status=docs_status, body="documentation",
                            elapsed_ms=1)
        return Response(status=status, body=body, elapsed_ms=2)
    return fetch


class WhatAProbeMustDeclare(unittest.TestCase):
    def test_a_probe_with_no_markers_is_refused(self) -> None:
        with self.assertRaises(ProbeError) as ctx:
            probe(markers=())
        self.assertIn("status check", str(ctx.exception))

    def test_an_unknown_capability_is_refused(self) -> None:
        with self.assertRaises(ProbeError):
            probe(capability="does_not_exist")

    def test_a_non_https_url_is_refused(self) -> None:
        with self.assertRaises(ProbeError):
            probe(url="http://example.org/api/x")

    def test_a_probe_with_no_documentation_is_refused(self) -> None:
        """The registry refuses an endpoint with no citation; so does the probe
        that would set one."""
        with self.assertRaises(ProbeError) as ctx:
            probe(documentation="")
        self.assertIn("documentation", str(ctx.exception))

    def test_the_base_is_the_service_not_the_record(self) -> None:
        self.assertEqual(probe().base, "https://example.org")


class WhatCountsAsAnswering(unittest.TestCase):
    def test_the_record_passes(self) -> None:
        check = run_probe(probe(), fetcher=answering('{"P07846": 1}'))
        self.assertTrue(check.ok)
        self.assertEqual(check.failure, "")
        self.assertEqual(check.status_code, 200)

    def test_a_404_page_returning_200_does_not_pass(self) -> None:
        """The SABIO-RK case: a redirect lands on the provider's own 404."""
        check = run_probe(probe(), fetcher=answering("<h1>Page not found</h1>"))
        self.assertFalse(check.ok)
        self.assertIn("not the record that was asked for", check.failure)
        self.assertIn("'P07846'", check.failure)

    def test_a_login_wall_returning_200_does_not_pass(self) -> None:
        check = run_probe(probe(),
                          fetcher=answering("<form>Sign in to continue</form>"))
        self.assertFalse(check.ok)

    def test_an_http_error_is_recorded_not_raised(self) -> None:
        check = run_probe(probe(), fetcher=answering("", status=503))
        self.assertFalse(check.ok)
        self.assertIn("503", check.failure)

    def test_a_connection_failure_is_recorded_not_raised(self) -> None:
        def dead(url: str, timeout: float) -> Response:
            return Response(status=0, body="URLError: no route to host",
                            elapsed_ms=0)
        check = run_probe(probe(), fetcher=dead)
        self.assertFalse(check.ok)
        self.assertIn("did not complete", check.failure)

    def test_documentation_that_does_not_answer_fails_the_probe(self) -> None:
        """An endpoint has to be traceable to a page a person can read."""
        check = run_probe(probe(),
                          fetcher=answering('{"P07846": 1}', docs_status=404))
        self.assertFalse(check.ok)
        self.assertIn("documentation", check.failure)

    def test_the_documentation_fetch_is_recorded(self) -> None:
        check = run_probe(probe(), fetcher=answering('{"P07846": 1}'))
        self.assertEqual(check.documentation_url, "https://example.org/docs")
        self.assertEqual(check.documentation_status, 200)

    def test_the_response_is_digested_not_stored(self) -> None:
        check = run_probe(probe(), fetcher=answering('{"P07846": 1}'))
        self.assertTrue(check.response_sha256.startswith("sha256:"))
        self.assertEqual(check.response_bytes, len('{"P07846": 1}'))

    def test_a_failed_check_carries_a_reason_the_model_demands(self) -> None:
        check = run_probe(probe(), fetcher=answering("nothing"))
        self.assertFalse(check.ok)
        self.assertTrue(check.failure.strip())


class WhatGetsWritten(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)

    def results(self, *, ok: bool = True):
        body = '{"P07846": 1}' if ok else "nothing"
        return run_probes([probe()], fetcher=answering(body))

    def test_a_pass_is_written_and_reads_back(self) -> None:
        write_checks(self.results(), self.dir)
        observed = load_observed(self.dir)
        self.assertEqual(set(observed), {"example"})
        self.assertTrue(observed["example"]["connectivity_verified"])
        self.assertEqual(observed["example"]["endpoint"], "https://example.org")

    def test_a_failure_is_not_written(self) -> None:
        """Unreachable from this container now is not "this route does not work"."""
        write_checks(self.results(ok=False), self.dir)
        self.assertEqual(load_observed(self.dir), {})

    def test_the_citation_says_the_page_was_opened(self) -> None:
        write_checks(self.results(), self.dir)
        cite = load_observed(self.dir)["example"]["citations"][0]
        self.assertIn("https://example.org/docs", cite)
        self.assertIn("fetched HTTP 200", cite)

    def test_the_endpoint_is_the_service_not_the_probe_url(self) -> None:
        write_checks(self.results(), self.dir)
        endpoint = load_observed(self.dir)["example"]["endpoint"]
        self.assertNotIn("P07846", endpoint)

    def test_the_file_says_it_is_machine_written(self) -> None:
        path = write_checks(self.results(), self.dir)
        self.assertEqual(path.name, OBSERVED_CONNECTIVITY_FILE)
        self.assertIn("MACHINE-WRITTEN", path.read_text(encoding="utf-8"))

    def test_no_file_means_nothing_is_verified(self) -> None:
        self.assertEqual(load_observed(self.dir), {})

    def test_a_rewrite_replaces_rather_than_accumulates(self) -> None:
        write_checks(self.results(), self.dir)
        write_checks(self.results(), self.dir)
        self.assertEqual(
            len(load_observed(self.dir)["example"]["connectivity_checks"]), 1)


class TheRegistryDemandsTheRecord(unittest.TestCase):
    def test_the_overlay_is_merged_onto_the_curated_entry(self) -> None:
        registry = SourceRegistry.from_directory()
        source = registry.get("uniprotkb")
        self.assertTrue(source.connectivity_verified)
        self.assertIn("exact_record_fetch", source.verified_capabilities)
        # The curated judgements are untouched by a sweep.
        self.assertTrue(source.needs_legal_review)
        self.assertIsNone(source.license)

    def test_an_observation_for_an_unregistered_source_is_refused(self) -> None:
        """A verified route for a source nobody registers is a claim about
        nothing, and the right time to find out is at load."""
        from eagent.datalayer.registry import RegistryIntegrityError
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            for name in ("reaction_and_chemistry.yaml",):
                src = (pathlib.Path(__file__).resolve().parents[1]
                       / "configs" / "datasources" / name)
                (directory / name).write_text(src.read_text(encoding="utf-8"),
                                              encoding="utf-8")
            (directory / OBSERVED_CONNECTIVITY_FILE).write_text(
                "observed_connectivity:\n"
                "- id: not_a_registered_source\n"
                "  endpoint: https://example.org\n"
                "  connectivity_verified: true\n", encoding="utf-8")
            with self.assertRaises(RegistryIntegrityError) as ctx:
                SourceRegistry.from_directory(directory)
            self.assertIn("not_a_registered_source", str(ctx.exception))

    def test_the_shipped_probes_all_declare_what_they_need(self) -> None:
        for p in PROBES:
            with self.subTest(probe=f"{p.source_id}/{p.capability}"):
                self.assertTrue(p.markers)
                self.assertTrue(p.documentation.startswith("https://"))
                self.assertTrue(p.description)

    def test_every_probed_source_is_registered(self) -> None:
        registry = SourceRegistry.from_directory()
        for p in PROBES:
            with self.subTest(source=p.source_id):
                source = registry.get(p.source_id)
                self.assertIsNot(
                    source.capabilities.get(p.capability).value,
                    "not_supported",
                    f"{p.source_id} is probed for {p.capability} but registers "
                    f"it as not supported")


if __name__ == "__main__":
    unittest.main(verbosity=2)
