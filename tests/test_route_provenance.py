"""Tests for where a probe's URL came from.

A probe is a hypothesis and the sweep is the experiment. That is a good design
only while a reader can tell the two kinds of failure apart:

* the resource was unreachable today, which is news about the network;
* the candidate route was never right, which is news about
  ``src/eagent/datalayer/probe.py``.

Twelve of the shipped probes were assembled from documentation read through a
search engine, from a container whose network policy denied every one of their
hosts, and shipped before anybody had called them. That is a legitimate thing
to ship -- a wrong URL is a printed failure and nothing written -- but only if
it says so, which is what :class:`RouteProvenance` is for. The sweep of
2026-10-25 reached all twelve (two after their URLs were corrected), and each
now says so while keeping where it came from.

No test here reaches the network: every one supplies its own fetcher.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest import mock

from click.testing import CliRunner

from eagent.cli import EXIT_USAGE, main
from eagent.datalayer import probe as probe_module
from eagent.datalayer.probe import (
    PROBES, ROUTE_PROVENANCE_GRADES, CapabilityProbe, ProbeError, Response,
    RouteProvenance, load_observed, never_called_probes, probe_hosts,
    run_probe, run_probes, write_checks,
)
from eagent.datalayer.registry import SourceRegistry, default_datasource_dir


def provenance(**kw) -> RouteProvenance:
    base = dict(grade="provider_documentation",
                detail="read off the provider's own documentation page",
                recorded_at="2026-10-25")
    base.update(kw)
    return RouteProvenance(**base)


def probe(**kw) -> CapabilityProbe:
    base = dict(source_id="example", capability="exact_record_fetch",
                url="https://example.org/api/P07846",
                markers=("P07846",), description="fetch one record",
                provenance=provenance(),
                documentation="https://example.org/docs")
    base.update(kw)
    return CapabilityProbe(**base)


def answering(body: str, status: int = 200, docs_status: int = 200):
    def fetch(url: str, timeout: float) -> Response:
        if url.endswith("/docs"):
            return Response(status=docs_status, body="documentation",
                            elapsed_ms=1)
        return Response(status=status, body=body, elapsed_ms=2)
    return fetch


class WhatAProvenanceMustDeclare(unittest.TestCase):
    def test_an_unknown_grade_is_refused(self) -> None:
        with self.assertRaises(ProbeError) as ctx:
            provenance(grade="seemed_right")
        self.assertIn("unknown route provenance grade", str(ctx.exception))

    def test_a_grade_with_no_detail_is_refused(self) -> None:
        with self.assertRaises(ProbeError) as ctx:
            provenance(detail="docs")
        self.assertIn("which part of the URL was guessed", str(ctx.exception))

    def test_a_date_that_is_not_a_date_is_refused(self) -> None:
        for bad in ("2026-10", "25/10/2026", "today", "2026-1-5"):
            with self.subTest(bad=bad), self.assertRaises(ProbeError):
                provenance(recorded_at=bad)

    def test_a_search_summary_without_its_query_is_refused(self) -> None:
        with self.assertRaises(ProbeError) as ctx:
            provenance(grade="search_summary", query="   ")
        self.assertIn("nobody can re-run", str(ctx.exception))

    def test_a_search_summary_with_its_query_is_accepted(self) -> None:
        got = provenance(grade="search_summary", query="KEGG REST API get")
        self.assertEqual(got.query, "KEGG REST API get")

    def test_a_probe_cannot_be_built_without_one(self) -> None:
        """No default. A probe that could skip it would re-admit the thing the
        registry nulled five endpoints to get rid of: a URL in the code that
        nobody can trace."""
        with self.assertRaises(TypeError):
            CapabilityProbe(source_id="example",
                            capability="exact_record_fetch",
                            url="https://example.org/api/X",
                            markers=("X",), description="d",
                            documentation="https://example.org/docs")


class WhatEverCalledMeans(unittest.TestCase):
    def test_only_called_from_here_counts_as_called(self) -> None:
        for grade in ROUTE_PROVENANCE_GRADES:
            kw = {"query": "q"} if grade == "search_summary" else {}
            with self.subTest(grade=grade):
                self.assertEqual(provenance(grade=grade, **kw).ever_called,
                                 grade == "called_from_here")

    def test_documentation_that_was_read_is_not_a_route_that_was_called(self) -> None:
        """The distinction the whole class exists for. Reading a provider's
        page tells you what the route should be; it does not tell you that the
        route answers, and only a call can."""
        self.assertFalse(provenance(grade="provider_documentation").ever_called)

    def test_the_record_carries_whether_it_was_called(self) -> None:
        self.assertIs(provenance().to_dict()["ever_called"], False)

    def test_an_absent_query_is_left_out_of_the_record(self) -> None:
        self.assertNotIn("query", provenance().to_dict())

    def test_a_reached_route_counts_as_called(self) -> None:
        reached = provenance().reached("2026-10-26")
        self.assertTrue(reached.ever_called)
        self.assertEqual(reached.grade, "called_from_here")
        self.assertEqual(reached.recorded_at, "2026-10-26")

    def test_reaching_a_route_keeps_where_it_came_from(self) -> None:
        """Where a URL came from is the first thing to check when it stops
        working, so a pass adds to the pedigree instead of replacing it."""
        before = provenance(grade="search_summary", query="the search behind it")
        after = before.reached("2026-10-26")
        self.assertIn(before.detail, after.detail)
        self.assertIn("search_summary", after.detail)
        self.assertEqual(after.query, "the search behind it")

    def test_reaching_a_route_already_called_changes_nothing(self) -> None:
        called = provenance().reached("2026-10-26")
        self.assertIs(called.reached("2026-10-27"), called)


def candidates_now_or_before() -> list[CapabilityProbe]:
    """Shipped probes that are candidates, or were until a sweep reached them."""
    return [p for p in PROBES if not p.provenance.ever_called
            or "Before that it was a candidate" in p.provenance.detail]


class TheShippedTable(unittest.TestCase):
    def test_every_shipped_probe_declares_its_provenance(self) -> None:
        for p in PROBES:
            with self.subTest(source=p.source_id, capability=p.capability):
                self.assertIsInstance(p.provenance, RouteProvenance)

    def test_a_probe_claiming_it_was_called_has_a_recorded_pass(self) -> None:
        """The invariant that keeps the claim honest. ``called_from_here`` is
        the one grade that asserts something about this repo's own history, so
        it has to agree with the file that holds that history -- otherwise the
        strongest grade would be the one nobody checks."""
        observed = load_observed(default_datasource_dir())
        for p in PROBES:
            if not p.provenance.ever_called:
                continue
            with self.subTest(source=p.source_id):
                self.assertIn(p.source_id, observed)
                urls = {c["url"] for c
                        in observed[p.source_id]["connectivity_checks"]}
                self.assertIn(p.url, urls)

    def test_a_source_with_no_recorded_pass_claims_no_call(self) -> None:
        observed = load_observed(default_datasource_dir())
        for p in PROBES:
            if p.source_id in observed:
                continue
            with self.subTest(source=p.source_id):
                self.assertFalse(p.provenance.ever_called)

    def test_a_probe_with_a_recorded_pass_says_it_was_called(self) -> None:
        """The other direction. A sweep that records a pass at a candidate's
        URL makes "never called" false, and a dry run printing NEVER CALLED
        beside a recorded pass would be the table contradicting the file."""
        observed = load_observed(default_datasource_dir())
        for p in PROBES:
            checks = observed.get(p.source_id, {}).get("connectivity_checks", ())
            if p.url not in {c["url"] for c in checks if c.get("ok")}:
                continue
            with self.subTest(source=p.source_id):
                self.assertTrue(p.provenance.ever_called)

    def test_the_candidates_are_the_ones_never_called(self) -> None:
        never = never_called_probes()
        self.assertEqual(set(never),
                         {p for p in PROBES if not p.provenance.ever_called})

    def test_a_candidate_names_what_the_author_supplied(self) -> None:
        """Every candidate's detail has to say more than which page was read,
        because the common failure is a good route with an id this project
        chose itself. A route a sweep has since reached keeps that detail."""
        candidates = candidates_now_or_before()
        self.assertTrue(candidates, "the candidate probes have gone missing")
        for p in candidates:
            with self.subTest(source=p.source_id):
                self.assertGreater(len(p.provenance.detail), 80)

    def test_every_candidate_carries_the_search_behind_it(self) -> None:
        for p in candidates_now_or_before():
            with self.subTest(source=p.source_id):
                self.assertTrue(p.provenance.query.strip())


class TheHostsASweepNeeds(unittest.TestCase):
    def test_documentation_hosts_are_in_the_list(self) -> None:
        """The regression this function exists for. A probe fails when its
        documentation is unreachable as well as when its route is, so an
        allow-list built from route URLs alone makes every probe fail on its
        documentation -- which reads like the resources being down."""
        hosts = probe_hosts()
        self.assertIn("www.rcsb.org", hosts)             # docs only
        self.assertIn("data.rcsb.org", hosts)            # route only
        self.assertIn("docs: rcsb_pdb", hosts["www.rcsb.org"])
        self.assertNotIn("route: rcsb_pdb/exact_record_fetch",
                         hosts["www.rcsb.org"])

    def test_every_probe_url_and_documentation_url_is_covered(self) -> None:
        hosts = set(probe_hosts())
        for p in PROBES:
            with self.subTest(source=p.source_id):
                for url in (p.url, p.documentation):
                    self.assertIn(url.split("://", 1)[1].split("/", 1)[0],
                                  hosts)

    def test_a_reason_is_given_for_every_host(self) -> None:
        for host, why in probe_hosts().items():
            with self.subTest(host=host):
                self.assertTrue(why)
                for reason in why:
                    self.assertTrue(reason.startswith(("route: ", "docs: ")))

    def test_it_is_derived_from_the_probes_it_is_given(self) -> None:
        one = probe(url="https://only.example/api/x",
                    documentation="https://only.example/docs")
        self.assertEqual(list(probe_hosts([one])), ["only.example"])


class WhatACandidateFailureDoesNotDo(unittest.TestCase):
    """The reason it is safe to ship a route nobody has called."""

    def test_a_wrong_route_writes_nothing(self) -> None:
        candidate = probe(url="https://example.org/api/wrong")
        check = run_probe(candidate, fetcher=answering("not found", status=404))
        self.assertFalse(check.ok)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_checks({"example": [check]}, pathlib.Path(tmp))
            self.assertNotIn("example", path.read_text(encoding="utf-8"))

    def test_a_route_that_answers_the_wrong_body_writes_nothing(self) -> None:
        check = run_probe(probe(), fetcher=answering("a cheerful html page"))
        self.assertFalse(check.ok)
        self.assertIn("it is not the record that was asked for", check.failure)

    def test_a_route_whose_documentation_is_gone_is_not_recorded(self) -> None:
        """Half the candidates' documentation URLs are themselves guesses, so
        this is the failure several of them will produce first."""
        check = run_probe(probe(),
                          fetcher=answering("P07846", docs_status=404))
        self.assertFalse(check.ok)
        self.assertIn("documentation", check.failure)
        self.assertIn("a person can read", check.failure)

    def test_the_sweep_does_not_stop_at_the_first_failure(self) -> None:
        results = run_probes(
            [probe(source_id="a"), probe(source_id="b", markers=("absent",))],
            fetcher=answering("P07846"))
        self.assertTrue(results["a"][0].ok)
        self.assertFalse(results["b"][0].ok)


class TheDocumentationCheck(unittest.TestCase):
    """A 200 is not a page, any more than it is a record.

    The module refuses a markerless probe because a redirect to a 404 page and
    a login wall both answer 200. The documentation fetch was making the same
    mistake one level up: it was status-only. A sweep on an open network found
    it passing on NCBI Bookshelf NBK25501 and PubChem's ``docs/pug-rest``,
    which answer 200 with a "Checking your browser" reCAPTCHA page, and on
    ``www.uniprot.org/help/api``, whose single-page-app shell is byte-identical
    to the shell served for a help page that does not exist.

    These tests use synthetic probes. Nothing here pins a shipped probe's
    markers: filling those in needs a network that can read the page, and a
    test that pinned them empty would break the moment somebody did.
    """

    def docs_body(self, body: str, status: int = 200):
        def fetch(url: str, timeout: float) -> Response:
            if url.endswith("/docs"):
                return Response(status=status, body=body, elapsed_ms=1)
            return Response(status=200, body="P07846", elapsed_ms=2)
        return fetch

    def test_without_markers_the_check_stays_status_only(self) -> None:
        """The older behaviour is the default, so adding the field changes no
        existing probe's verdict."""
        check = run_probe(probe(), fetcher=self.docs_body("anything at all"))
        self.assertTrue(check.ok, check.failure)
        self.assertEqual(check.documentation_markers, [])

    def test_markers_that_are_present_pass(self) -> None:
        check = run_probe(probe(documentation_markers=("GET /api/{id}",)),
                          fetcher=self.docs_body("the route is GET /api/{id}"))
        self.assertTrue(check.ok, check.failure)

    def test_a_recaptcha_interstitial_no_longer_passes(self) -> None:
        check = run_probe(probe(documentation_markers=("PUG-REST",)),
                          fetcher=self.docs_body(
                              "Checking your browser - reCAPTCHA"))
        self.assertFalse(check.ok)
        self.assertIn("'PUG-REST'", check.failure)
        self.assertIn("it is not the documentation", check.failure)

    def test_an_app_shell_no_longer_passes(self) -> None:
        shell = "<div id=root></div>"      # the same for a page that exists
        check = run_probe(probe(documentation_markers=("UniProt REST",)),
                          fetcher=self.docs_body(shell))
        self.assertFalse(check.ok)
        self.assertIn("single-page-app shell", check.failure)

    def test_the_failure_names_every_missing_marker(self) -> None:
        check = run_probe(probe(documentation_markers=("one", "two", "three")),
                          fetcher=self.docs_body("two"))
        self.assertIn("'one'", check.failure)
        self.assertIn("'three'", check.failure)
        self.assertNotIn("'two'", check.failure)

    def test_a_route_failure_is_still_reported_as_the_route(self) -> None:
        """Order matters: a dead route with good documentation must not read as
        a documentation problem."""
        check = run_probe(probe(documentation_markers=("absent",)),
                          fetcher=answering("not found", status=404))
        self.assertFalse(check.ok)
        self.assertIn("HTTP 404", check.failure)
        self.assertNotIn("not the documentation", check.failure)

    def test_the_record_says_which_kind_of_check_it_was(self) -> None:
        """An empty list is part of the record, not an absent detail: it says
        the documentation was taken on its status alone."""
        status_only = run_probe(probe(), fetcher=self.docs_body("x"))
        with_markers = run_probe(probe(documentation_markers=("x",)),
                                 fetcher=self.docs_body("x"))
        self.assertEqual(status_only.documentation_markers, [])
        self.assertEqual(with_markers.documentation_markers, ["x"])


class TheCommands(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_hosts_plain_prints_only_host_names(self) -> None:
        result = self.runner.invoke(main, ["sources", "hosts", "--plain"])
        self.assertEqual(result.exit_code, 0, result.output)
        printed = [line for line in result.output.splitlines() if line.strip()]
        self.assertEqual(printed, list(probe_hosts()))
        for line in printed:
            self.assertNotIn("/", line)

    def test_hosts_explains_what_needs_each_one(self) -> None:
        result = self.runner.invoke(main, ["sources", "hosts"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("docs: uniprotkb", result.output)
        self.assertIn("route: uniprotkb/exact_record_fetch", result.output)

    def test_hosts_says_which_sources_it_does_not_cover(self) -> None:
        """The list is honest about being the probes' hosts and not the
        registry's: a reader who pasted it into a network policy and expected
        all 49 sources to answer would be wrong."""
        result = self.runner.invoke(main, ["sources", "hosts"])
        self.assertIn("record no endpoint, and have no probe either",
                      result.output)

    def test_hosts_for_one_source(self) -> None:
        result = self.runner.invoke(
            main, ["sources", "hosts", "--source", "kegg", "--plain"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(sorted(result.output.split()),
                         ["rest.kegg.jp", "www.kegg.jp"])

    def test_hosts_refuses_an_unknown_source(self) -> None:
        result = self.runner.invoke(
            main, ["sources", "hosts", "--source", "not_a_source"])
        self.assertEqual(result.exit_code, EXIT_USAGE, result.output)

    def with_a_candidate(self):
        """The shipped table plus one route nobody has called.

        Every shipped candidate has been reached by now, so the commands'
        handling of a candidate is tested on one made for the purpose -- the
        next route added from documentation will need it, and these tests
        should not wait for one to ship.
        """
        candidate = probe(
            source_id="chebi", url="https://example.org/chebi/CHEBI:15377",
            markers=("CHEBI:15377",),
            provenance=provenance(grade="third_party_client"))
        return mock.patch.object(probe_module, "PROBES", PROBES + (candidate,))

    def test_the_dry_run_marks_the_routes_nobody_has_called(self) -> None:
        with self.with_a_candidate():
            result = self.runner.invoke(
                main, ["sources", "verify", "--no-network", "--source", "chebi"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("NEVER CALLED", result.output)
        self.assertIn("third_party_client", result.output)
        self.assertIn("have never been called", result.output)

    def test_coverage_counts_candidates_apart_from_verified_sources(self) -> None:
        """Two numbers, never one. A route to try and a route that works are
        different facts, and a reader given one number would have to guess
        which of the two it was."""
        with self.with_a_candidate():
            result = self.runner.invoke(main, ["sources", "coverage"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("carry a recorded connectivity check", result.output)
        self.assertIn("1 further source(s) carry a candidate route",
                      result.output)
        self.assertIn("(chebi)", result.output)
        self.assertIn("not a route that works", result.output)

    def test_coverage_does_not_fold_candidates_into_the_verified_count(self) -> None:
        registry = SourceRegistry.from_directory()
        sources = list(registry)
        verified_ids = [s_.id for s_ in sources if s_.verified_capabilities]
        with self.with_a_candidate():
            result = self.runner.invoke(main, ["sources", "coverage"])
        verified = [s_ for s_ in result.output.splitlines()
                    if "carry a recorded connectivity check" in s_]
        self.assertEqual(len(verified), 1, result.output)
        self.assertIn(f"{len(verified_ids)} of {len(sources)}", verified[0])
        self.assertNotIn("chebi", verified_ids)
        self.assertNotIn("chebi", verified[0])

    def test_coverage_names_no_candidate_when_every_route_was_reached(self) -> None:
        """An empty list is not printed as "0 further source(s)": with every
        shipped route reached, the line would be a count of nothing."""
        with mock.patch.object(probe_module, "PROBES", tuple(
                p for p in PROBES if p.provenance.ever_called)):
            result = self.runner.invoke(main, ["sources", "coverage"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("carry a candidate route", result.output)

    def test_the_dry_run_does_not_mark_a_route_with_a_recorded_pass(self) -> None:
        result = self.runner.invoke(
            main, ["sources", "verify", "--no-network", "--source", "uniprotkb"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("NEVER CALLED", result.output)
        self.assertIn("called before", result.output)


if __name__ == "__main__":
    unittest.main()
