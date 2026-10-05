"""Tests for the map from a registered data source to the code that reads it.

Before this existed, the five per-resource connector modules imported cleanly
and were referenced by nothing, so a run could register 49 sources and reach
none of them. The catalogue's job is to make that reachable *and* to say
honestly which sources have real code behind them.
"""

from __future__ import annotations

import pathlib
import unittest

from eagent.connectors.base import OfflineConnector
from eagent.connectors.catalog import (
    CONCRETE_CONNECTORS, CURATED_IMPORTERS, AccessKind, build_connectors,
    connector_class_for, coverage_report, importer_class_for,
)
from eagent.datalayer.registry import SourceRegistry

CONFIGS = pathlib.Path(__file__).resolve().parent.parent / "configs" / "datasources"


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = SourceRegistry.from_directory(CONFIGS)

    # -- the catalogue matches the registry ------------------------------
    def test_every_catalogued_id_is_a_registered_source(self) -> None:
        """A catalogue entry for a source nobody registered is dead weight."""
        known = {s.id for s in self.registry}
        for sid in list(CONCRETE_CONNECTORS) + list(CURATED_IMPORTERS):
            self.assertIn(sid, known, f"{sid} is catalogued but not registered")

    def test_no_source_is_both_a_client_and_an_importer(self) -> None:
        overlap = set(CONCRETE_CONNECTORS) & set(CURATED_IMPORTERS)
        self.assertEqual(overlap, set(),
                         "a source is reached one way or the other, not both")

    def test_every_catalogue_entry_resolves_to_a_real_class(self) -> None:
        """Catches a typo at test time rather than deep inside a run."""
        for sid in CONCRETE_CONNECTORS:
            self.assertIsNotNone(connector_class_for(sid), sid)
        for sid in CURATED_IMPORTERS:
            self.assertIsNotNone(importer_class_for(sid), sid)

    def test_an_uncatalogued_source_has_no_class(self) -> None:
        self.assertIsNone(connector_class_for("not_a_source"))
        self.assertIsNone(importer_class_for("not_a_source"))

    # -- the human-import rule -------------------------------------------
    def test_human_import_only_sources_get_importers_never_clients(self) -> None:
        """A website-only resource must not be dressed up as a live API."""
        for source in self.registry:
            if getattr(source, "is_human_import_only", False):
                self.assertNotIn(
                    source.id, CONCRETE_CONNECTORS,
                    f"{source.id} is human-import only and must not have a client")
                self.assertIn(
                    source.id, CURATED_IMPORTERS,
                    f"{source.id} is human-import only and needs an importer")

    def test_the_named_offline_resources_are_importers(self) -> None:
        for sid in ("sdred", "akr_superfamily", "retrobiocat_db", "oed",
                    "protabank", "strenda_db"):
            self.assertIn(sid, CURATED_IMPORTERS)

    # -- building ---------------------------------------------------------
    def test_every_registered_source_wires_without_failure(self) -> None:
        report = build_connectors([s.id for s in self.registry],
                                  registry=self.registry)
        self.assertEqual(report.failures, {})
        self.assertEqual(len(report.built), len(list(self.registry)))

    def test_a_source_with_no_code_falls_back_to_cache_only_and_says_so(self) -> None:
        """The honest fallback: it can serve a cached record, not fetch one."""
        report = build_connectors(["bacdive"], registry=self.registry)
        built = report.built["bacdive"]
        self.assertEqual(built.kind, AccessKind.CACHE_ONLY)
        self.assertIsInstance(built.obj, OfflineConnector)
        self.assertIn("no client written yet", built.reason)
        self.assertIn("bacdive", report.by_kind(AccessKind.CACHE_ONLY))

    def test_the_report_states_what_can_actually_be_read(self) -> None:
        report = build_connectors([s.id for s in self.registry],
                                  registry=self.registry)
        lines = report.report_lines()
        self.assertTrue(lines[0].startswith(f"{len(list(self.registry))} source(s) wired"))
        self.assertIn("cache-only", lines[0])
        # Every cache-only source is named, so the shortfall is not a bare count.
        for sid in report.by_kind(AccessKind.CACHE_ONLY):
            self.assertTrue(any(sid in line for line in lines), sid)

    def test_coverage_partitions_every_registered_source_exactly_once(self) -> None:
        cov = coverage_report(self.registry)
        total = sum(len(v) for v in cov.values())
        self.assertEqual(total, len(list(self.registry)))
        seen: set[str] = set()
        for ids in cov.values():
            self.assertFalse(seen & set(ids), "a source appears in two buckets")
            seen |= set(ids)

    def test_a_client_is_built_for_a_programmatic_source(self) -> None:
        report = build_connectors(["rhea"], registry=self.registry)
        self.assertEqual(report.built["rhea"].kind, AccessKind.CONNECTOR)

    def test_an_importer_is_built_for_a_curated_source(self) -> None:
        report = build_connectors(["sdred"], registry=self.registry)
        self.assertEqual(report.built["sdred"].kind, AccessKind.IMPORTER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
