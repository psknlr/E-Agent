"""The RCSB file route: probed first, hashed before it is named, verified when present.

``RCSBPDBConnector.download_structure`` fetches one entry's asymmetric-unit mmCIF
from the file host, which is a different service from the data API. These tests
defend the order of operations -- the route must have a passing probe on record
(and the host called is the one that probe passed for, read from the registry,
never a literal in the code), the body is read under a cap, checked to be *this
entry's* mmCIF, hashed against the pin, and only then written under its final
name -- and that a file already on disk is verified, not replaced.

No test touches the network. The body served by the fake route is a one-atom
mmCIF built here.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from eagent.connectors.base import AccessPolicy, FileCache, NetworkDisabledError
from eagent.connectors.chemistry import LayerSemanticsError
from eagent.connectors.literature import ChecksumMismatchError
from eagent.connectors.structure import (
    DEFAULT_MAX_STRUCTURE_BYTES, FileRouteNotVerifiedError,
    RCSB_STRUCTURE_FILE_PATH, RCSBPDBConnector, validate_mmcif_bytes,
)
from eagent.datalayer.probe import PROBES
from eagent.datalayer.registry import SourceRegistry, default_datasource_dir
from eagent.science.structure_io import mmcif_categories, read_mmcif

#: The host the shipped probe passed for. A test may name a host; the connector
#: package may not (see test_connectors_structure: no module hard-codes a URL).
FILES_HOST = "https://files.rcsb.org"

EXCERPT = (
    "data_1IPF\n#\n_entry.id 1IPF\n#\n"
    "_struct.title 'TROPINONE REDUCTASE-II COMPLEXED WITH NADPH AND TROPINONE'\n#\n"
    "loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
    "_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n"
    "_atom_site.label_asym_id\n_atom_site.label_seq_id\n_atom_site.Cartn_x\n"
    "_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n"
    "_atom_site.B_iso_or_equiv\n_atom_site.auth_seq_id\n_atom_site.auth_comp_id\n"
    "_atom_site.auth_asym_id\n_atom_site.auth_atom_id\n"
    "HETATM 1 C C3 . TNE A . 6.553 33.412 147.697 1.00 30.0 262 TNE A C3\n#\n"
).encode("utf-8")
EXCERPT_SHA = hashlib.sha256(EXCERPT).hexdigest()


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dest = self.root / "pdb"

    def connector(self, body: bytes | None = EXCERPT, *, registry=None,
                  allow_network: bool = True) -> RCSBPDBConnector:
        connector = RCSBPDBConnector(
            cache=FileCache(self.root / "cache"),
            access=AccessPolicy(allow_network=allow_network), registry=registry)
        self.urls: list[str] = []
        self.caps: list[int] = []

        def http_bytes(url, *, max_bytes, timeout=120.0, chunk=1 << 16):
            self.urls.append(url)
            self.caps.append(max_bytes)
            return body

        connector._http_bytes = http_bytes
        return connector


# ==========================================================================
class TheRouteIsTheOneThatWasProbed(unittest.TestCase):
    def test_the_url_the_client_builds_is_the_url_a_probe_fetched(self) -> None:
        connector = RCSBPDBConnector(access=AccessPolicy(allow_network=False))
        built = connector._file_route_base() + RCSB_STRUCTURE_FILE_PATH.format(pdb_id="1IPF")
        self.assertIn(built, {p.url for p in PROBES if p.source_id == "rcsb_pdb"})

    def test_the_file_host_is_not_the_data_api_host_and_is_read_from_the_registry(self) -> None:
        registry = SourceRegistry.from_directory()
        connector = RCSBPDBConnector(access=AccessPolicy(allow_network=False))
        self.assertEqual(connector._file_route_base(), FILES_HOST)
        self.assertNotEqual(connector._file_route_base(),
                            registry.get("rcsb_pdb").endpoint)

    def test_the_probe_has_markers_that_only_the_real_file_carries(self) -> None:
        probe = next(p for p in PROBES if p.url.startswith(FILES_HOST))
        self.assertIn("_atom_site.Cartn_x", probe.markers)
        self.assertIn("data_1IPF", probe.markers)
        self.assertTrue(probe.documentation.startswith("https://www.rcsb.org/docs/"))

    def test_the_shipped_registry_records_a_passing_check_for_it(self) -> None:
        source = SourceRegistry.from_directory().get("rcsb_pdb")
        checks = [c for c in source.connectivity_checks
                  if c.ok and c.url.startswith(FILES_HOST + "/download/")]
        self.assertTrue(checks, "run `eagent sources verify --allow-network --write`")
        self.assertIn("_atom_site.Cartn_x", checks[0].markers)


# ==========================================================================
class ADownloadCannotPutABadFileUnderAGoodName(_Tmp):
    def test_a_good_file_is_written_and_named_by_its_hash(self) -> None:
        got = self.connector().download_structure("1ipf", self.dest,
                                                  expected_sha256=EXCERPT_SHA)
        self.assertEqual(got.path, self.dest / "1IPF.cif")
        self.assertEqual(got.sha256, EXCERPT_SHA)
        self.assertEqual(got.size_bytes, len(EXCERPT))
        self.assertFalse(got.from_cache)
        self.assertEqual(self.urls, [FILES_HOST + "/download/1IPF.cif"])
        self.assertEqual(self.caps, [DEFAULT_MAX_STRUCTURE_BYTES])
        self.assertEqual(list(self.dest.glob(".*part")), [], "no temporary file left behind")

    def test_a_hash_that_is_not_the_pinned_one_writes_nothing(self) -> None:
        with self.assertRaisesRegex(ChecksumMismatchError, "different revision"):
            self.connector().download_structure("1IPF", self.dest,
                                                expected_sha256="0" * 64)
        self.assertFalse((self.dest / "1IPF.cif").exists())

    def test_an_error_page_with_status_200_is_not_a_coordinate_file(self) -> None:
        html = b"<!doctype html><html><body>Service unavailable</body></html>"
        with self.assertRaisesRegex(LayerSemanticsError, "something other than"):
            self.connector(html).download_structure("1IPF", self.dest)
        self.assertFalse(self.dest.exists() and any(self.dest.iterdir()))

    def test_another_entrys_file_under_this_entrys_name_is_refused(self) -> None:
        other = EXCERPT.replace(b"data_1IPF", b"data_2AE2").replace(b"_entry.id 1IPF", b"_entry.id 2AE2")
        with self.assertRaises(LayerSemanticsError):
            self.connector(other).download_structure("1IPF", self.dest)

    def test_a_file_whose_entry_id_disagrees_with_its_data_block_is_refused(self) -> None:
        confused = EXCERPT.replace(b"_entry.id 1IPF", b"_entry.id 9XYZ")
        with self.assertRaisesRegex(LayerSemanticsError, "_entry.id"):
            self.connector(confused).download_structure("1IPF", self.dest)

    def test_a_file_with_no_coordinates_is_refused(self) -> None:
        stub = b"data_1IPF\n_entry.id 1IPF\n"
        with self.assertRaisesRegex(LayerSemanticsError, "no _atom_site"):
            self.connector(stub).download_structure("1IPF", self.dest)

    def test_binary_content_is_refused(self) -> None:
        with self.assertRaisesRegex(LayerSemanticsError, "not UTF-8"):
            self.connector(b"\x1f\x8b\x08\x00\xff\xfe").download_structure("1IPF", self.dest)

    def test_a_404_is_a_refusal_naming_the_entry_not_an_empty_file(self) -> None:
        with self.assertRaisesRegex(LayerSemanticsError, "no coordinate file for 1IPF"):
            self.connector(None).download_structure("1IPF", self.dest)

    def test_an_identifier_is_validated_before_a_url_is_built_from_it(self) -> None:
        connector = self.connector()
        for bad in ("../x", "1IP", "1IPFF", "", "1IP/", "zIPF "):
            with self.assertRaises(LayerSemanticsError, msg=bad):
                connector.download_structure(bad, self.dest)
        self.assertEqual(self.urls, [])

    def test_the_size_cap_is_passed_to_the_reader_not_checked_afterwards(self) -> None:
        self.connector().download_structure("1IPF", self.dest, max_bytes=1234)
        self.assertEqual(self.caps, [1234])


class AFileThatIsAlreadyThereIsVerifiedNotReplaced(_Tmp):
    def _place(self, body: bytes) -> Path:
        self.dest.mkdir(parents=True)
        path = self.dest / "1IPF.cif"
        path.write_bytes(body)
        return path

    def test_a_present_file_with_the_pinned_hash_needs_no_network(self) -> None:
        self._place(EXCERPT)
        got = self.connector(allow_network=False).download_structure(
            "1IPF", self.dest, expected_sha256=EXCERPT_SHA)
        self.assertTrue(got.from_cache)
        self.assertIsNone(got.retrieved_at, "nothing was fetched, so no fetch time is claimed")
        self.assertEqual(self.urls, [])

    def test_a_present_file_with_another_hash_is_left_alone_and_the_call_raises(self) -> None:
        path = self._place(EXCERPT + b"# edited\n")
        before = path.read_bytes()
        with self.assertRaisesRegex(ChecksumMismatchError, "left in place"):
            self.connector().download_structure("1IPF", self.dest,
                                                expected_sha256=EXCERPT_SHA)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.urls, [], "a failed verification must not trigger a re-download")

    def test_refresh_is_an_explicit_request_to_fetch_again(self) -> None:
        path = self._place(b"stale")
        got = self.connector().download_structure("1IPF", self.dest, refresh=True)
        self.assertEqual(path.read_bytes(), EXCERPT)
        self.assertFalse(got.from_cache)

    def test_a_present_file_that_is_not_a_coordinate_file_is_refused(self) -> None:
        self._place(b"<html></html>")
        with self.assertRaises(LayerSemanticsError):
            self.connector().download_structure("1IPF", self.dest)


class TheNetworkAndTheRouteAreBothRequired(_Tmp):
    def test_with_the_network_off_nothing_is_called_and_the_error_says_why(self) -> None:
        with self.assertRaisesRegex(NetworkDisabledError, "allow_network is false"):
            self.connector(allow_network=False).download_structure("1IPF", self.dest)
        self.assertEqual(self.urls, [])

    def test_a_registry_with_no_passing_check_for_the_file_host_refuses(self) -> None:
        directory = self.root / "datasources"
        directory.mkdir()
        for path in default_datasource_dir().glob("*.yaml"):
            if path.name != "connectivity.observed.yaml":
                shutil.copy(path, directory / path.name)
        observed = yaml.safe_load((default_datasource_dir() / "connectivity.observed.yaml")
                                  .read_text(encoding="utf-8"))
        for entry in observed["observed_connectivity"]:
            entry["connectivity_checks"] = [
                c for c in entry["connectivity_checks"]
                if not c["url"].startswith(FILES_HOST)]
        (directory / "connectivity.observed.yaml").write_text(
            yaml.safe_dump(observed), encoding="utf-8")
        registry = SourceRegistry.from_directory(directory)
        # the data API route is still verified: only the file host is not
        self.assertTrue(registry.get("rcsb_pdb").connectivity_verified)
        with self.assertRaisesRegex(FileRouteNotVerifiedError,
                                    "different services"):
            self.connector(registry=registry).download_structure("1IPF", self.dest)
        self.assertEqual(self.urls, [])

    def test_the_host_called_is_the_host_the_registry_recorded_not_one_in_the_code(self) -> None:
        directory = self.root / "datasources"
        directory.mkdir()
        for path in default_datasource_dir().glob("*.yaml"):
            if path.name != "connectivity.observed.yaml":
                shutil.copy(path, directory / path.name)
        observed = yaml.safe_load((default_datasource_dir() / "connectivity.observed.yaml")
                                  .read_text(encoding="utf-8"))
        for entry in observed["observed_connectivity"]:
            for c in entry["connectivity_checks"]:
                if c["url"].startswith(FILES_HOST):
                    c["url"] = c["url"].replace("files.rcsb.org", "files.example.org")
        (directory / "connectivity.observed.yaml").write_text(
            yaml.safe_dump(observed), encoding="utf-8")
        self.connector(registry=SourceRegistry.from_directory(directory)
                       ).download_structure("1IPF", self.dest)
        self.assertEqual(self.urls, ["https://files.example.org/download/1IPF.cif"])

    def test_a_failing_check_does_not_count_as_a_verified_route(self) -> None:
        directory = self.root / "datasources"
        directory.mkdir()
        for path in default_datasource_dir().glob("*.yaml"):
            if path.name != "connectivity.observed.yaml":
                shutil.copy(path, directory / path.name)
        observed = yaml.safe_load((default_datasource_dir() / "connectivity.observed.yaml")
                                  .read_text(encoding="utf-8"))
        for entry in observed["observed_connectivity"]:
            for c in entry["connectivity_checks"]:
                if c["url"].startswith(FILES_HOST):
                    c["ok"] = False
                    c["failure"] = "HTTP 503"
        (directory / "connectivity.observed.yaml").write_text(
            yaml.safe_dump(observed), encoding="utf-8")
        with self.assertRaises(FileRouteNotVerifiedError):
            self.connector(registry=SourceRegistry.from_directory(directory)
                           ).download_structure("1IPF", self.dest)


class TheValidatorIsSpecific(unittest.TestCase):
    def test_a_minimal_mmcif_passes_and_returns_its_text(self) -> None:
        text = validate_mmcif_bytes(EXCERPT, "1ipf")
        self.assertTrue(text.startswith("data_1IPF"))
        self.assertIn("_atom_site.Cartn_x", text)

    def test_a_quoted_entry_id_is_accepted_because_cif_allows_it(self) -> None:
        quoted = EXCERPT.replace(b"_entry.id 1IPF", b"_entry.id '1IPF'")
        validate_mmcif_bytes(quoted, "1IPF")


class TheCategoryReaderUsesTheCoordinateReadersQuoting(unittest.TestCase):
    def test_single_values_and_quoted_values_are_read(self) -> None:
        cats = mmcif_categories(EXCERPT.decode("utf-8"), ["_struct", "_entry"])
        self.assertEqual(cats["_entry"][0]["id"], "1IPF")
        self.assertEqual(cats["_struct"][0]["title"],
                         "TROPINONE REDUCTASE-II COMPLEXED WITH NADPH AND TROPINONE")

    def test_a_loop_comes_back_as_one_row_per_record(self) -> None:
        text = ("data_X\nloop_\n_entity.id\n_entity.type\n1 polymer\n2 non-polymer\n"
                "3 water\n")
        rows = mmcif_categories(text, ["_entity"])["_entity"]
        self.assertEqual([r["type"] for r in rows], ["polymer", "non-polymer", "water"])

    def test_a_truncated_loop_is_refused_not_padded(self) -> None:
        from eagent.science.structure_io import StructureParseError
        with self.assertRaises(StructureParseError):
            mmcif_categories("data_X\nloop_\n_entity.id\n_entity.type\n1 polymer 2\n",
                             ["_entity"])

    def test_the_coordinate_reader_still_reads_the_excerpt(self) -> None:
        structure = read_mmcif(EXCERPT.decode("utf-8"), structure_id="1IPF")
        self.assertEqual(structure.n_atoms(), 1)


if __name__ == "__main__":
    unittest.main()
