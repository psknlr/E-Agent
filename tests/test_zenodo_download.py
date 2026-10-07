"""A dataset download that cannot put a bad file under a good name.

The SDR substrate-classification deposit (Zenodo 7141435, CC BY 4.0) is the
first dataset this project ingests. The connector fetches its record, then its
files, and what these tests defend is the order of operations: bytes are read
under a hard cap, hashed, compared with the checksum **the record states**, and
only then written -- to a temporary name, renamed on success. A file that fails
verification never exists under its final name, so nothing can load it by
accident, and a snapshot pinned to it would be a snapshot of corruption.

The record is the one recorded from the live service
(``tests/fixtures/zenodo``). No test touches the network.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest

from eagent.connectors.base import AccessPolicy, FileCache, ResponseStatus
from eagent.connectors.literature import (
    ChecksumMismatchError, ZenodoConnector, zenodo_record_payload,
)
from eagent.connectors.chemistry import LayerSemanticsError
from eagent.datalayer.probe import PROBES

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "zenodo"
RECORD = json.loads((FIXTURES / "record_7141435.json").read_text(encoding="utf-8"))

README = b"readme body for the test\n"
README_MD5 = hashlib.md5(README).hexdigest()


def record_with(files: list[dict]) -> dict:
    out = json.loads(json.dumps(RECORD))
    out["files"] = files
    return out


def file_entry(key: str, body: bytes | None, *, md5: str | None = None,
               size: int | None = None) -> dict:
    return {"key": key, "id": "x",
            "size": len(body) if size is None and body is not None else size,
            "checksum": (f"md5:{md5}" if md5 else
                         (f"md5:{hashlib.md5(body).hexdigest()}"
                          if body is not None else None)),
            "links": {"self": f"https://zenodo.org/api/records/7141435/files/{key}/content"}}


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        self.cache = FileCache(self.root / "cache")
        self.dest = self.root / "data"

    def connector(self, record: dict, body: bytes | None = README):
        connector = ZenodoConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        self.json_urls: list[str] = []
        self.byte_urls: list[str] = []

        def http_json(url, timeout=30.0):
            self.json_urls.append(url)
            return record, None

        def http_bytes(url, *, max_bytes, timeout=120.0, chunk=1 << 16):
            self.byte_urls.append(url)
            return body

        connector._http_json = http_json
        connector._http_bytes = http_bytes
        return connector


class TheRecordIsTranslated(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = zenodo_record_payload(RECORD)

    def test_the_two_dois_are_not_swapped(self) -> None:
        """A run pinned to the concept DOI is not pinned: it resolves to latest."""
        self.assertEqual(self.payload["version_doi"], "10.5281/zenodo.7141435")
        self.assertEqual(self.payload["concept_doi"], "10.5281/zenodo.7141434")

    def test_the_licence_is_the_deposit_s_own(self) -> None:
        self.assertEqual(self.payload["license"], "cc-by-4.0")

    def test_a_deposit_with_no_licence_has_none_not_an_open_one(self) -> None:
        record = json.loads(json.dumps(RECORD))
        record["metadata"].pop("license")
        self.assertIsNone(zenodo_record_payload(record)["license"])

    def test_every_file_keeps_the_checksum_zenodo_records(self) -> None:
        for entry in self.payload["files"]:
            self.assertTrue(entry["checksum"].startswith("md5:"), entry)
        self.assertEqual(len(self.payload["files"]), 14)

    def test_a_file_with_no_name_is_not_listed(self) -> None:
        record = record_with([{"size": 1, "checksum": "md5:ab"}])
        self.assertEqual(zenodo_record_payload(record)["files"], [])


class TheRecordRoundTrips(_Tmp):
    def test_the_deposit_is_built_from_a_live_shaped_record(self) -> None:
        deposit = self.connector(RECORD).deposit("7141435")
        assert deposit is not None
        self.assertEqual(deposit.deposit_id, "7141435")
        self.assertEqual(deposit.license, "cc-by-4.0")
        self.assertTrue(deposit.redistributable)
        self.assertEqual(deposit.require_version_doi(), "10.5281/zenodo.7141435")

    def test_the_url_built_is_one_a_probe_fetched(self) -> None:
        self.connector(RECORD).deposit("7141435")
        self.assertIn(self.json_urls[0], {p.url for p in PROBES
                                          if p.source_id == "zenodo"})

    def test_the_release_recorded_is_the_version_doi(self) -> None:
        connector = self.connector(RECORD)
        connector.fetch("7141435")
        envelope = self.cache.read("zenodo", connector.version,
                                   {"op": "fetch", "key": "7141435"})
        self.assertEqual(envelope["database_version"], "10.5281/zenodo.7141435")


class TheDownloadIsVerifiedBeforeItIsWritten(_Tmp):
    def deposit(self, record, body=README):
        connector = self.connector(record, body)
        deposit = connector.deposit("7141435")
        assert deposit is not None
        return connector, deposit

    def test_a_good_file_is_written_and_described(self) -> None:
        record = record_with([file_entry("README.txt", README)])
        connector, deposit = self.deposit(record)
        got = connector.download_file(deposit, "README.txt", self.dest)
        self.assertEqual(got.path.read_bytes(), README)
        self.assertEqual(got.md5, README_MD5)
        self.assertEqual(got.sha256, hashlib.sha256(README).hexdigest())
        self.assertEqual(got.license, "cc-by-4.0")
        self.assertEqual(got.version_doi, "10.5281/zenodo.7141435")

    def test_the_download_url_is_the_probed_shape(self) -> None:
        record = record_with([file_entry("README.txt", README)])
        connector, deposit = self.deposit(record)
        connector.download_file(deposit, "README.txt", self.dest)
        self.assertEqual(
            self.byte_urls[0],
            "https://zenodo.org/api/records/7141435/files/README.txt/content")
        self.assertIn(self.byte_urls[0],
                      {p.url for p in PROBES if p.source_id == "zenodo"})

    def test_a_corrupted_file_is_refused_and_never_written(self) -> None:
        record = record_with([file_entry("README.txt", README)])
        connector, deposit = self.deposit(record, body=b"tampered or truncated")
        with self.assertRaises(ChecksumMismatchError) as ctx:
            connector.download_file(deposit, "README.txt", self.dest)
        self.assertIn("was not written", str(ctx.exception))
        self.assertEqual(list(self.dest.glob("*")) if self.dest.exists() else [],
                         [], "a failed file must not exist under any name")

    def test_a_size_mismatch_is_refused(self) -> None:
        record = record_with([file_entry("README.txt", README,
                                         size=len(README) + 5)])
        connector, deposit = self.deposit(record)
        with self.assertRaises(ChecksumMismatchError):
            connector.download_file(deposit, "README.txt", self.dest)

    def test_a_file_the_deposit_does_not_list_is_refused(self) -> None:
        connector, deposit = self.deposit(record_with([file_entry("a.txt", README)]))
        with self.assertRaises(LayerSemanticsError) as ctx:
            connector.download_file(deposit, "b.txt", self.dest)
        self.assertIn("lists no file named", str(ctx.exception))
        self.assertEqual(self.byte_urls, [], "nothing may be fetched for it")

    def test_an_oversized_file_is_refused_before_a_byte_is_fetched(self) -> None:
        record = record_with([file_entry("big.zip", None, size=10**9,
                                         md5="a" * 32)])
        connector, deposit = self.deposit(record)
        with self.assertRaises(LayerSemanticsError) as ctx:
            connector.download_file(deposit, "big.zip", self.dest,
                                    max_bytes=10**6)
        self.assertIn("over the", str(ctx.exception))
        self.assertEqual(self.byte_urls, [])

    def test_a_file_with_no_checksum_is_refused_by_default(self) -> None:
        entry = file_entry("README.txt", README)
        entry["checksum"] = None
        connector, deposit = self.deposit(record_with([entry]))
        with self.assertRaises(LayerSemanticsError) as ctx:
            connector.download_file(deposit, "README.txt", self.dest)
        self.assertIn("same file next time", str(ctx.exception))

    def test_accepting_no_checksum_is_an_explicit_choice(self) -> None:
        entry = file_entry("README.txt", README)
        entry["checksum"] = None
        connector, deposit = self.deposit(record_with([entry]))
        got = connector.download_file(deposit, "README.txt", self.dest,
                                      require_checksum=False)
        self.assertTrue(got.path.exists())

    def test_a_non_md5_checksum_is_not_silently_ignored(self) -> None:
        entry = file_entry("README.txt", README)
        entry["checksum"] = "sha1:" + "0" * 40
        connector, deposit = self.deposit(record_with([entry]))
        with self.assertRaises(LayerSemanticsError):
            connector.download_file(deposit, "README.txt", self.dest)

    def test_a_listed_file_the_service_will_not_serve_is_refused(self) -> None:
        record = record_with([file_entry("README.txt", README)])
        connector, deposit = self.deposit(record, body=None)
        with self.assertRaises(LayerSemanticsError) as ctx:
            connector.download_file(deposit, "README.txt", self.dest)
        self.assertIn("returned nothing", str(ctx.exception))

    def test_a_name_with_a_path_cannot_escape_the_destination(self) -> None:
        record = record_with([file_entry("../../escape.txt", README)])
        connector, deposit = self.deposit(record)
        got = connector.download_file(deposit, "../../escape.txt", self.dest)
        self.assertEqual(got.path.parent.resolve(), self.dest.resolve())
        self.assertFalse((self.root / "escape.txt").exists())

    def test_a_bare_dot_dot_name_is_refused(self) -> None:
        record = record_with([file_entry("..", README)])
        connector, deposit = self.deposit(record)
        with self.assertRaises(LayerSemanticsError) as ctx:
            connector.download_file(deposit, "..", self.dest)
        self.assertIn("outside", str(ctx.exception))

    def test_no_temporary_file_is_left_behind(self) -> None:
        record = record_with([file_entry("README.txt", README)])
        connector, deposit = self.deposit(record)
        connector.download_file(deposit, "README.txt", self.dest)
        self.assertEqual([p.name for p in self.dest.iterdir()], ["README.txt"])


class TheByteDownloadHasAHardCap(_Tmp):
    """The cap is enforced while reading, not after."""

    def run_http(self, chunks: list[bytes], *, declared: str | None, cap: int):
        import urllib.request
        from unittest import mock

        class Handle:
            def __init__(self):
                self.headers = {"Content-Length": declared} if declared else {}
                self._chunks = list(chunks)
            def read(self, n=-1):
                return self._chunks.pop(0) if self._chunks else b""
            def __enter__(self): return self
            def __exit__(self, *a): return False

        connector = ZenodoConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        with mock.patch.object(urllib.request, "urlopen",
                               lambda *a, **k: Handle()):
            return connector._http_bytes("https://zenodo.org/x", max_bytes=cap)

    def test_a_declared_length_over_the_cap_is_refused_unread(self) -> None:
        from eagent.connectors.base import RemoteCallFailedError
        with self.assertRaises(RemoteCallFailedError) as ctx:
            self.run_http([b"x" * 10], declared="9999", cap=100)
        self.assertIn("nothing was read", str(ctx.exception))

    def test_a_lying_length_header_cannot_get_past_the_cap(self) -> None:
        from eagent.connectors.base import RemoteCallFailedError
        with self.assertRaises(RemoteCallFailedError) as ctx:
            self.run_http([b"x" * 60, b"x" * 60], declared="10", cap=100)
        self.assertIn("while being read", str(ctx.exception))

    def test_a_body_inside_the_cap_is_returned_whole(self) -> None:
        self.assertEqual(self.run_http([b"ab", b"cd"], declared="4", cap=100),
                         b"abcd")

    def test_the_network_policy_is_still_enforced(self) -> None:
        from eagent.connectors.base import NetworkDisabledError
        connector = ZenodoConnector(cache=self.cache)           # offline default
        with self.assertRaises(NetworkDisabledError):
            connector._http_bytes("https://zenodo.org/x", max_bytes=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
