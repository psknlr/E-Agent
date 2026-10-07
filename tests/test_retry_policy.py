"""When a failed request is tried again, and when it must not be.

A live download of the SDR dataset failed on its second file with
``SSL: UNEXPECTED_EOF_WHILE_READING`` after the first had succeeded: a dropped
handshake, not a verdict about the file. The connector reported it correctly
-- as a failure, not as "the service returned nothing" -- but a dropped
connection is the commonest thing a public service does and a run that dies on
it is not a run anybody can use.

The rule these tests pin down is deliberately narrow, because an over-eager
retry is its own bug. Only a *transient* failure is retried (a dropped
connection, a timeout, a 429 or a 5xx), a bounded number of times; a 4xx, a body
over its cap and a response that is not JSON fail at once, because the same
request will fail the same way and retrying would only delay saying so.
"""

from __future__ import annotations

import io
import pathlib
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

from eagent.connectors.base import (
    AccessPolicy, FileCache, RemoteCallFailedError, ResponseStatus,
)
from eagent.connectors.literature import ZenodoConnector


class Handle:
    def __init__(self, body: bytes = b"{}", headers=None):
        self._body = io.BytesIO(body)
        self.headers = headers or {}
    def read(self, n=-1): return self._body.read(n)
    def __enter__(self): return self
    def __exit__(self, *a): return False


def http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://zenodo.org/x", code, "msg", {},
                                  io.BytesIO(body))


class _Net(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = FileCache(pathlib.Path(self._tmp.name))
        self.connector = ZenodoConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        self.slept: list[float] = []
        self.connector._sleep = self.slept.append          # no real waiting
        self.calls = 0

    def serve(self, *outcomes):
        """Each call returns or raises the next outcome; the last repeats."""
        outcomes = list(outcomes)

        def urlopen(request, timeout=None):
            self.calls += 1
            outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return mock.patch.object(urllib.request, "urlopen", urlopen)


class WhatIsRetried(_Net):
    def test_a_dropped_tls_handshake_is_retried_and_then_succeeds(self) -> None:
        eof = urllib.error.URLError("[SSL: UNEXPECTED_EOF_WHILE_READING]")
        with self.serve(eof, Handle(b'{"ok": 1}')):
            body, _ = self.connector._http_json("https://zenodo.org/x")
        self.assertEqual(body, {"ok": 1})
        self.assertEqual(self.calls, 2)
        self.assertEqual(self.slept, [1.0])

    def test_a_timeout_is_retried(self) -> None:
        with self.serve(TimeoutError("timed out"), Handle(b"{}")):
            self.connector._http_json("https://zenodo.org/x")
        self.assertEqual(self.calls, 2)

    def test_a_503_is_retried(self) -> None:
        with self.serve(http_error(503), Handle(b"{}")):
            self.connector._http_json("https://zenodo.org/x")
        self.assertEqual(self.calls, 2)

    def test_a_429_is_retried(self) -> None:
        with self.serve(http_error(429), Handle(b"{}")):
            self.connector._http_json("https://zenodo.org/x")
        self.assertEqual(self.calls, 2)

    def test_the_delays_back_off(self) -> None:
        eof = urllib.error.URLError("reset")
        with self.serve(eof, eof, Handle(b"{}")):
            self.connector._http_json("https://zenodo.org/x")
        self.assertEqual(self.slept, [1.0, 3.0])

    def test_the_bytes_path_retries_too(self) -> None:
        with self.serve(urllib.error.URLError("reset"), Handle(b"abc")):
            self.assertEqual(
                self.connector._http_bytes("https://zenodo.org/x", max_bytes=100),
                b"abc")


class WhatIsNotRetried(_Net):
    def test_a_404_is_an_answer_and_is_never_retried(self) -> None:
        with self.serve(http_error(404)):
            self.assertEqual(self.connector._http_json("https://zenodo.org/x"),
                             (None, None))
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.slept, [])

    def test_a_400_fails_at_once(self) -> None:
        """The same request will fail the same way."""
        with self.serve(http_error(400, b"bad accession")):
            with self.assertRaises(RemoteCallFailedError) as ctx:
                self.connector._http_json("https://zenodo.org/x")
        self.assertEqual(self.calls, 1)
        self.assertFalse(ctx.exception.transient)
        self.assertIn("bad accession", str(ctx.exception))

    def test_a_403_fails_at_once(self) -> None:
        with self.serve(http_error(403)):
            with self.assertRaises(RemoteCallFailedError):
                self.connector._http_json("https://zenodo.org/x")
        self.assertEqual(self.calls, 1)

    def test_a_body_that_is_not_json_is_not_retried(self) -> None:
        with self.serve(Handle(b"<html>login</html>")):
            with self.assertRaises(RemoteCallFailedError) as ctx:
                self.connector._http_json("https://zenodo.org/x")
        self.assertEqual(self.calls, 1)
        self.assertIn("not JSON", str(ctx.exception))

    def test_a_body_over_its_cap_is_not_retried(self) -> None:
        with self.serve(Handle(b"x" * 500)):
            with self.assertRaises(RemoteCallFailedError) as ctx:
                self.connector._http_bytes("https://zenodo.org/x", max_bytes=100)
        self.assertEqual(self.calls, 1)
        self.assertFalse(ctx.exception.transient)

    def test_the_network_policy_is_checked_before_any_attempt(self) -> None:
        from eagent.connectors.base import NetworkDisabledError
        offline = ZenodoConnector(cache=self.cache)
        with self.serve(Handle(b"{}")):
            with self.assertRaises(NetworkDisabledError):
                offline._http_json("https://zenodo.org/x")
        self.assertEqual(self.calls, 0)


class WhenItKeepsFailing(_Net):
    def test_it_stops_after_the_configured_retries(self) -> None:
        eof = urllib.error.URLError("reset")
        with self.serve(eof):
            with self.assertRaises(RemoteCallFailedError) as ctx:
                self.connector._http_json("https://zenodo.org/x")
        self.assertEqual(self.calls, 1 + len(ZenodoConnector.retry_delays_s))
        self.assertEqual(ctx.exception.attempts, self.calls)
        self.assertTrue(ctx.exception.transient)

    def test_the_failure_says_how_many_times_it_tried(self) -> None:
        with self.serve(urllib.error.URLError("reset")):
            with self.assertRaises(RemoteCallFailedError) as ctx:
                self.connector._http_json("https://zenodo.org/x")
        self.assertIn("after 3 attempts", str(ctx.exception))

    def test_a_persistent_outage_is_an_error_not_a_miss(self) -> None:
        with self.serve(http_error(503)):
            response = self.connector.fetch("7141435")
        self.assertIs(response.status, ResponseStatus.ERROR)
        self.assertIn("after 3 attempts", response.miss_reason)

    def test_nothing_is_cached_from_a_failure(self) -> None:
        with self.serve(http_error(503)):
            self.connector.fetch("7141435")
        good = {"id": 7141435, "doi": "10.5281/zenodo.7141435",
                "metadata": {"title": "t"}, "files": []}
        import json
        with self.serve(Handle(json.dumps(good).encode())):
            self.assertIs(self.connector.fetch("7141435").status,
                          ResponseStatus.FETCHED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
