"""Calling a registered source to find out whether the route actually works.

WHY THIS IS NOT ONE LINE OF CURL
================================
The registry records what each resource is *documented* to offer. That is the
right thing to record and it is not a route: a documented REST API can have
moved, be behind a login, or answer with a cheerful HTML page. The registry
used to refuse ``connectivity_verified=True`` outright for exactly that
reason -- nobody had called anything, so the honest value was false, and a
boolean asserted without a call would have told a planner a route works.

This module is what makes the flag earnable. A probe is a specific request
with a declared expectation, and it passes only when the response *contains
what that record must contain*:

* SABIO-RK's documented REST path, called from this environment, answers 302
  and then 200 -- on the provider's own 404 page. A status check passes it.
* A login wall returns 200 with a form in it.
* A cached CDN error page returns 200 with nothing in it at all.

So every probe declares ``markers``: strings only the real record carries.
``P07846`` in a UniProt entry, ``1CDO`` in an RCSB entry. A probe with no
markers is refused at construction, because it would be the status check
again under a longer name.

WHAT A PASS DOES AND DOES NOT ESTABLISH
=======================================
A pass says: *this capability, at this URL, answered correctly at this time,
from this environment.* It says nothing about the other capabilities, nothing
about rate limits or bulk access, and nothing about the licence. The check is
recorded per capability for that reason, and
:attr:`~eagent.datalayer.registry.DataSource.verified_capabilities` lists only
the ones a call actually exercised.

NETWORK ACCESS IS THE CALLER'S DECISION
=======================================
Nothing here runs without being handed a fetcher. The default one uses the
standard library and honours the run policy's ``allow_network``; a test
passes its own. A module that reached the network as a side effect of being
imported, or of a registry load, would make every offline run depend on the
weather.
"""

from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..errors import EAgentError
from ..provenance import utc_now
from .registry import (
    CAPABILITY_NAMES, OBSERVED_CONNECTIVITY_FILE, ConnectivityCheck,
    default_datasource_dir,
)

__all__ = [
    "ProbeError",
    "DEFAULT_TIMEOUT_S",
    "USER_AGENT",
    "Response",
    "Fetcher",
    "urllib_fetcher",
    "CapabilityProbe",
    "run_probe",
    "run_probes",
    "PROBES",
    "probes_for",
    "CONNECTIVITY_FILE",
    "write_checks",
    "load_observed",
]


class ProbeError(EAgentError):
    """A probe is malformed, or was asked to run without a way to call out."""


#: Seconds before a probe gives up. A source that needs longer than this to
#: answer one small request is not a route a run can depend on, and the
#: failure is recorded rather than waited out.
DEFAULT_TIMEOUT_S: float = 30.0

#: Sent on every request, by the probes AND by the connectors' clients -- one
#: constant, imported by both, because a probe that identifies itself
#: differently from the client it is meant to vouch for verifies a request
#: nobody makes. That is not hypothetical: Rhea's CDN answers the default
#: ``Python-urllib`` agent with 403 and this one with 200, so a probe sending
#: one agent while the client sent the other passed while the client failed.
#:
#: A shared resource is also entitled to know who is calling it and where to
#: complain, and an unidentified client is the first thing a provider blocks.
USER_AGENT: str = "E-Agent (enzyme function mining research agent; connectors and probes)"


@dataclass(frozen=True)
class Response:
    """What a fetcher returns. Deliberately smaller than an HTTP library's."""

    status: int
    body: str
    elapsed_ms: int


#: How a probe reaches the network. A protocol so a test can supply one.
Fetcher = Callable[[str, float], Response]


def urllib_fetcher(url: str, timeout: float = DEFAULT_TIMEOUT_S) -> Response:
    """The default fetcher: standard library, no redirects beyond urllib's own.

    Errors are turned into a :class:`Response` rather than raised, because a
    404 and a DNS failure are both results a probe must record. Only the
    status is interpreted here; whether the *body* is the record is the
    probe's question.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "*/*"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            body = handle.read().decode("utf-8", errors="replace")
            status = int(getattr(handle, "status", 200) or 200)
    except urllib.error.HTTPError as exc:                # a real HTTP answer
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        status = int(exc.code)
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return Response(status=0, body=f"{type(exc).__name__}: {exc}",
                        elapsed_ms=int((time.monotonic() - started) * 1000))
    return Response(status=status, body=body,
                    elapsed_ms=int((time.monotonic() - started) * 1000))


@dataclass(frozen=True)
class CapabilityProbe:
    """One request that would demonstrate one capability of one source.

    ``markers`` are what the response must contain. They are chosen to be
    things only the intended record carries, not things any page from that
    host would: ``"uniprot"`` appears on UniProt's error pages too, while the
    accession asked for does not.
    """

    source_id: str
    capability: str
    url: str
    markers: tuple[str, ...]
    description: str
    #: The provider's own API documentation for this route. Required: the
    #: registry already refuses an endpoint with no citation, because an
    #: endpoint the code will call has to be traceable to something a person
    #: can read. A probe that could set an endpoint without one would route
    #: around that rule.
    documentation: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.capability not in CAPABILITY_NAMES:
            raise ProbeError(
                f"{self.source_id}: unknown capability '{self.capability}'; "
                f"known: {', '.join(CAPABILITY_NAMES)}")
        if not self.markers:
            raise ProbeError(
                f"{self.source_id}/{self.capability}: a probe with no markers "
                f"is a status check. A 200 comes back from a redirect to a 404 "
                f"page and from a login wall; name what only the real record "
                f"contains")
        if not self.url.lower().startswith("https://"):
            raise ProbeError(
                f"{self.source_id}/{self.capability}: probe URL must be https, "
                f"got {self.url!r}")
        if not self.documentation.lower().startswith("https://"):
            raise ProbeError(
                f"{self.source_id}/{self.capability}: no documentation URL. "
                f"A passing probe sets the registry's endpoint, and the "
                f"registry refuses an endpoint that cannot be traced to "
                f"documentation somebody can read")

    @property
    def base(self) -> str:
        """The scheme and host, which is what a registry endpoint records."""
        rest = self.url.split("://", 1)[1]
        return self.url.split("://", 1)[0] + "://" + rest.split("/", 1)[0]


def run_probe(probe: CapabilityProbe, *, fetcher: Fetcher | None = None,
              timeout: float = DEFAULT_TIMEOUT_S,
              checked_by: str = "eagent.datalayer.probe") -> ConnectivityCheck:
    """Run one probe and return the record of what happened, pass or fail.

    Never raises on a failed call. A source being unreachable is a fact about
    the run that belongs in the registry next to the sources that answered,
    not an exception that stops the sweep at the first firewall.
    """
    fetch = fetcher or urllib_fetcher
    response = fetch(probe.url, timeout)
    digest = hashlib.sha256(response.body.encode("utf-8")).hexdigest()
    missing = [m for m in probe.markers if m not in response.body]
    # The documentation is fetched in the same sweep. A documentation URL
    # nobody opened is an assertion, and it is the thing the registry demands
    # before it will hold an endpoint -- so the probe has to have opened it,
    # not merely have been given it.
    docs = fetch(probe.documentation, timeout)
    if response.status == 0:
        failure = f"the request did not complete: {response.body[:200]}"
    elif response.status >= 400:
        failure = f"HTTP {response.status}"
    elif missing:
        failure = (
            f"HTTP {response.status}, but the response does not contain "
            f"{', '.join(repr(m) for m in missing)}. Something answered; it is "
            f"not the record that was asked for")
    elif docs.status == 0 or docs.status >= 400:
        failure = (
            f"the route answered, but its documentation at "
            f"{probe.documentation} did not (HTTP {docs.status or 'no reply'}). "
            f"An endpoint the code will call has to be traceable to a page a "
            f"person can read, so this is not recorded as verified")
    else:
        failure = ""
    return ConnectivityCheck(
        capability=probe.capability, url=probe.url, checked_at=utc_now(),
        checked_by=checked_by, ok=not failure, status_code=response.status or None,
        markers=list(probe.markers), response_sha256=f"sha256:{digest}",
        response_bytes=len(response.body.encode("utf-8")),
        elapsed_ms=response.elapsed_ms,
        documentation_url=probe.documentation,
        documentation_status=docs.status or None,
        failure=failure, note=probe.note)


def run_probes(probes: Iterable[CapabilityProbe], *,
               fetcher: Fetcher | None = None,
               timeout: float = DEFAULT_TIMEOUT_S,
               checked_by: str = "eagent.datalayer.probe",
               ) -> dict[str, list[ConnectivityCheck]]:
    """Run several probes, grouped by source id, in the order given."""
    out: dict[str, list[ConnectivityCheck]] = {}
    for probe in probes:
        out.setdefault(probe.source_id, []).append(
            run_probe(probe, fetcher=fetcher, timeout=timeout,
                      checked_by=checked_by))
    return out


# ==========================================================================
# The probes this project ships
# ==========================================================================
#
# One source per layer, chosen because each is a documented public API with no
# account: a run can be reproduced by somebody else without being given
# credentials. Each probe asks for one well-known record and names markers that
# only that record carries.
#
# These are not an endorsement of the resources' *content*. UniProt's
# annotation is still annotation, and the evidence ceiling in the registry is
# unchanged by the route being reachable.

PROBES: tuple[CapabilityProbe, ...] = (
    CapabilityProbe(
        source_id="uniprotkb",
        capability="exact_record_fetch",
        url=("https://rest.uniprot.org/uniprotkb/P07846.json?fields="
             "accession,id,protein_name,organism_name,reviewed,ec,"
             "cc_catalytic_activity,cc_cofactor,sequence,xref_pdb"),
        markers=("P07846", "primaryAccession", "uniProtkbId"),
        description=("fetch one reviewed UniProtKB entry by accession, with "
                     "exactly the field list the connector's client requests: "
                     "the list is the parser, so a probe that asked for fewer "
                     "fields would verify a request nobody makes"),
        documentation="https://www.uniprot.org/help/api",
        note=("P07846 is an alcohol dehydrogenase, chosen because it is a "
              "reviewed entry in the mechanistic neighbourhood of this "
              "project's pilot task"),
    ),
    CapabilityProbe(
        source_id="uniprotkb",
        capability="version_information",
        url="https://rest.uniprot.org/uniprotkb/search?query=accession:P07846&size=1",
        markers=("P07846", "results"),
        description=("a search response, from which the release can be read "
                     "out of the response headers the API documents"),
        documentation="https://www.uniprot.org/help/api",
    ),
    CapabilityProbe(
        source_id="rcsb_pdb",
        capability="exact_record_fetch",
        url="https://data.rcsb.org/rest/v1/core/entry/1CDO",
        markers=("1CDO", "rcsb_entry_info"),
        description="fetch one PDB entry's core metadata by its four-character id",
        documentation="https://www.rcsb.org/docs/programmatic-access/web-apis-overview",
        note=("1CDO is a horse liver alcohol dehydrogenase structure: an "
              "NAD-dependent oxidoreductase, so the record exercises the "
              "fields this project reads"),
    ),
    CapabilityProbe(
        source_id="rcsb_pdb",
        capability="exact_record_fetch",
        url="https://data.rcsb.org/rest/v1/core/polymer_entity/1CDO/1",
        markers=("1CDO_1", "entity_poly"),
        description=("fetch one polymer entity of an entry: the sequence and "
                     "the chains it occupies"),
        documentation="https://www.rcsb.org/docs/programmatic-access/web-apis-overview",
        note=("a client that makes this request is verified only if this "
              "request itself was probed, not merely the entry one"),
    ),
    CapabilityProbe(
        source_id="rcsb_pdb",
        capability="exact_record_fetch",
        url="https://data.rcsb.org/rest/v1/core/nonpolymer_entity/1CDO/3",
        markers=("1CDO_3", "pdbx_entity_nonpoly"),
        description=("fetch one non-polymer entity of an entry: a ligand or "
                     "cofactor component and the chains it sits on"),
        documentation="https://www.rcsb.org/docs/programmatic-access/web-apis-overview",
        note="1CDO entity 3 is NAD, the cofactor of this alcohol dehydrogenase",
    ),
    CapabilityProbe(
        source_id="rhea",
        capability="keyword_query",
        url=("https://www.rhea-db.org/rhea?query=ec:1.1.1.1"
             "&columns=rhea-id,equation,ec,chebi-id&format=tsv&limit=100"),
        markers=("Reaction identifier\tEquation\tEC number\tChEBI identifier",
                 "EC:1.1.1.1"),
        description=("an EC-number query returning the pinned column set; the "
                     "header is part of the marker because the column list is "
                     "the parser"),
        documentation="https://www.rhea-db.org/help/rest-api",
        note=("EC 1.1.1.1 is alcohol dehydrogenase, the mechanistic "
              "neighbourhood of this project's pilot task"),
    ),
    CapabilityProbe(
        source_id="rhea",
        capability="keyword_query",
        url=("https://www.rhea-db.org/rhea?query=RHEA:10740"
             "&columns=rhea-id,equation,ec,chebi-id&format=tsv&limit=100"),
        markers=("RHEA:10740", "secondary alcohol"),
        description="one reaction by identifier, through the same query route",
        documentation="https://www.rhea-db.org/help/rest-api",
        note=("a secondary alcohol to a ketone with NAD(+): the reaction "
              "class of the pilot task, read in the oxidation direction"),
    ),
)


def probes_for(source_id: str) -> tuple[CapabilityProbe, ...]:
    """Every shipped probe for one source."""
    return tuple(p for p in PROBES if p.source_id == source_id)


# ==========================================================================
# Recording a pass
# ==========================================================================

#: File the sweep writes, and the loader merges. Separate from the curated
#: source files on purpose -- see :func:`write_checks`. Defined in the
#: registry so the loader and the writer cannot disagree about the name.
CONNECTIVITY_FILE: str = OBSERVED_CONNECTIVITY_FILE

_FILE_HEADER = """\
# MACHINE-WRITTEN. Do not hand-edit; `eagent sources verify --write` rewrites
# this file in full.
#
# WHY THIS IS NOT IN THE SOURCE FILES
# ===================================
# The curated entries in configs/datasources/*.yaml are judgements: what a
# resource is good for, what it may never claim, its licence, its evidence
# ceiling. They change when somebody decides something.
#
# What is below is an observation: a URL answered, at a moment, from one
# environment. It goes stale on its own, without anybody deciding anything.
# Merging the two would mean a nightly sweep rewriting a file full of
# hand-written reasoning, and a reader being unable to tell which lines were
# curated and which were measured.
#
# So the loader merges this onto the matching sources, and a source is
# `connectivity_verified` exactly when a check here says so. Delete this file
# and every source is unverified again, which is the correct state for an
# environment nobody has run the sweep in.
#
# A pass says: this capability, at this URL, answered correctly at this time,
# from this environment. It is not a licence, not permission to redistribute,
# and not a statement about any other capability.
"""


def write_checks(
    results: Mapping[str, Sequence[ConnectivityCheck]],
    directory: "Path | None" = None,
) -> Path:
    """Write the **passing** checks to the observation file. Returns its path.

    Three rules, each about not overstating what a call showed:

    * **only passes are written.** A source that did not answer today keeps
      whatever it had: "unreachable from this container at this moment" is a
      different statement from "this route does not work", and recording the
      second from the first would retire a usable resource;
    * **the endpoint recorded is the host, not the probe URL.** The probe asked
      for one record; the endpoint is the service. Writing the full probe URL
      would leave the registry pointing at a single accession;
    * **nothing curated is touched.** Not the licence, not the evidence
      ceiling, not ``needs_legal_review``. Reachability is not permission, and
      a sweep that quietly cleared a legal-review flag would be the worst edit
      this tool could make -- which is most of why it writes its own file.
    """
    import yaml

    base = Path(directory) if directory is not None else default_datasource_dir()
    passing = {sid: [c for c in checks if c.ok]
               for sid, checks in results.items()}
    passing = {sid: checks for sid, checks in passing.items() if checks}

    document: dict[str, Any] = {"observed_connectivity": []}
    for sid in sorted(passing):
        checks = passing[sid]
        # The service, not the record. Derived from the URL that actually
        # answered rather than from the shipped probe table, so a caller's own
        # probe records an endpoint too -- and so the endpoint can never
        # disagree with the call behind it.
        endpoint = _service_base(checks[0].url)
        # Not a bare URL: a citation records that the page was opened, and
        # when. A URL on its own is an assertion that documentation exists.
        citations = sorted({
            f"{c.documentation_url} (fetched HTTP {c.documentation_status} "
            f"on {c.checked_at})"
            for c in checks if c.documentation_url})
        document["observed_connectivity"].append({
            "id": sid,
            "endpoint": endpoint,
            "citations": citations,
            "connectivity_verified": True,
            "connectivity_checks": [
                json.loads(c.model_dump_json()) for c in checks],
        })
    path = base / CONNECTIVITY_FILE
    path.write_text(
        _FILE_HEADER + "\n" + yaml.safe_dump(
            document, sort_keys=False, allow_unicode=True,
            default_flow_style=False, width=88),
        encoding="utf-8")
    return path


def load_observed(directory: "Path | None" = None) -> dict[str, dict[str, Any]]:
    """Read the observation file through the registry's own loader."""
    from .registry import _load_observed_connectivity

    base = Path(directory) if directory is not None else default_datasource_dir()
    return _load_observed_connectivity(base / CONNECTIVITY_FILE)


def _service_base(url: str) -> str:
    """Scheme and host of a URL: what a registry endpoint records.

    The probe asked for one record. Recording the whole probe URL would leave
    the registry pointing at a single accession, which the next caller would
    then build a path on top of.
    """
    scheme, rest = url.split("://", 1)
    return f"{scheme}://{rest.split('/', 1)[0]}"
