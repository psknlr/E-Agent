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
import re
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
    "ROUTE_PROVENANCE_GRADES",
    "RouteProvenance",
    "CapabilityProbe",
    "run_probe",
    "run_probes",
    "PROBES",
    "probes_for",
    "never_called_probes",
    "probe_hosts",
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


# ==========================================================================
# Where a probe's URL came from
# ==========================================================================
#
# The registry refuses an endpoint nobody has called, and keeps a remembered
# URL in ``curation_notes`` as a hint instead. A probe is where such a hint
# becomes testable: it is a hypothesis, and the sweep is the experiment.
#
# That only works if the hypothesis's pedigree travels with it. A sweep that
# reports ten passes and twelve failures is unreadable without it, because two
# very different things look identical in the output -- a resource that is
# unreachable today, and a candidate route that was never right. The first is
# news about the network. The second is news about this file.

#: How a probe's URL was obtained, strongest first. None of these grades says
#: the route *works*; they say how close the author got to the provider's own
#: words before the sweep was asked to find out.
ROUTE_PROVENANCE_GRADES: tuple[str, ...] = (
    "called_from_here",         # a sweep in this repo recorded a pass at it
    "provider_documentation",   # stated on the provider's own documentation
    "live_response_seen",       # a response from this route shape was seen
    "third_party_client",       # a client library's source, not the provider
    "search_summary",           # a search engine's summary and nothing more
)


@dataclass(frozen=True)
class RouteProvenance:
    """Where one probe's URL came from, and whether anybody has called it.

    ``detail`` is prose and is the part that matters: the grade is a label for
    sorting output, and a label cannot say *which* page, *which* example id, or
    which part of the URL the author filled in themselves. A probe whose route
    came from documentation but whose record id is the author's own choice is a
    normal and useful thing to ship -- it just has to say so, because when it
    404s the fix is the id, not the host.
    """

    #: One of :data:`ROUTE_PROVENANCE_GRADES`.
    grade: str
    #: What was found, where, and what the author supplied instead.
    detail: str
    #: ``YYYY-MM-DD``. A route's pedigree goes stale like the route does: a
    #: page read two years ago is weaker evidence than the same page today.
    recorded_at: str
    #: The search that produced it, where that is what happened. Required for
    #: ``search_summary``, because a summary with no query behind it cannot be
    #: re-run by the next person.
    query: str = ""

    def __post_init__(self) -> None:
        if self.grade not in ROUTE_PROVENANCE_GRADES:
            raise ProbeError(
                f"unknown route provenance grade {self.grade!r}; known: "
                f"{', '.join(ROUTE_PROVENANCE_GRADES)}")
        if len(self.detail.strip()) < 24:
            raise ProbeError(
                f"route provenance {self.grade!r} has no usable detail. The "
                f"grade is a label; the detail is what tells the next person "
                f"which page was read and which part of the URL was guessed")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.recorded_at):
            raise ProbeError(
                f"route provenance recorded_at must be YYYY-MM-DD, got "
                f"{self.recorded_at!r}")
        if self.grade == "search_summary" and not self.query.strip():
            raise ProbeError(
                "a 'search_summary' provenance must carry the query: a "
                "summary nobody can re-run is not evidence anybody can check")

    @property
    def ever_called(self) -> bool:
        """Whether a sweep in this repo has reached this URL.

        False is the honest state for a route assembled from documentation and
        never dialled, and it is what separates "this went down" from "this was
        never right" in a sweep's output.
        """
        return self.grade == "called_from_here"

    def to_dict(self) -> dict[str, Any]:
        """The record, for printing or for a report."""
        out: dict[str, Any] = {"grade": self.grade, "detail": self.detail,
                               "recorded_at": self.recorded_at,
                               "ever_called": self.ever_called}
        if self.query:
            out["query"] = self.query
        return out


#: The four sources whose routes a sweep in this project actually reached, on
#: 2026-10-08, recorded in ``connectivity.observed.yaml``. Shared by their
#: probes because the pedigree is the same fact about the same sweep.
_CALLED = RouteProvenance(
    grade="called_from_here",
    detail=("a sweep from a container in this project reached this route and "
            "its documentation, and the pass is recorded in "
            "configs/datasources/connectivity.observed.yaml"),
    recorded_at="2026-10-08",
)

#: Documentation markers for a page whose real text could not be read from
#: here. On 2026-10-25 NCBI Bookshelf NBK25501 and PubChem's docs/pug-rest
#: answered a container in this project with a "Checking your browser"
#: reCAPTCHA page, so there was nothing to choose a marker from, and a guessed
#: one would fail wherever the page is served properly. Status alone, by
#: necessity rather than choice -- and recorded as such, because a check
#: carrying an empty list says its documentation was taken on status.
_STATUS_ALONE: tuple[str, ...] = ()


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
    #: Where this URL came from. Required, with no default, because a probe
    #: that could be added without one would quietly re-admit the thing the
    #: registry nulled five endpoints to get rid of: a URL in the code that
    #: nobody can trace. The sweep still decides whether it works.
    provenance: RouteProvenance
    #: The provider's own API documentation for this route. Required: the
    #: registry already refuses an endpoint with no citation, because an
    #: endpoint the code will call has to be traceable to something a person
    #: can read. A probe that could set an endpoint without one would route
    #: around that rule.
    documentation: str = ""
    #: Strings that must appear in the documentation page's body, applying to
    #: the documentation fetch the rule this module already applies to the
    #: route. Empty keeps the older, status-only behaviour.
    #:
    #: It exists because the status-only check was found passing on pages that
    #: are not documentation, by a sweep run on an open network on 2026-10-25:
    #: NCBI Bookshelf NBK25501 and PubChem's ``docs/pug-rest`` both answered
    #: 200 with a "Checking your browser" reCAPTCHA page, and
    #: ``www.uniprot.org/help/api`` answered 200 with a single-page-app shell
    #: byte-identical to the shell served for a help page that does not exist.
    #: That is the same mistake one level up from the one this module refuses a
    #: markerless probe for: a 200 is not a page, any more than it is a record.
    #:
    #: Choosing a marker means knowing what the real page contains, so the
    #: shipped ones were read off the pages that answered a sweep on
    #: 2026-10-25, from a container that could reach them: each names the
    #: route or operation its page documents, which no challenge page, app
    #: shell or 404 page carries. Three probes stay empty, as
    #: :data:`_STATUS_ALONE` explains, because their page could not be read.
    #: Where the page a person reads builds its text in the browser, the
    #: documentation is the provider's OpenAPI document for the same API,
    #: which is served whole.
    documentation_markers: tuple[str, ...] = ()
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
    elif missing_docs := [m for m in probe.documentation_markers
                          if m not in docs.body]:
        failure = (
            f"the route answered, and its documentation at "
            f"{probe.documentation} answered HTTP {docs.status}, but that "
            f"response does not contain "
            f"{', '.join(repr(m) for m in missing_docs)}. Something is served "
            f"there; it is not the documentation. A reCAPTCHA interstitial and "
            f"a single-page-app shell both answer 200")
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
        documentation_markers=list(probe.documentation_markers),
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
        provenance=_CALLED,
        # The service's OpenAPI document, not www.uniprot.org/help/api:
        # that page is a single-page-app shell, byte-identical to the one
        # served for a help page that does not exist.
        documentation="https://rest.uniprot.org/uniprotkb/api/docs",
        documentation_markers=("/uniprotkb/{accession}",),
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
        provenance=_CALLED,
        documentation="https://rest.uniprot.org/uniprotkb/api/docs",
        documentation_markers=("/uniprotkb/search",),
    ),
    CapabilityProbe(
        source_id="rcsb_pdb",
        capability="exact_record_fetch",
        url="https://data.rcsb.org/rest/v1/core/entry/1CDO",
        markers=("1CDO", "rcsb_entry_info"),
        description="fetch one PDB entry's core metadata by its four-character id",
        provenance=_CALLED,
        documentation="https://www.rcsb.org/docs/programmatic-access/web-apis-overview",
        documentation_markers=("Data API", "data.rcsb.org/redoc"),
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
        provenance=_CALLED,
        documentation="https://www.rcsb.org/docs/programmatic-access/web-apis-overview",
        documentation_markers=("Data API", "data.rcsb.org/redoc"),
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
        provenance=_CALLED,
        documentation="https://www.rcsb.org/docs/programmatic-access/web-apis-overview",
        documentation_markers=("Data API", "data.rcsb.org/redoc"),
        note="1CDO entity 3 is NAD, the cofactor of this alcohol dehydrogenase",
    ),
    CapabilityProbe(
        source_id="rcsb_pdb",
        capability="exact_record_fetch",
        url="https://files.rcsb.org/download/1IPF.cif",
        markers=("data_1IPF", "_entry.id", "_atom_site.Cartn_x"),
        description=("fetch one entry's asymmetric-unit mmCIF from the RCSB "
                     "file host, in exactly the shape "
                     "RCSBPDBConnector.download_structure requests "
                     "(RCSB_STRUCTURE_FILE_PATH on the host this probe passed "
                     "for): the file host is a different "
                     "service from the data API above, so it needs its own "
                     "probe, and the connector refuses to download until this "
                     "one has passed"),
        provenance=_CALLED,
        documentation=("https://www.rcsb.org/docs/programmatic-access/"
                       "file-download-services"),
        documentation_markers=("files.rcsb.org/download",),
        note=("1IPF is tropinone reductase II with NADPH and tropinone, the "
              "one pre-reaction ternary SDR complex in the KRED calibration "
              "reference set; a 450 KB file, so the probe is cheap"),
    ),
    CapabilityProbe(
        source_id="zenodo",
        capability="exact_record_fetch",
        url="https://zenodo.org/api/records/7141435",
        markers=("10.5281/zenodo.7141435", "SDR_sequences.fasta", "cc-by-4.0"),
        description=("one deposit's metadata: DOIs, licence, and every file "
                     "with the checksum Zenodo records for it"),
        provenance=_CALLED,
        documentation="https://developers.zenodo.org/",
        documentation_markers=("/api/records",),
        note=("record 7141435 is the SDR substrate-classification dataset "
              "(Jinich and Rappoport, 2022), the first dataset this project "
              "ingests; chosen because it is small, openly licensed and "
              "about the enzyme family of the pilot task"),
    ),
    CapabilityProbe(
        source_id="zenodo",
        capability="exact_record_fetch",
        url="https://zenodo.org/api/records/7141435/files/README.txt/content",
        markers=("Enzyme Substrate Classification Dataset", "Jinich"),
        description=("one file of a deposit, by name: the route the "
                     "connector's download_file uses"),
        provenance=_CALLED,
        documentation="https://developers.zenodo.org/",
        documentation_markers=("/api/records",),
        note="a 4 KB README, so the probe itself costs nothing to run",
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
        provenance=_CALLED,
        documentation="https://www.rhea-db.org/help/rest-api",
        documentation_markers=("format=tsv", "columns="),
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
        provenance=_CALLED,
        documentation="https://www.rhea-db.org/help/rest-api",
        documentation_markers=("format=tsv", "columns="),
        note=("a secondary alcohol to a ketone with NAD(+): the reaction "
              "class of the pilot task, read in the oxidation direction"),
    ),

    # ----------------------------------------------------------------------
    # Candidates: routes assembled from documentation, never dialled
    # ----------------------------------------------------------------------
    #
    # Everything below has provenance that is not ``called_from_here``. Each
    # was written because the registry's own worklist asks for it -- 31
    # network-capable sources record no route at all -- and because a probe is
    # the only artefact in this repo that can be *wrong without lying*: it
    # fails, the failure is printed, nothing is written, and the failure text
    # says whether the route, the markers or the documentation was the problem.
    #
    # They were assembled on 2026-10-25 from a container whose network policy
    # denied every one of these hosts, so not one of them has been called.
    # ``eagent sources hosts`` prints the hosts a sweep would need.

    CapabilityProbe(
        source_id="uniref",
        capability="exact_record_fetch",
        url="https://rest.uniprot.org/uniref/UniRef100_P05067.json",
        markers=("UniRef100_P05067", "representativeMember"),
        description=("one UniRef cluster by id: the record a redundancy "
                     "filter reads to find which sequences a representative "
                     "stands for"),
        provenance=RouteProvenance(
            grade="provider_documentation",
            detail=("UniProt's OpenAPI page at rest.uniprot.org/uniref/api/"
                    "docs documents GET /uniref/{id} and gives "
                    "UniRef100_P05067 as its example id. The .json suffix is "
                    "not from that page: it is the spelling the uniprotkb "
                    "probe above already had a pass on, from the same service"),
            recorded_at="2026-10-25",
            query="UniProt REST API uniparc uniref endpoint documentation",
        ),
        documentation="https://rest.uniprot.org/uniref/api/docs",
        documentation_markers=("/uniref/{id}",),
        note=("the provider's own example id rather than one from this "
              "project's enzyme neighbourhood. A UniRef id has the "
              "representative's accession in it, so UniRef100_P07846 is a "
              "guess about which member UniProt picked -- and a probe that "
              "404s on a guessed id teaches nothing about the route"),
    ),
    CapabilityProbe(
        source_id="uniparc",
        capability="exact_record_fetch",
        url="https://rest.uniprot.org/uniparc/search?query=P07846&size=1",
        markers=("uniParcId", "UPI"),
        description=("the UniParc record behind one UniProtKB accession: "
                     "where a sequence seen under two accessions is one entry"),
        provenance=RouteProvenance(
            grade="provider_documentation",
            detail=("rest.uniprot.org/uniparc/api/docs documents GET "
                    "/uniparc/{upi} and /uniparc/{upi}/light. This project "
                    "has no UPI it has verified, so the probe uses the search "
                    "form keyed by accession, which that page's summary did "
                    "not list -- if the form does not exist the probe fails "
                    "and the fix is this URL, not the host"),
            recorded_at="2026-10-25",
            query="UniProt REST API uniparc uniref endpoint documentation",
        ),
        documentation="https://rest.uniprot.org/uniparc/api/docs",
        documentation_markers=("/uniparc/search",),
        note=("P07846 is the accession the uniprotkb probe already has a pass "
              "on, so a failure here is about UniParc and not about the id"),
    ),
    CapabilityProbe(
        source_id="pfam",
        capability="exact_record_fetch",
        url="https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00106/",
        markers=("PF00106", "metadata", "source_database"),
        description=("one Pfam family's entry, through InterPro's API: the "
                     "family record behind a profile hit"),
        provenance=RouteProvenance(
            grade="live_response_seen",
            detail=("a search for the InterPro API returned live responses "
                    "from this exact route shape -- /api/entry/pfam/PF03574/ "
                    "-- described as HTTP 200 application/json with a "
                    "top-level metadata object carrying accession, type and "
                    "source database. PF00106 is the author's own choice of "
                    "accession, not one the search confirmed"),
            recorded_at="2026-10-25",
            query="InterPro API documentation ebi.ac.uk/interpro/api entry pfam JSON",
        ),
        documentation="https://interpro-documentation.readthedocs.io/en/latest/api.html",
        documentation_markers=("/interpro/api/", "entry/pfam"),
        note=("PF00106 is the short-chain dehydrogenase/reductase family this "
              "project's pilot task is about, so a pass exercises the record "
              "the agent would actually read. Pfam is served by InterPro: "
              "there is no pfam.xfam.org route to probe any more"),
    ),
    CapabilityProbe(
        source_id="interpro",
        capability="exact_record_fetch",
        url="https://www.ebi.ac.uk/interpro/api/entry/interpro/IPR002347/",
        markers=("IPR002347", "metadata"),
        description="one InterPro entry by accession, with its member databases",
        provenance=RouteProvenance(
            grade="live_response_seen",
            detail=("same route shape as the pfam probe above, which a search "
                    "returned live responses for; the sourcedb segment is "
                    "'interpro' here rather than 'pfam', which InterPro's "
                    "documented api/datatype/sourcedb/accession form allows. "
                    "IPR002347 is the author's own choice of accession"),
            recorded_at="2026-10-25",
            query="InterPro API documentation ebi.ac.uk/interpro/api entry pfam JSON",
        ),
        documentation="https://interpro-documentation.readthedocs.io/en/latest/api.html",
        documentation_markers=("/interpro/api/", "entry/interpro"),
        note=("IPR002347 is meant to be the SDR family entry. If it is the "
              "wrong accession the probe 404s, and that is the informative "
              "failure: the route is fine and this file is wrong"),
    ),
    CapabilityProbe(
        source_id="mcsa",
        capability="keyword_query",
        url=("https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/"
             "?format=json&entries.reactions.ecs.codes=1.1.1.1"),
        markers=("1.1.1.1", "results"),
        description=("catalytic-mechanism entries for one EC number: the "
                     "residue-level mechanism evidence behind a family"),
        provenance=RouteProvenance(
            grade="provider_documentation",
            detail=("M-CSA's own download page, which a search returned, "
                    "calls the public API the preferred access route and "
                    "names /api/entries/?format=json and /api/residues/"
                    "?format=json, filterable by EC number, UniProt accession "
                    "or M-CSA id. The exact filter spelling here is the "
                    "author's reading of the examples quoted in that summary"),
            recorded_at="2026-10-25",
            query=("M-CSA Mechanism Catalytic Site Atlas API documentation "
                   "ebi.ac.uk thornton-srv m-csa"),
        ),
        documentation="https://www.ebi.ac.uk/thornton-srv/m-csa/download/",
        documentation_markers=("/api/entries/",),
        note=("EC 1.1.1.1 is alcohol dehydrogenase, this project's "
              "neighbourhood. A third-party wrapper reports that the list "
              "endpoint ignores server-side filters and returns large "
              "records; if that is so the response is big but the markers "
              "still decide, which is the point of having markers"),
    ),
    CapabilityProbe(
        source_id="europe_pmc",
        capability="keyword_query",
        url=("https://www.ebi.ac.uk/europepmc/webservices/rest/search"
             "?query=ketoreductase&format=json&pageSize=1"),
        markers=("hitCount", "resultList"),
        description="a literature search returning one hit and the total count",
        provenance=RouteProvenance(
            grade="provider_documentation",
            detail=("the path /europepmc/webservices/rest/search under an EBI "
                    "host first came from an archived Europe PMC page and a "
                    "third-party client's default base URL. On 2026-10-25 the "
                    "service's own OpenAPI definition, served from the API "
                    "host at /europepmc/webservices/rest/v3/api-docs, was "
                    "read from this container: it documents GET /search with "
                    "query, format and pageSize, and spells this base out in "
                    "full. The search term and pageSize=1 are the author's"),
            recorded_at="2026-10-25",
            query=("Europe PMC RESTful web service base URL documentation "
                   "search endpoint"),
        ),
        documentation=("https://www.ebi.ac.uk/europepmc/webservices/rest/"
                       "v3/api-docs"),
        documentation_markers=("webservices/rest/search",),
        note=("this entry is one of the five whose remembered endpoint was "
              "nulled; the hint in its curation notes is where this URL "
              "came from. The documentation is the service's OpenAPI "
              "definition rather than europepmc.org/RestfulWebService, the "
              "page a person reads, because that host answered this "
              "client's user agent with a Cloudflare challenge (HTTP 403) "
              "on 2026-10-25, and a challenge is not documentation anybody "
              "opened. The definition is what the API's own Swagger UI "
              "renders"),
    ),
    CapabilityProbe(
        source_id="alphafold_db",
        capability="exact_record_fetch",
        url="https://alphafold.ebi.ac.uk/api/prediction/P07846",
        markers=("P07846", "cifUrl"),
        description=("the predicted-structure metadata for one accession, "
                     "including the model file URLs"),
        provenance=RouteProvenance(
            grade="third_party_client",
            detail=("base https://alphafold.ebi.ac.uk/api/ and GET "
                    "/prediction/{uniprot_id} come from a mirrored API "
                    "reference and several community wrappers. Two searches "
                    "failed to return EBI's own API page, so the "
                    "documentation URL below is a guess at where it lives"),
            recorded_at="2026-10-25",
            query="alphafold.ebi.ac.uk api-docs official AlphaFold DB API documentation page",
        ),
        documentation="https://alphafold.ebi.ac.uk/api/openapi.json",
        documentation_markers=("/prediction/{qualifier}",),
        note=("P07846 again, so the id is not what fails. The documentation "
              "is the OpenAPI document rather than alphafold.ebi.ac.uk/"
              "api-docs, the interactive page that renders it: that page "
              "builds its text in the browser, so the HTML it serves names no "
              "route at all"),
    ),
    CapabilityProbe(
        source_id="pubchem",
        capability="chemical_structure_query",
        url=("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/"
             "property/MolecularFormula/JSON"),
        markers=("C9H8O4", "MolecularFormula"),
        description="one compound property by CID, through the PUG-REST prolog",
        provenance=RouteProvenance(
            grade="provider_documentation",
            detail=("PubChem's own programmatic-access page describes PUG-"
                    "REST and the prolog pubchem.ncbi.nlm.nih.gov/rest/pug/, "
                    "and a tutorial in the same results builds this exact CID "
                    "2244 molecular-formula call and states C9H8O4 as the "
                    "answer -- which is where the marker comes from, so this "
                    "probe's expected value is documented rather than assumed"),
            recorded_at="2026-10-25",
            query="PubChem PUG REST documentation base URL compound property request",
        ),
        documentation="https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
        documentation_markers=_STATUS_ALONE,
        note=("CID 2244 is aspirin, not a cofactor this project reads. It is "
              "used anyway because it is the one example whose answer the "
              "documentation writes down, and a probe whose expected response "
              "is documented is worth more than one in the right chemical "
              "neighbourhood whose answer nobody has published"),
    ),
    CapabilityProbe(
        source_id="pubmed",
        capability="keyword_query",
        url=("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
             "?db=pubmed&term=ketoreductase&retmax=1&retmode=json"),
        markers=("esearchresult", "idlist"),
        description="an E-utilities search returning one pmid and the count",
        provenance=RouteProvenance(
            grade="provider_documentation",
            detail=("the E-utilities base /entrez/eutils/ and esearch.fcgi "
                    "are both stated in the search results, and NCBI's own "
                    "Bookshelf guide came back as the documentation. The "
                    "retmode=json spelling is the author's"),
            recorded_at="2026-10-25",
            query="NCBI E-utilities efetch protein documentation eutils base URL esearch",
        ),
        documentation="https://www.ncbi.nlm.nih.gov/books/NBK25501/",
        documentation_markers=_STATUS_ALONE,
        note=("no api_key, so NCBI's unauthenticated rate limit applies: "
              "three requests a second across this host, which the two NCBI "
              "probes here stay inside only because the sweep is serial"),
    ),
    CapabilityProbe(
        source_id="ncbi_protein",
        capability="exact_record_fetch",
        url=("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
             "?db=protein&id=P07846&rettype=gp&retmode=text"),
        markers=("P07846", "LOCUS"),
        description="one protein record as a GenPept flat file, by accession",
        provenance=RouteProvenance(
            grade="search_summary",
            detail=("weaker than the pubmed probe beside it on purpose: the "
                    "search stated the base path and esearch.fcgi, and said "
                    "in as many words that no result named the efetch "
                    "endpoint directly. efetch.fcgi is therefore inferred "
                    "from the base, and db=protein from a Bioperl page that "
                    "lists protein among EFetch's databases"),
            recorded_at="2026-10-25",
            query="NCBI E-utilities efetch protein documentation eutils base URL esearch",
        ),
        documentation="https://www.ncbi.nlm.nih.gov/books/NBK25501/",
        documentation_markers=_STATUS_ALONE,
        note=("P07846 is a UniProtKB accession; NCBI indexes the same entry, "
              "so if the route works and the id does not resolve the failure "
              "will name the id. LOCUS is the first line of a GenPept file "
              "and is not on NCBI's error page"),
    ),
    CapabilityProbe(
        source_id="sabio_rk",
        capability="keyword_query",
        url=("https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/sbml"
             "?q=ECNumber:1.1.1.1&pageSize=1"),
        markers=("<sbml", "kineticLaw", "ec-code/1.1.1.1"),
        description=("kinetic laws for one EC number as SBML: the one "
                     "registered source that can supply a measurement"),
        provenance=RouteProvenance(
            grade="provider_documentation",
            detail=("SABIO-RK has replaced the web services its old RESTful "
                    "manual described. On 2026-10-25 the legacy "
                    "/sabioRestWebServices/searchKineticLaws/sbml path and "
                    "the manual itself both redirected, from this container, "
                    "to the new interface's 404 page. The provider's Export "
                    "API page (sabiork.h-its.org/ui/export-api) maps the "
                    "legacy /searchKineticLaws/sbml?q=... to "
                    "/kinlaw-entry/sbml?q=... and quotes ECNumber:1.1.1.1 "
                    "among its query examples, and its OpenAPI document lists "
                    "GET /export-api/sabio/kinlaw-entry/sbml with q, page and "
                    "pageSize. pageSize=1 is the author's, to keep the probe "
                    "to one entry"),
            recorded_at="2026-10-25",
            query="SABIO-RK REST API documentation base URL webServices",
        ),
        documentation="https://sabiork.h-its.org/openapi/export-api.json",
        documentation_markers=("/export-api/sabio/kinlaw-entry/sbml",),
        note=("this host is the reason markers exist in this module: its "
              "legacy REST path answered 302 and then 200, on the provider's "
              "own 404 page. The documentation is the OpenAPI document "
              "rather than the Export API page because every /ui/ path, "
              "including the 404 page, answers 200 with the same application "
              "shell: a status check on one would pass on a page that does "
              "not exist, while an unknown OpenAPI name is a real 404. The "
              "export API is rate-limited to 60 requests a minute per IP"),
    ),
    CapabilityProbe(
        source_id="kegg",
        capability="exact_record_fetch",
        url="https://rest.kegg.jp/get/ec:1.1.1.1",
        markers=("1.1.1.1", "ENTRY"),
        description="one EC entry as a KEGG flat file, through the get operation",
        provenance=RouteProvenance(
            grade="third_party_client",
            detail=("the base rest.kegg.jp and the get/{id} form come from a "
                    "client library's source and a Bioconductor vignette; "
                    "KEGG's own API page was named by a secondary source but "
                    "did not come back in the search, so the documentation "
                    "URL below is unconfirmed. The library gives http://, "
                    "raised to https:// here because this module refuses "
                    "anything else"),
            recorded_at="2026-10-25",
            query="KEGG REST API documentation rest.kegg.jp get entry operation",
        ),
        documentation="https://www.kegg.jp/kegg/rest/keggapi.html",
        documentation_markers=("rest.kegg.jp/get",),
        note=("reachability is not permission here more than anywhere: KEGG's "
              "terms restrict bulk and commercial use, and the registry's "
              "needs_legal_review for this entry stays true whatever this "
              "probe does"),
    ),
)


def probes_for(source_id: str) -> tuple[CapabilityProbe, ...]:
    """Every shipped probe for one source."""
    return tuple(p for p in PROBES if p.source_id == source_id)


def never_called_probes(
    probes: Iterable[CapabilityProbe] | None = None,
) -> tuple[CapabilityProbe, ...]:
    """The probes whose URL nobody in this project has reached yet.

    These are candidates, not claims. Each one fails harmlessly: a wrong URL
    is a printed failure and nothing written, which is the whole reason it is
    safe to ship a route read off documentation rather than wait until
    somebody has a network that can reach it.
    """
    return tuple(p for p in (PROBES if probes is None else probes)
                 if not p.provenance.ever_called)


def probe_hosts(
    probes: Iterable[CapabilityProbe] | None = None,
) -> dict[str, list[str]]:
    """Every host the sweep contacts, mapped to what needs it.

    Two reasons this is not just the route URLs. A probe fails when its
    *documentation* is unreachable as well as when its route is, because the
    registry refuses an endpoint that cannot be traced to a page a person can
    read -- so an allow-list built from routes alone makes every probe fail on
    its documentation, which reads like the resources being down. And the
    documentation often lives on a different host from the API: RCSB's data
    API is ``data.rcsb.org`` and its documentation is ``www.rcsb.org``.
    """
    out: dict[str, set[str]] = {}
    for probe in (PROBES if probes is None else probes):
        out.setdefault(_host(probe.url), set()).add(
            f"route: {probe.source_id}/{probe.capability}")
        out.setdefault(_host(probe.documentation), set()).add(
            f"docs: {probe.source_id}")
    return {host: sorted(why) for host, why in sorted(out.items())}


def _host(url: str) -> str:
    """The host of a URL, which is the unit a network policy allows."""
    return url.split("://", 1)[1].split("/", 1)[0]


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
