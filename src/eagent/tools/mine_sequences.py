"""Interface ``mine_sequences``: widen a candidate pool from characterised seeds.

Why this module is shaped the way it is
---------------------------------------

**Multi-seed by construction.** The usual failure of an enzyme-mining campaign
is not that the search was bad, it is that the search was run from one
favourite enzyme. Everything retrieved is then a homologue of that one
scaffold, the pool looks large, and the diversity is an illusion -- a single
unlucky property of the seed (an unstable fold, a narrow pocket, a cofactor
preference) is inherited by every candidate and the whole round fails for one
reason. :meth:`MineSequences.execute` therefore takes ``seeds`` as a sequence
and refuses a one-element set unless the caller passes
``allow_single_seed=True`` together with a written justification. A single-seed
run remains possible, because sometimes it is all that exists, but it is an
explicit, flagged, recorded choice rather than the default path.

**Seeds must carry experimental support.** A seed is the anchor the entire pool
hangs from. Starting from an entry whose function is itself an annotation
propagated by similarity produces a pool with no experimental anchor at all,
while still looking exactly like a well-founded one. Seeds without at least
homolog-level experimental evidence are dropped and listed; if none survive the
step fails rather than mining from annotation.

**Relevance is never relaxed to hit a number.** ``Budget.initial_sequence_target``
is a planning target, not a quota to be met. There is deliberately no retry
loop, no "widen the e-value until we have 2000", and no fallback policy object
in this module: the :class:`RetentionPolicy` is fingerprinted on entry and
re-checked after filtering, so a future edit that mutates it mid-run raises
:class:`~eagent.errors.FabricationGuardError` instead of quietly producing a
padded pool. A short pool is reported as a short pool, with the reason.

**Nothing here invents a sequence.** Search output gives subject identifiers;
the residues come from the local sequence database file. An identifier that
cannot be resolved in that file is dropped and counted, never reconstructed.

**Offline by default.** The adapters drive local binaries against a local
database file, which is the on-disk cache. No adapter in this module reaches
the network, and a database marked remote, or an adapter flag that would make a
local binary call out (``blastp -remote``), is refused unless
``ctx.policy.allow_network`` is set. Independently of the network flag, an
unpublished seed sequence may not be submitted to an external service without
the ``external_sequence_submission`` approval; that is enforced in
:func:`guard_external_submission`, not in a comment.

**The binaries are not installed here.** blastp, mmseqs2 and hmmsearch are
absent from this environment on purpose, so every adapter resolves its
executable through an injectable ``executable_finder`` and runs it through an
injectable :class:`CommandRunner`. That seam is the supported way to test the
adapters, with either a fake runner or a fake executable on ``PATH``, and it is
also how a workflow engine would run them in a container.
"""

from __future__ import annotations

import abc
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..context import RunContext
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..errors import ApprovalRequiredError, EAgentError, FabricationGuardError, ToolUnavailableError
from ..provenance import sequence_hash, sha256_file, sha256_obj
from ..schemas import EvidenceRef, EvidenceStrength, SequenceRecord
from ..science.diversity import jaccard_distance, kmer_set
from ..science.numbering import needleman_wunsch
from .base import ScientificInterface

__all__ = [
    "STANDARD_AA",
    "EXTERNAL_SUBMISSION_GATE",
    "DEFAULT_FRAGMENT_LENGTH_FRACTION",
    "DEFAULT_EXCESS_LENGTH_FACTOR",
    "DEFAULT_CLUSTER_IDENTITY",
    "DEFAULT_KMER_PREFILTER_SIMILARITY",
    "DEFAULT_MAX_CLUSTER_ALIGNMENTS",
    "FastaEntry",
    "parse_fasta",
    "read_fasta",
    "write_fasta",
    "index_fasta",
    "CommandResult",
    "CommandRunner",
    "SubprocessRunner",
    "SearchExecutionError",
    "SearchQuery",
    "SearchHit",
    "SearchAdapter",
    "BlastpAdapter",
    "MMseqs2Adapter",
    "HmmsearchAdapter",
    "SequenceDatabase",
    "SeedSequence",
    "FamilyProfile",
    "RetentionPolicy",
    "LengthExpectation",
    "SequenceCluster",
    "cluster_sequences",
    "guard_external_submission",
    "RetrievalRow",
    "MineSequences",
]


#: The twenty residues a standard search tool and a standard expression host
#: both understand. Anything else (X, B, Z, J, U, O, *) means the record is
#: ambiguous, selenocysteine/pyrrolysine, or a translated fragment -- all of
#: which must be flagged rather than silently accepted into a synthesis pool.
STANDARD_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: Approval gate for sending sequence data to a service outside this machine.
#: Named here so the controller, the manifest and this module agree on the
#: string; an unpublished sequence leaving the building is a disclosure event,
#: not a technical detail.
EXTERNAL_SUBMISSION_GATE: str = "external_sequence_submission"

# --------------------------------------------------------------------------
# QC localisation defaults.
#
# These are *not* catalytic thresholds and nothing scientific is concluded from
# them. They localise quality problems for a human to look at, and each needs
# per-family calibration before it is trusted: a family with a natural 250-500
# residue length spread will trip the fragment screen constantly, and a family
# of near-identical paralogues will collapse into one cluster. Override them
# per run through :class:`LengthExpectation` and the ``cluster_identity``
# argument rather than editing these numbers.
# --------------------------------------------------------------------------

#: Below this fraction of the shortest expected family length a hit is called a
#: fragment. Needs per-family calibration.
DEFAULT_FRAGMENT_LENGTH_FRACTION: float = 0.80

#: Above this multiple of the longest expected family length a hit is called
#: implausibly long (fusion protein, mis-predicted ORF). Needs per-family
#: calibration.
DEFAULT_EXCESS_LENGTH_FACTOR: float = 1.25

#: Default greedy clustering identity. Purely a bookkeeping grain for the
#: downstream diversity step; it removes nothing from the pool.
DEFAULT_CLUSTER_IDENTITY: float = 0.90

#: k-mer Jaccard similarity below which the clusterer does not bother to align
#: a pair. A screen for speed, not a similarity claim: the k-mer/identity
#: relationship is monotone in practice but is not a bound, so a pair just
#: under the screen can in principle be a true cluster-mate. The count of
#: screened-out pairs is reported so the approximation is visible.
DEFAULT_KMER_PREFILTER_SIMILARITY: float = 0.10

#: Alignment budget for one clustering run. Exceeding it degrades loudly
#: (PARTIAL + flag) instead of silently leaving the tail unclustered.
DEFAULT_MAX_CLUSTER_ALIGNMENTS: int = 200_000


# ==========================================================================
# FASTA, in pure Python
# ==========================================================================

@dataclass(frozen=True)
class FastaEntry:
    """One FASTA record, with the header split where tools split it.

    Search tools report the first whitespace-delimited token of the header as
    the subject id and throw the rest away. Keeping ``identifier`` and
    ``description`` apart means a hit can be joined back to its sequence by the
    same rule the search tool used, instead of by a string comparison that
    happens to work until a description contains a space.
    """

    identifier: str
    description: str
    sequence: str
    was_softmasked: bool = False

    @property
    def header(self) -> str:
        """The ``>`` line without the marker."""
        return f"{self.identifier} {self.description}".strip()

    @property
    def length(self) -> int:
        """Residue count, the quantity the fragment screen is applied to."""
        return len(self.sequence)

    @property
    def sequence_sha256(self) -> str:
        """Join key shared with the rest of the pipeline."""
        return sequence_hash(self.sequence)


def parse_fasta(text: str, *, source: str = "<text>") -> list[FastaEntry]:
    """Parse FASTA text.

    Raises on content before the first ``>``: a truncated download that begins
    mid-sequence would otherwise silently contribute a headless fragment to the
    first record, producing a chimeric sequence that still looks like a
    protein.

    Lowercase residues are upper-cased and the record is marked
    ``was_softmasked``. Soft-masking marks low-complexity or repeat-masked
    regions; discarding the case silently would hide that part of the sequence
    was flagged by whatever produced the file. A trailing ``*`` (stop codon
    from a translated ORF) is removed, and an internal ``*`` is kept so the QC
    screen can see it.
    """
    entries: list[FastaEntry] = []
    identifier: str | None = None
    description = ""
    chunks: list[str] = []
    softmasked = False

    def flush() -> None:
        """Emit the record accumulated so far, if any."""
        if identifier is None:
            return
        seq = "".join(chunks)
        if seq.endswith("*"):
            seq = seq[:-1]
        entries.append(FastaEntry(identifier, description, seq, softmasked))

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            head = line[1:].strip()
            parts = head.split(None, 1)
            identifier = parts[0] if parts else ""
            description = parts[1] if len(parts) > 1 else ""
            chunks = []
            softmasked = False
            if not identifier:
                raise ValueError(
                    f"{source}:{lineno}: FASTA header with no identifier; a "
                    f"sequence that cannot be named cannot be traced"
                )
            continue
        if identifier is None:
            raise ValueError(
                f"{source}:{lineno}: sequence data before the first '>' header; "
                f"the file is truncated and the leading residues belong to an "
                f"unknown record"
            )
        if any(c.islower() for c in line):
            softmasked = True
        chunks.append("".join(line.split()).upper())

    flush()
    return entries


def read_fasta(path: str | Path) -> list[FastaEntry]:
    """Read a FASTA file from disk, naming the file in any parse error."""
    p = Path(path)
    return parse_fasta(p.read_text(encoding="utf-8"), source=str(p))


def write_fasta(entries: Iterable[FastaEntry], path: str | Path,
                line_width: int = 60) -> Path:
    """Write FASTA with wrapped lines, creating parent directories.

    Wrapping at a fixed width rather than writing one long line because several
    downstream tools (and every text editor a human will open this in) handle
    wrapped FASTA better, and because a diff of two wrapped files is readable.
    """
    if line_width < 1:
        raise ValueError(f"line_width must be >= 1, got {line_width}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for e in entries:
        out.append(f">{e.header}\n" if e.description else f">{e.identifier}\n")
        seq = e.sequence
        for i in range(0, len(seq), line_width):
            out.append(seq[i:i + line_width] + "\n")
    p.write_text("".join(out), encoding="utf-8")
    return p


def index_fasta(entries: Iterable[FastaEntry]) -> tuple[dict[str, FastaEntry], list[str]]:
    """Index records by identifier, returning ``(index, duplicate_ids)``.

    Duplicates are returned rather than resolved. Two records sharing an
    identifier mean the database file was concatenated from overlapping
    sources, and picking either one at random would attach the wrong organism,
    length and provenance to every hit on that id.

    Secondary keys are added for pipe-delimited headers (``sp|P12345|NAME``),
    because BLAST may report either form depending on how the database was
    built. A secondary key is registered only when it is unambiguous.
    """
    index: dict[str, FastaEntry] = {}
    duplicates: list[str] = []
    secondary: dict[str, list[FastaEntry]] = {}
    for e in entries:
        if e.identifier in index:
            duplicates.append(e.identifier)
            continue
        index[e.identifier] = e
        for token in e.identifier.split("|"):
            token = token.strip()
            if token and token != e.identifier:
                secondary.setdefault(token, []).append(e)
    for key, hits in secondary.items():
        if key not in index and len(hits) == 1:
            index[key] = hits[0]
    return index, duplicates


# ==========================================================================
# The subprocess seam
# ==========================================================================

@dataclass(frozen=True)
class CommandResult:
    """What a runner reports back. Mirrors the part of
    :class:`subprocess.CompletedProcess` the adapters actually read."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    """How an adapter executes a binary.

    A protocol rather than a direct :func:`subprocess.run` call so the adapters
    can be exercised without the binaries, and so a deployment can route them
    through a container or a scheduler without touching this module.
    """

    def __call__(self, argv: Sequence[str], *, cwd: Path | None = None,
                 timeout: float | None = None) -> CommandResult:
        ...


class SubprocessRunner:
    """Default runner: a real local subprocess, output captured as text.

    ``check=False`` on purpose. A non-zero exit is handled by the adapter,
    which can then report the tool's own stderr; a :class:`CalledProcessError`
    traceback would lose it.
    """

    def __call__(self, argv: Sequence[str], *, cwd: Path | None = None,
                 timeout: float | None = None) -> CommandResult:
        proc = subprocess.run(
            list(argv), cwd=str(cwd) if cwd else None, timeout=timeout,
            capture_output=True, text=True, check=False,
        )
        return CommandResult(tuple(argv), proc.returncode, proc.stdout, proc.stderr)


class SearchExecutionError(EAgentError):
    """A search binary was present but failed, or produced unusable output.

    Distinct from :class:`~eagent.errors.ToolUnavailableError`: "not installed"
    is a configuration problem the operator fixes once, while "ran and failed"
    usually means the database or the query is wrong and the run must stop
    rather than continue with the hits that happened to be parsed first.
    """


# ==========================================================================
# Inputs
# ==========================================================================

class SequenceDatabase(BaseModel):
    """A searchable sequence database, pinned to a version and a local file.

    ``version`` is required and may not be blank. An accession without the
    release it came from is not a stable identity -- entries are merged,
    demerged and re-annotated between releases -- so a pool whose database
    version is "unknown" cannot be reproduced or re-joined later.

    ``fasta_path`` is the local cache this module reads residues from, and is
    required even when ``search_target`` points at a pre-built BLAST or MMseqs2
    index: an index is not readable as sequences, and reconstructing residues
    from a hit table is exactly the fabrication this pipeline forbids.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    fasta_path: str
    search_target: str | None = Field(
        None, description="Path or alias handed to the search binary; defaults "
                          "to fasta_path."
    )
    is_remote: bool = Field(
        False, description="True if searching this target sends data off this "
                           "machine. Gated by ExecutionPolicy.allow_network and "
                           "by the external-submission approval."
    )
    license: str | None = None
    notes: str = ""

    @field_validator("name", "version", "fasta_path")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError(
                "name, version and fasta_path are all required: a pool mined "
                "from an unidentified database snapshot cannot be reproduced"
            )
        return str(v).strip()

    def target(self) -> str:
        """What the search binary is pointed at: an index alias, or the FASTA."""
        return self.search_target or self.fasta_path

    def resolve_fasta(self, cache_dir: Path | None = None) -> Path:
        """Absolute path to the local FASTA, relative paths taken from the cache."""
        p = Path(self.fasta_path)
        if not p.is_absolute() and cache_dir is not None:
            p = Path(cache_dir) / p
        return p

    @property
    def label(self) -> str:
        """``name@version``: the only form in which this database may be cited."""
        return f"{self.name}@{self.version}"


class SeedSequence(BaseModel):
    """A characterised starting point for the search.

    Carries its evidence rather than a boolean, so the step can say *why* a
    seed is trusted and refuse the ones that are trusted only by annotation
    transfer.

    ``is_published`` is three-valued and ``None`` means "not recorded". The
    submission guard treats unknown as unpublished, because the two mistakes
    are not symmetric: over-restricting costs an approval click, while
    under-restricting discloses an unpublished sequence to a third party and
    cannot be undone.
    """

    model_config = ConfigDict(extra="forbid")

    accession: str
    sequence: str
    organism: str | None = None
    description: str | None = None
    family_template_id: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    is_published: bool | None = None

    @field_validator("sequence")
    @classmethod
    def _clean(cls, v: str) -> str:
        seq = "".join(str(v).split()).upper()
        if not seq:
            raise ValueError("seed sequence is empty")
        return seq

    @property
    def sequence_sha256(self) -> str:
        """Join key, so a seed and a mined hit of the same protein collapse."""
        return sequence_hash(self.sequence)

    @property
    def max_strength(self) -> EvidenceStrength:
        """Strongest evidence attached, defaulting to the weakest possible class.

        Defaults down, not up: a seed with no evidence at all is treated as a
        computational construct rather than as unannotated-but-probably-fine.
        """
        if not self.evidence:
            return EvidenceStrength.COMPUTATIONAL_CONSTRUCT
        return max((e.strength for e in self.evidence), key=lambda s: s.rank)

    @property
    def has_experimental_support(self) -> bool:
        """True when some evidence is an experiment on this sequence or a homologue.

        An EC number or a catalytic-activity line is an annotation, and a pool
        anchored on one has no experimental anchor anywhere in it.
        """
        return self.max_strength.rank >= EvidenceStrength.HOMOLOG_EXPERIMENTAL.rank

    def to_query(self) -> "SearchQuery":
        """Query form of this seed, carrying its publication status to the guard."""
        return SearchQuery(
            identifier=self.accession, kind="sequence", sequence=self.sequence,
            family_template_id=self.family_template_id,
            is_published=self.is_published,
        )


class FamilyProfile(BaseModel):
    """An HMM profile standing in for a whole family rather than one sequence.

    A profile search is the one retrieval route that carries family evidence by
    itself: a hit to the family's HMM is a statement about the family, whereas a
    BLAST hit to one member is a statement about that member. ``source`` records
    where the profile came from, because a profile built in-house from an
    unreviewed alignment and one taken from Pfam are not interchangeable.
    """

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    profile_path: str
    family_template_id: str
    source: str
    notes: str = ""

    @field_validator("profile_id", "profile_path", "family_template_id", "source")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("profile_id, profile_path, family_template_id and "
                             "source are all required for a profile search")
        return str(v).strip()

    def to_query(self) -> "SearchQuery":
        """Query form of this profile; a published profile discloses nothing."""
        return SearchQuery(
            identifier=self.profile_id, kind="profile",
            profile_path=self.profile_path,
            family_template_id=self.family_template_id,
            is_published=True,   # a published family profile, not private data
        )


@dataclass(frozen=True)
class SearchQuery:
    """What is handed to an adapter: a seed sequence or a family profile."""

    identifier: str
    kind: str                     # "sequence" | "profile"
    sequence: str | None = None
    profile_path: str | None = None
    family_template_id: str | None = None
    is_published: bool | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("sequence", "profile"):
            raise ValueError(f"unknown query kind {self.kind!r}")
        if self.kind == "sequence" and not self.sequence:
            raise ValueError(f"sequence query {self.identifier!r} has no sequence")
        if self.kind == "profile" and not self.profile_path:
            raise ValueError(f"profile query {self.identifier!r} has no profile path")


@dataclass(frozen=True)
class SearchHit:
    """One subject reported by one method for one query.

    Every optional field is ``None`` when the method does not measure it.
    ``None`` is not zero: hmmsearch reports no percent identity at all, and a
    0.0 there would read as "no similarity" and silently disqualify every
    profile hit in a downstream identity filter.
    """

    query_id: str
    query_kind: str
    subject_id: str
    search_method: str
    database_name: str
    database_version: str
    percent_identity: float | None = None   # 0-100
    query_coverage: float | None = None     # fraction 0-1, uniform across adapters
    evalue: float | None = None
    bitscore: float | None = None
    family_template_id: str | None = None
    raw: Mapping[str, str] = field(default_factory=dict)


def _float_or_none(token: str | None) -> float | None:
    """Parse a numeric column, or ``None`` when the tool wrote something else.

    Returning ``None`` rather than 0.0 for an unparsable cell keeps a parser
    surprise from becoming a measured value.
    """
    if token is None:
        return None
    t = token.strip()
    if not t or t in ("-", "NA", "N/A", "nan", "NaN"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _safe_token(text: str) -> str:
    """Filename-safe version of an accession, for scratch query files."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)[:80] or "query"


# ==========================================================================
# Search adapters
# ==========================================================================

class SearchAdapter(abc.ABC):
    """Base class for a homology-search backend.

    Three deliberate properties:

    * the executable is resolved through ``shutil.which`` (or an injected
      finder) and a missing binary raises
      :class:`~eagent.errors.ToolUnavailableError` with an install hint -- the
      step degrades loudly instead of falling back to a pure-Python imitation
      of BLAST, which would produce numbers that look like e-values and are not;
    * execution goes through an injected :class:`CommandRunner`, which is the
      supported test seam in an environment with none of these tools installed;
    * ``reaches_network`` is declared per adapter and checked against the run
      policy before anything is executed.
    """

    method: ClassVar[str] = "unnamed"
    binary: ClassVar[str] = ""
    install_hint: ClassVar[str] = ""
    accepts_query_kind: ClassVar[str] = "sequence"
    reaches_network: ClassVar[bool] = False
    #: Flags that would make a nominally local binary contact a remote service.
    network_flags: ClassVar[frozenset[str]] = frozenset({"-remote", "--remote"})

    def __init__(
        self,
        runner: CommandRunner | None = None,
        executable_finder: Callable[[str], str | None] | None = None,
        extra_args: Sequence[str] = (),
        timeout_s: float | None = 3600.0,
        max_target_seqs: int = 5000,
        threads: int = 1,
    ) -> None:
        self.runner: CommandRunner = runner or SubprocessRunner()
        self.executable_finder: Callable[[str], str | None] = executable_finder or shutil.which
        self.extra_args: tuple[str, ...] = tuple(extra_args)
        self.timeout_s = timeout_s
        self.max_target_seqs = max_target_seqs
        self.threads = threads

    # -- availability ---------------------------------------------------
    def resolve_binary(self) -> str:
        """Absolute path to the executable, or raise with an install hint."""
        found = self.executable_finder(self.binary)
        if not found:
            raise ToolUnavailableError(self.binary, self.install_hint)
        return found

    def is_available(self) -> bool:
        """Whether the binary is on PATH. Never used to pick a silent fallback."""
        return bool(self.executable_finder(self.binary))

    # -- subclass hooks --------------------------------------------------
    @abc.abstractmethod
    def build_argv(self, exe: str, query: SearchQuery, database: SequenceDatabase,
                   workdir: Path, max_evalue: float) -> tuple[list[str], Path | None]:
        """Command line plus the output file to read, or ``None`` for stdout."""

    @abc.abstractmethod
    def parse(self, text: str, query: SearchQuery,
              database: SequenceDatabase) -> list[SearchHit]:
        """Turn the tool's tabular output into hits. Unmeasured fields stay ``None``."""

    # -- fixed driver ----------------------------------------------------
    def search(self, query: SearchQuery, database: SequenceDatabase, workdir: Path,
               *, max_evalue: float, allow_network: bool = False) -> list[SearchHit]:
        """Run one query against one database and return parsed hits."""
        if query.kind != self.accepts_query_kind:
            raise SearchExecutionError(
                f"{self.method} takes a {self.accepts_query_kind} query, got "
                f"{query.kind!r} ({query.identifier}); using the wrong backend "
                f"for a query silently changes what the hits mean"
            )
        if (self.reaches_network or self._requests_network()) and not allow_network:
            raise ToolUnavailableError(
                f"{self.binary} (network mode)",
                "ExecutionPolicy.allow_network is False; this run may only "
                "search local database files",
            )
        exe = self.resolve_binary()
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        argv, out_path = self.build_argv(exe, query, database, workdir, max_evalue)
        result = self.runner(argv, cwd=workdir, timeout=self.timeout_s)
        if result.returncode != 0:
            raise SearchExecutionError(
                f"{self.method} exited {result.returncode} for query "
                f"{query.identifier} against {database.label}: "
                f"{(result.stderr or result.stdout or '').strip()[:500]}"
            )
        if out_path is None:
            text = result.stdout
        else:
            if not out_path.exists():
                raise SearchExecutionError(
                    f"{self.method} reported success but wrote no output at "
                    f"{out_path}; refusing to treat a missing file as zero hits"
                )
            text = out_path.read_text(encoding="utf-8")
        return self.parse(text, query, database)

    def _requests_network(self) -> bool:
        return any(a in self.network_flags for a in self.extra_args)

    def _write_query_fasta(self, query: SearchQuery, workdir: Path) -> Path:
        path = workdir / f"query_{_safe_token(query.identifier)}.fasta"
        write_fasta(
            [FastaEntry(query.identifier, "", query.sequence or "")], path
        )
        return path


class BlastpAdapter(SearchAdapter):
    """blastp through a fixed tabular format.

    The output format is pinned here rather than taken from the caller because
    the column order *is* the parser. ``qcovs`` is requested explicitly: BLAST's
    default format has no coverage column, and identity over a 40-residue local
    alignment is not a statement about the protein.
    """

    method = "blastp"
    binary = "blastp"
    install_hint = "conda install -c bioconda blast, or apt-get install ncbi-blast+"
    accepts_query_kind = "sequence"
    reaches_network = False

    #: qseqid sseqid pident length qstart qend sstart send evalue bitscore qcovs
    outfmt: ClassVar[str] = (
        "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qcovs"
    )
    columns: ClassVar[tuple[str, ...]] = (
        "qseqid", "sseqid", "pident", "length", "qstart", "qend",
        "sstart", "send", "evalue", "bitscore", "qcovs",
    )

    def build_argv(self, exe: str, query: SearchQuery, database: SequenceDatabase,
                   workdir: Path, max_evalue: float) -> tuple[list[str], Path | None]:
        """Command line for blastp; results come back on stdout.

        The e-value is pushed into the binary as well as being applied later,
        so a huge hit table is never produced in the first place. The later
        check is still the authoritative one.
        """
        qpath = self._write_query_fasta(query, workdir)
        argv = [
            exe,
            "-query", str(qpath),
            "-db", database.target(),
            "-outfmt", self.outfmt,
            "-evalue", repr(max_evalue),
            "-max_target_seqs", str(self.max_target_seqs),
            "-num_threads", str(self.threads),
        ]
        argv.extend(self.extra_args)
        return argv, None

    def parse(self, text: str, query: SearchQuery,
              database: SequenceDatabase) -> list[SearchHit]:
        """Parse the pinned tabular format, refusing any row of the wrong width.

        A short row means the format string and this parser have diverged, and
        every value after the missing column would be read from the wrong
        field -- an e-value that is really a bitscore passes a filter happily.
        """
        hits: list[SearchHit] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < len(self.columns):
                raise SearchExecutionError(
                    f"blastp output line has {len(parts)} columns, expected "
                    f"{len(self.columns)}: {line[:200]!r}. The -outfmt string "
                    f"and the parser have diverged; every number after the "
                    f"short column would be read from the wrong field."
                )
            row = dict(zip(self.columns, parts))
            qcovs = _float_or_none(row["qcovs"])
            hits.append(SearchHit(
                query_id=query.identifier,
                query_kind=query.kind,
                subject_id=row["sseqid"],
                search_method=self.method,
                database_name=database.name,
                database_version=database.version,
                percent_identity=_float_or_none(row["pident"]),
                # BLAST reports qcovs as a percent; the pool stores a fraction
                # so that coverage from three different tools is comparable.
                query_coverage=None if qcovs is None else qcovs / 100.0,
                evalue=_float_or_none(row["evalue"]),
                bitscore=_float_or_none(row["bitscore"]),
                family_template_id=query.family_template_id,
                raw=row,
            ))
        return hits


class MMseqs2Adapter(SearchAdapter):
    """mmseqs2 ``easy-search``, the preferred backend at pool scale.

    Preferred over blastp here only for throughput; the hits mean the same
    thing. ``fident`` is a fraction and is converted to a percent at parse
    time, so a pool mixing mmseqs2 and blastp rows is not comparing 0.42
    against 42.
    """

    method = "mmseqs2"
    binary = "mmseqs"
    install_hint = "conda install -c bioconda mmseqs2"
    accepts_query_kind = "sequence"
    reaches_network = False

    format_output: ClassVar[str] = "query,target,fident,qcov,evalue,bits"
    columns: ClassVar[tuple[str, ...]] = (
        "query", "target", "fident", "qcov", "evalue", "bits",
    )

    def build_argv(self, exe: str, query: SearchQuery, database: SequenceDatabase,
                   workdir: Path, max_evalue: float) -> tuple[list[str], Path | None]:
        """Command line for ``mmseqs easy-search``, which writes to a file.

        The scratch directory is per query so two queries cannot share mmseqs2
        temporary state and silently reuse each other's intermediate results.
        """
        qpath = self._write_query_fasta(query, workdir)
        out_path = workdir / f"mmseqs_{_safe_token(query.identifier)}.tsv"
        tmp_dir = workdir / f"mmseqs_tmp_{_safe_token(query.identifier)}"
        argv = [
            exe, "easy-search",
            str(qpath), database.target(), str(out_path), str(tmp_dir),
            "--format-output", self.format_output,
            "-e", repr(max_evalue),
            "--max-seqs", str(self.max_target_seqs),
            "--threads", str(self.threads),
        ]
        argv.extend(self.extra_args)
        return argv, out_path

    def parse(self, text: str, query: SearchQuery,
              database: SequenceDatabase) -> list[SearchHit]:
        """Parse ``--format-output`` columns, converting ``fident`` to a percent."""
        hits: list[SearchHit] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < len(self.columns):
                raise SearchExecutionError(
                    f"mmseqs2 output line has {len(parts)} columns, expected "
                    f"{len(self.columns)}: {line[:200]!r}"
                )
            row = dict(zip(self.columns, parts))
            fident = _float_or_none(row["fident"])
            hits.append(SearchHit(
                query_id=query.identifier,
                query_kind=query.kind,
                subject_id=row["target"],
                search_method=self.method,
                database_name=database.name,
                database_version=database.version,
                percent_identity=None if fident is None else fident * 100.0,
                query_coverage=_float_or_none(row["qcov"]),
                evalue=_float_or_none(row["evalue"]),
                bitscore=_float_or_none(row["bits"]),
                family_template_id=query.family_template_id,
                raw=row,
            ))
        return hits


class HmmsearchAdapter(SearchAdapter):
    """hmmsearch against a family profile, parsed from ``--domtblout``.

    Two things this adapter refuses to do:

    * it reports **no** percent identity, because hmmsearch does not compute
      one. A profile match is a statement about family membership, not about
      pairwise similarity to any particular member, and manufacturing a number
      here would let a family-level signal masquerade as a sequence-level one;
    * it does not sum overlapping domain envelopes. Coverage is the length of
      the *union* of matched profile intervals over the profile length, so a
      repeated domain cannot report 180% coverage.
    """

    method = "hmmsearch"
    binary = "hmmsearch"
    install_hint = "conda install -c bioconda hmmer, or apt-get install hmmer"
    accepts_query_kind = "profile"
    reaches_network = False

    def build_argv(self, exe: str, query: SearchQuery, database: SequenceDatabase,
                   workdir: Path, max_evalue: float) -> tuple[list[str], Path | None]:
        """Command line for hmmsearch, reading the per-domain table.

        The domain table rather than the sequence table because the profile
        coverage of each hit is only computable from the domain envelopes, and
        coverage is what distinguishes a full-length family member from a
        single matched domain of a multi-domain protein.
        """
        out_path = workdir / f"hmmsearch_{_safe_token(query.identifier)}.domtbl"
        argv = [
            exe,
            "--domtblout", str(out_path),
            "-E", repr(max_evalue),
            "--cpu", str(self.threads),
            "-o", "/dev/null",
            str(query.profile_path),
            database.target(),
        ]
        argv.extend(self.extra_args)
        return argv, out_path

    def parse(self, text: str, query: SearchQuery,
              database: SequenceDatabase) -> list[SearchHit]:
        """Aggregate the domain rows of each target into one hit."""
        # target(0) ... qlen(5) full-E(6) full-score(7) ... hmm_from(15) hmm_to(16)
        per_target: dict[str, dict[str, Any]] = {}
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 23:
                raise SearchExecutionError(
                    f"hmmsearch --domtblout line has {len(cols)} fields, "
                    f"expected at least 23: {line[:200]!r}"
                )
            target = cols[0]
            qlen = _float_or_none(cols[5])
            entry = per_target.setdefault(target, {
                "qlen": qlen,
                "evalue": _float_or_none(cols[6]),
                "bitscore": _float_or_none(cols[7]),
                "intervals": [],
            })
            hmm_from = _float_or_none(cols[15])
            hmm_to = _float_or_none(cols[16])
            if hmm_from is not None and hmm_to is not None:
                entry["intervals"].append((int(hmm_from), int(hmm_to)))

        hits: list[SearchHit] = []
        for target, entry in per_target.items():
            qlen = entry["qlen"]
            coverage: float | None = None
            if qlen:
                covered = _union_length(entry["intervals"])
                coverage = min(1.0, covered / float(qlen)) if covered else 0.0
            hits.append(SearchHit(
                query_id=query.identifier,
                query_kind=query.kind,
                subject_id=target,
                search_method=self.method,
                database_name=database.name,
                database_version=database.version,
                percent_identity=None,          # not measured by a profile search
                query_coverage=coverage,
                evalue=entry["evalue"],
                bitscore=entry["bitscore"],
                family_template_id=query.family_template_id,
                raw={"profile": query.identifier, "qlen": str(entry["qlen"])},
            ))
        return hits


def _union_length(intervals: Sequence[tuple[int, int]]) -> int:
    """Total length covered by possibly overlapping inclusive intervals."""
    if not intervals:
        return 0
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    total = 0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start > cur_end + 1:
            total += cur_end - cur_start + 1
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start + 1
    return total


# ==========================================================================
# Retention policy and length expectation
# ==========================================================================

class RetentionPolicy(BaseModel):
    """The run's stated relevance settings, fixed before any hit is seen.

    These are *search relevance* settings, not catalytic thresholds: they decide
    which rows of a hit table are worth carrying forward, and nothing about
    whether an enzyme works. They are recorded verbatim in provenance, and this
    object is fingerprinted at the start of filtering and re-checked afterwards,
    so a later edit that loosens a cut-off to reach a target count cannot pass
    unnoticed.

    ``family_evidence_min_identity`` is ``None`` by default and that default is
    meaningful: with no configured level, a BLAST hit to a characterised seed is
    accepted as *homology-level* family evidence and flagged for confirmation by
    ``annotate_family``, rather than being promoted here by a hard-coded
    twilight-zone percentage that this module has no basis to choose.
    """

    model_config = ConfigDict(extra="forbid")

    max_evalue: float = Field(1e-5, gt=0.0)
    min_query_coverage: float | None = Field(0.60, ge=0.0, le=1.0)
    min_percent_identity: float | None = Field(None, ge=0.0, le=100.0)
    max_percent_identity: float | None = Field(
        None, ge=0.0, le=100.0,
        description="Optional upper bound, for a run that deliberately excludes "
                    "near-identical paralogues of the seeds.",
    )
    require_family_evidence: bool = True
    family_evidence_min_identity: float | None = Field(None, ge=0.0, le=100.0)
    source: str = Field(
        "run policy",
        description="Who chose these settings. Recorded so a reviewer can ask.",
    )

    @model_validator(mode="after")
    def _bounds_consistent(self) -> "RetentionPolicy":
        lo, hi = self.min_percent_identity, self.max_percent_identity
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"min_percent_identity ({lo}) above max_percent_identity ({hi}) "
                f"retains nothing; this is a configuration error, not a filter"
            )
        return self

    def fingerprint(self) -> str:
        """Hash of the settings, compared before and after filtering.

        This is what makes "relevance is never relaxed to reach a target" a
        checked property rather than a promise in a docstring.
        """
        return sha256_obj(self.model_dump(mode="json"))


class LengthExpectation(BaseModel):
    """The length window a family member is expected to fall in.

    There is no universal protein-length rule, and :class:`FamilyTemplate`
    carries no length field, so the window must come from somewhere real.
    :meth:`from_seeds` derives it from the measured lengths of the run's own
    seeds, which is data rather than a guess, and records that in ``source``.
    A caller with a curated window passes it directly.

    The window localises QC only: a sequence outside it is flagged for review,
    never scored, and never deleted from the retrieval record.
    """

    model_config = ConfigDict(extra="forbid")

    min_length: int = Field(..., gt=0)
    max_length: int = Field(..., gt=0)
    source: str

    @model_validator(mode="after")
    def _ordered(self) -> "LengthExpectation":
        if self.min_length > self.max_length:
            raise ValueError("min_length exceeds max_length")
        if not self.source.strip():
            raise ValueError("a length window without a stated source is a guess")
        return self

    @classmethod
    def from_seeds(
        cls,
        seeds: Sequence[SeedSequence],
        lower_fraction: float = DEFAULT_FRAGMENT_LENGTH_FRACTION,
        upper_factor: float = DEFAULT_EXCESS_LENGTH_FACTOR,
    ) -> "LengthExpectation":
        """Window around the measured seed lengths.

        Widened by ``lower_fraction`` / ``upper_factor``, which are documented
        QC-localisation defaults needing per-family calibration -- a family with
        a genuinely broad length distribution will flag honest members until
        they are recalibrated.
        """
        lengths = [len(s.sequence) for s in seeds if s.sequence]
        if not lengths:
            raise ValueError("cannot derive a length window from zero seeds")
        lo = max(1, int(min(lengths) * lower_fraction))
        hi = max(lo, int(max(lengths) * upper_factor))
        return cls(
            min_length=lo, max_length=hi,
            source=(f"derived from {len(lengths)} seed length(s) "
                    f"{min(lengths)}-{max(lengths)} with lower_fraction="
                    f"{lower_fraction} and upper_factor={upper_factor}; "
                    f"needs per-family calibration"),
        )

    def classify(self, length: int) -> str | None:
        """``'fragment'``, ``'implausibly_long'`` or ``None`` when in window."""
        if length < self.min_length:
            return "fragment"
        if length > self.max_length:
            return "implausibly_long"
        return None


# ==========================================================================
# Greedy clustering
# ==========================================================================

@dataclass
class SequenceCluster:
    """One greedy identity cluster: a representative and its members."""

    cluster_id: str
    representative: str                  # sequence_sha256 of the representative
    members: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Member count, used by the downstream diversity step as a weight."""
        return len(self.members)


@dataclass
class ClusteringReport:
    """Clusters plus the honesty counters for how they were obtained."""

    clusters: list[SequenceCluster]
    assignment: dict[str, str]           # sequence_sha256 -> cluster_id
    identity_threshold: float
    n_alignments: int = 0
    n_prefiltered_pairs: int = 0
    budget_exhausted: bool = False
    method: str = "greedy_pure_python"


def cluster_sequences(
    items: Sequence[tuple[str, str]],
    identity_threshold: float = DEFAULT_CLUSTER_IDENTITY,
    *,
    kmer_prefilter: float | None = DEFAULT_KMER_PREFILTER_SIMILARITY,
    kmer_size: int = 3,
    max_alignments: int = DEFAULT_MAX_CLUSTER_ALIGNMENTS,
) -> ClusteringReport:
    """Greedy single-representative clustering of ``(key, sequence)`` pairs.

    **mmseqs2 ``easy-cluster`` is the preferred clusterer whenever it is
    installed.** This fallback is cruder in three specific ways, stated so no
    one reads its output as equivalent:

    1. it is greedy and order-dependent -- sequences are processed longest
       first, so the representative set depends on the length distribution, not
       on a global optimum;
    2. a member joins the *first* representative it passes, so a sequence near
       two cluster boundaries lands in whichever was created earlier;
    3. the k-mer screen that keeps it affordable is a heuristic, not a bound, so
       a true cluster-mate just under the screen can be missed. The number of
       pairs it skipped is returned rather than hidden.

    Identity is measured by global alignment (:func:`needleman_wunsch`) and is
    multiplied by the alignment coverage of the shorter sequence, because
    identity over a short aligned stretch is not identity between two proteins:
    a 60-residue fragment matching a 300-residue protein perfectly over its
    length is not a 100% cluster-mate.

    Clustering here **removes nothing**. It assigns a grain for the downstream
    diversity step; every input key appears in the returned assignment.
    """
    if not 0.0 < identity_threshold <= 1.0:
        raise ValueError(
            f"identity_threshold is a fraction in (0, 1], got {identity_threshold}"
        )
    ordered = sorted(items, key=lambda kv: (-len(kv[1]), kv[0]))
    kmers: dict[str, frozenset[str]] = {
        key: kmer_set(seq, kmer_size) for key, seq in ordered
    }
    clusters: list[SequenceCluster] = []
    assignment: dict[str, str] = {}
    reps: list[tuple[str, str]] = []     # (key, sequence) of each representative
    n_align = 0
    n_prefiltered = 0
    exhausted = False

    for key, seq in ordered:
        placed = False
        for rep_key, rep_seq in reps:
            if kmer_prefilter is not None:
                dist = jaccard_distance(kmers[key], kmers[rep_key])
                if dist is not None and (1.0 - dist) < kmer_prefilter:
                    n_prefiltered += 1
                    continue
            if n_align >= max_alignments:
                exhausted = True
                break
            n_align += 1
            if _pairwise_identity(seq, rep_seq) >= identity_threshold:
                cid = assignment[rep_key]
                assignment[key] = cid
                next(c for c in clusters if c.cluster_id == cid).members.append(key)
                placed = True
                break
        if not placed:
            cid = f"clu_{len(clusters) + 1:05d}"
            clusters.append(SequenceCluster(cid, key, [key]))
            assignment[key] = cid
            reps.append((key, seq))

    return ClusteringReport(
        clusters=clusters, assignment=assignment,
        identity_threshold=identity_threshold, n_alignments=n_align,
        n_prefiltered_pairs=n_prefiltered, budget_exhausted=exhausted,
    )


def _pairwise_identity(seq_a: str, seq_b: str) -> float:
    """Global identity scaled by coverage of the shorter sequence, in [0, 1]."""
    aln = needleman_wunsch(seq_a, seq_b)
    coverage = min(aln.coverage_a(), aln.coverage_b())
    return aln.identity * coverage


# ==========================================================================
# Disclosure guard
# ==========================================================================

def guard_external_submission(
    ctx: RunContext,
    queries: Sequence[SearchQuery],
    adapters: Sequence[SearchAdapter],
    databases: Sequence[SequenceDatabase],
) -> None:
    """Refuse to send an unpublished sequence to an external service.

    This is the code that makes the policy real. A comment saying "do not
    submit unpublished sequences" is not a control; a call that raises
    :class:`~eagent.errors.ApprovalRequiredError` before the first subprocess
    starts is. Unknown publication status counts as unpublished, because the
    cost of asking is an approval click and the cost of guessing wrong is an
    irreversible disclosure.

    Raises :class:`~eagent.errors.ApprovalRequiredError` when anything in this
    call would leave the machine and the ``external_sequence_submission`` gate
    has not been cleared on the task or recorded in the manifest.
    """
    leaves_machine = [a.method for a in adapters
                      if a.reaches_network or a._requests_network()]
    leaves_machine += [d.label for d in databases if d.is_remote]
    if not leaves_machine:
        return
    at_risk = sorted({q.identifier for q in queries
                      if q.kind == "sequence" and not q.is_published})
    if not at_risk:
        return
    ctx.require_approval(
        EXTERNAL_SUBMISSION_GATE,
        detail=(f"{len(at_risk)} sequence(s) with unconfirmed publication status "
                f"({', '.join(at_risk[:5])}{'...' if len(at_risk) > 5 else ''}) "
                f"would be sent to: {', '.join(sorted(set(leaves_machine)))}"),
    )


# ==========================================================================
# Retrieval rows and TSV
# ==========================================================================

@dataclass
class RetrievalRow:
    """One line of the retrieval provenance table, retained or excluded.

    Excluded rows stay in the table. A pool whose rejects are invisible cannot
    be audited: "we retrieved 2000 sequences" and "we retrieved 40000 and kept
    the 2000 that passed" are different claims, and only the table distinguishes
    them.
    """

    sequence_sha256: str
    accession: str
    length: int
    seed_accession: str
    search_method: str
    percent_identity: float | None
    query_coverage: float | None
    evalue: float | None
    bitscore: float | None
    source_database: str
    database_version: str
    family_evidence: str
    family_template_id: str | None
    n_methods: int
    n_queries: int
    all_queries: str
    status: str
    reason: str
    candidate_id: str | None = None
    cluster_id: str | None = None
    is_fragment: bool | None = None
    has_nonstandard_residues: bool = False

    columns: ClassVar[tuple[str, ...]] = (
        "candidate_id", "sequence_sha256", "accession", "length",
        "seed_accession", "search_method", "percent_identity", "query_coverage",
        "evalue", "bitscore", "source_database", "database_version",
        "family_evidence", "family_template_id", "cluster_id", "is_fragment",
        "has_nonstandard_residues", "n_methods", "n_queries", "all_queries",
        "status", "reason",
    )

    def as_row(self) -> list[str]:
        """Render in ``columns`` order, which is the order the header declares."""
        return [_tsv_cell(getattr(self, c)) for c in self.columns]


def _tsv_cell(value: Any) -> str:
    """Render one cell.

    An empty cell means "this method does not measure this", never zero. Tabs
    and newlines inside a value are replaced with spaces so a stray description
    cannot shift every later column of the row.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    return path


# ==========================================================================
# The interface
# ==========================================================================

class MineSequences(ScientificInterface):
    """Mine a candidate sequence pool from several experimentally supported seeds.

    The scientific contract of this step is narrow on purpose: it produces
    *candidates for examination*, not candidates for synthesis. Nothing here
    claims activity, family membership or mechanism. Family membership is
    asserted only by ``annotate_family``, which combines several signals; this
    step merely records which retrieval route found each sequence and whether
    that route carried any family evidence at all.
    """

    name: ClassVar[str] = "mine_sequences"
    description: ClassVar[str] = (
        "Multi-seed homology and profile search against local sequence "
        "databases, with de-duplication, greedy clustering and honest coverage "
        "reporting."
    )
    required_fields: ClassVar[tuple[str, ...]] = ("budget.initial_sequence_target",)
    required_approvals: ClassVar[tuple[str, ...]] = ()
    depends_on: ClassVar[tuple[str, ...]] = ()
    version: ClassVar[str] = "0.1.0"

    def execute(
        self,
        ctx: RunContext,
        *,
        seeds: Sequence[SeedSequence] | None = None,
        databases: Sequence[SequenceDatabase] | None = None,
        adapters: Sequence[SearchAdapter] | None = None,
        profiles: Sequence[FamilyProfile] = (),
        retention: RetentionPolicy | None = None,
        length_expectation: LengthExpectation | None = None,
        cluster_identity: float = DEFAULT_CLUSTER_IDENTITY,
        cache_dir: str | Path | None = None,
        allow_single_seed: bool = False,
        single_seed_justification: str = "",
        max_cluster_alignments: int = DEFAULT_MAX_CLUSTER_ALIGNMENTS,
        **_: Any,
    ) -> ToolResult:
        """Run every (query, adapter, database) combination and assemble the pool.

        ``seeds`` is plural and a one-element list is refused unless
        ``allow_single_seed`` is True *and* ``single_seed_justification`` says
        why; see the module docstring for the monoculture failure this prevents.

        Returns ``PARTIAL`` whenever the pool is smaller than
        ``ctx.task.budget.initial_sequence_target``, with the shortfall and its
        causes in ``data['coverage']``. It never retries with looser settings.
        """
        seeds = list(seeds or [])
        databases = list(databases or [])
        adapters = list(adapters or [])
        profiles = list(profiles or [])
        retention = retention or RetentionPolicy()
        cache_path = Path(cache_dir) if cache_dir is not None else None

        gate = self._check_seed_set(seeds, allow_single_seed, single_seed_justification)
        if gate is not None:
            return gate
        if not databases:
            return ToolResult.failure(
                self.name,
                "no sequence database supplied; this step cannot search anything "
                "and will not invent a pool",
                code="no_database",
            )
        if not adapters:
            return ToolResult.failure(
                self.name,
                "no search adapter supplied; blastp, mmseqs2 and hmmsearch are "
                "the supported backends and none was configured",
                code="no_search_adapter",
            )

        result = ToolResult(status=Status.SUCCESS)
        usable_seeds, dropped_seeds = self._partition_seeds(seeds, result)
        if not usable_seeds:
            return ToolResult.failure(
                self.name,
                "every seed lacks experimental support (best evidence is "
                "annotation or weaker); mining from annotation would produce a "
                "pool with no experimental anchor",
                code="no_supported_seed",
            )

        queries: list[SearchQuery] = [s.to_query() for s in usable_seeds]
        queries += [p.to_query() for p in profiles]

        # -- network and disclosure gates, before any subprocess starts ----
        # Network first, then disclosure: with the network off nothing can
        # leave the machine anyway, and reporting "approval required" for a
        # transfer that could not happen would send the operator to clear a
        # gate instead of to the real configuration problem. Neither ordering
        # can let data out, because both return before the first search.
        blocked = self._network_blocked(ctx, adapters, databases)
        if blocked:
            return ToolResult.failure(
                self.name,
                "offline run: " + "; ".join(blocked) +
                ". Set ExecutionPolicy.allow_network or point the run at a local "
                "database snapshot.",
                code="network_disabled",
            )
        try:
            guard_external_submission(ctx, queries, adapters, databases)
        except ApprovalRequiredError as exc:
            r = ToolResult.failure(self.name, str(exc), code="approval_required")
            r.add_next("request_approval",
                       "Unpublished sequence data would leave this machine",
                       {"gate": EXTERNAL_SUBMISSION_GATE}, requires_human=True)
            return r

        # -- local cache check ---------------------------------------------
        missing = [f"{db.label} -> {db.resolve_fasta(cache_path)}"
                   for db in databases if not db.resolve_fasta(cache_path).is_file()]
        if missing:
            return ToolResult.failure(
                self.name,
                "local sequence cache miss; this step needs these database FASTA "
                "files and will not substitute database content: "
                + "; ".join(missing),
                code="cache_miss",
            )

        # -- search ---------------------------------------------------------
        workdir = ctx.dir("mine_sequences")
        hits, search_log, search_errors = self._run_searches(
            ctx, queries, adapters, databases, workdir, retention, result
        )
        if not hits and search_errors:
            return ToolResult.failure(
                self.name,
                "every configured search failed: " + "; ".join(search_errors[:5]),
                code="search_failed",
            )

        # -- resolve residues from the local databases -----------------------
        indices, dup_ids = self._load_databases(databases, cache_path)
        for db_label, dups in dup_ids.items():
            if dups:
                result.add_flag(
                    "duplicate_database_identifiers", Severity.WARN,
                    f"{db_label}: {len(dups)} identifier(s) occur more than once "
                    f"(e.g. {', '.join(sorted(set(dups))[:3])}); hits on them were "
                    f"resolved to the first record and must be checked",
                    subject=db_label,
                )

        # -- QC and retention -------------------------------------------------
        expectation = length_expectation or LengthExpectation.from_seeds(usable_seeds)
        policy_before = retention.fingerprint()
        rows, records, unresolved = self._filter_hits(
            hits, indices, retention, expectation
        )
        if retention.fingerprint() != policy_before:
            raise FabricationGuardError(
                "the retention policy changed during filtering; a pool produced "
                "by moving the cut-off after seeing the hits is not a pool"
            )
        if unresolved:
            result.add_flag(
                "unresolved_subject_ids", Severity.WARN,
                f"{len(unresolved)} hit identifier(s) were absent from the local "
                f"database FASTA and were dropped rather than reconstructed "
                f"(e.g. {', '.join(unresolved[:3])})",
            )

        # -- clustering --------------------------------------------------------
        clustering = cluster_sequences(
            [(r.sequence_sha256, r.sequence) for r in records],
            identity_threshold=cluster_identity,
            max_alignments=max_cluster_alignments,
        ) if records else ClusteringReport([], {}, cluster_identity)
        row_by_hash = {row.sequence_sha256: row for row in rows}
        for seq_hash, cluster_id in clustering.assignment.items():
            row = row_by_hash.get(seq_hash)
            if row is not None:
                row.cluster_id = cluster_id
        if clustering.budget_exhausted:
            result.add_flag(
                "clustering_budget_exhausted", Severity.WARN,
                f"the clustering alignment budget ({max_cluster_alignments}) was "
                f"reached; the tail of the pool is in singleton clusters that may "
                f"not be singletons. Install mmseqs2 for a full clustering.",
            )

        # -- artifacts -----------------------------------------------------------
        fasta_path = self._write_pool_fasta(ctx, records)
        tsv_path = self._write_provenance_tsv(ctx, rows)
        result.artifacts.append(Artifact(
            key="candidate_sequences", path=str(fasta_path), kind="file",
            sha256=sha256_file(fasta_path), n_records=len(records),
            summary=f"{len(records)} de-duplicated candidate sequences",
        ))
        result.artifacts.append(Artifact(
            key="retrieval_provenance", path=str(tsv_path), kind="table",
            sha256=sha256_file(tsv_path), n_records=len(rows),
            summary=("one row per (sequence, best hit), retained and excluded "
                     "alike, with the method and database version that found it"),
        ))

        # -- provenance ------------------------------------------------------------
        result.provenance = Provenance(
            tool=self.name,
            tool_version=self.version,
            inputs_sha256={
                "seeds": sha256_obj([s.model_dump(mode="json") for s in usable_seeds]),
                "profiles": sha256_obj([p.model_dump(mode="json") for p in profiles]),
                "databases": sha256_obj([d.model_dump(mode="json") for d in databases]),
                "retention_policy": policy_before,
            },
            databases={d.name: d.version for d in databases},
            models={p.profile_id: p.source for p in profiles},
            parameters={
                "n_seeds": len(usable_seeds),
                "n_seeds_dropped_unsupported": len(dropped_seeds),
                "single_seed_mode": len(usable_seeds) == 1,
                "single_seed_justification": single_seed_justification,
                "search_methods": sorted({a.method for a in adapters}),
                "retention_policy": retention.model_dump(mode="json"),
                "length_expectation": expectation.model_dump(mode="json"),
                "cluster_identity": cluster_identity,
                "cluster_method": clustering.method,
                "cluster_prefilter_similarity": DEFAULT_KMER_PREFILTER_SIMILARITY,
                "max_cluster_alignments": max_cluster_alignments,
                "search_log": search_log,
            },
            random_seed=ctx.seed_for(self.name),
        )

        # -- coverage honesty --------------------------------------------------------
        self._report_coverage(ctx, result, records, rows, clustering,
                              search_errors, dropped_seeds)
        result.data.update({
            "sequence_records": [r.model_dump(mode="json") for r in records],
            "retrieval_rows": [dict(zip(RetrievalRow.columns, r.as_row()))
                               for r in rows],
            "clusters": {c.cluster_id: c.members for c in clustering.clusters},
            "cluster_assignment": clustering.assignment,
            "dropped_seeds": dropped_seeds,
            "search_errors": search_errors,
            "numbering_note": ("percent_identity is None for profile hits because "
                               "hmmsearch does not measure one; it is not 0"),
        })
        return result

    # -- stages ------------------------------------------------------------
    def _check_seed_set(self, seeds: Sequence[SeedSequence], allow_single: bool,
                        justification: str) -> ToolResult | None:
        """Refuse an empty or unjustified single-seed search."""
        if not seeds:
            return ToolResult.failure(
                self.name,
                "no seeds supplied; mining is defined relative to characterised "
                "starting points and cannot be run from nothing",
                code="no_seeds",
            )
        if len(seeds) == 1 and not allow_single:
            r = ToolResult.failure(
                self.name,
                "a single seed was supplied. One seed produces a pool of that "
                "seed's homologues, whose diversity is an illusion and whose "
                "failure modes are shared. Supply further seeds, or pass "
                "allow_single_seed=True with single_seed_justification=... to "
                "record the choice.",
                code="single_seed_not_authorised",
            )
            r.add_next("mine_sequences",
                       "Re-run with additional experimentally supported seeds",
                       {"allow_single_seed": False}, requires_human=True)
            return r
        if len(seeds) == 1 and allow_single and not justification.strip():
            return ToolResult.failure(
                self.name,
                "allow_single_seed=True requires single_seed_justification; the "
                "reason is recorded in provenance, not assumed",
                code="single_seed_unjustified",
            )
        return None

    def _partition_seeds(self, seeds: Sequence[SeedSequence],
                         result: ToolResult) -> tuple[list[SeedSequence], list[dict[str, str]]]:
        """Split seeds into usable and unsupported, flagging the second group."""
        usable: list[SeedSequence] = []
        dropped: list[dict[str, str]] = []
        for s in seeds:
            if s.has_experimental_support:
                usable.append(s)
            else:
                dropped.append({"accession": s.accession,
                                "best_evidence": s.max_strength.value})
        if dropped:
            result.add_flag(
                "seed_without_experimental_support", Severity.WARN,
                f"{len(dropped)} seed(s) dropped because their best evidence is "
                f"annotation or weaker: "
                f"{', '.join(d['accession'] for d in dropped[:5])}",
            )
        if len(usable) == 1:
            result.add_flag(
                "single_seed_mining", Severity.WARN,
                "the pool derives from one seed; treat its diversity as the "
                "diversity of one scaffold's homologues",
                subject=usable[0].accession,
            )
            result.add_uncertainty(
                "single_seed_bias",
                "Which additional characterised enzymes could anchor an "
                "independent region of this family?",
                affects=[self.name], resolvable_by="operator input or literature search",
            )
        return usable, dropped

    def _network_blocked(self, ctx: RunContext, adapters: Sequence[SearchAdapter],
                         databases: Sequence[SequenceDatabase]) -> list[str]:
        """Names of configured things that would need the network, when it is off."""
        if ctx.policy.allow_network:
            return []
        blocked = [f"adapter '{a.method}' is configured to use a remote service"
                   for a in adapters if a.reaches_network or a._requests_network()]
        blocked += [f"database '{d.label}' is marked remote" for d in databases
                    if d.is_remote]
        return blocked

    def _run_searches(
        self,
        ctx: RunContext,
        queries: Sequence[SearchQuery],
        adapters: Sequence[SearchAdapter],
        databases: Sequence[SequenceDatabase],
        workdir: Path,
        retention: RetentionPolicy,
        result: ToolResult,
    ) -> tuple[list[SearchHit], list[dict[str, Any]], list[str]]:
        """Run each compatible (query, adapter, database) triple.

        A missing binary or a failing search is recorded and the remaining
        combinations still run, because a pool from two of three methods with
        the gap stated is useful, while a pool silently missing a method is not.
        """
        hits: list[SearchHit] = []
        log: list[dict[str, Any]] = []
        errors: list[str] = []
        for adapter in adapters:
            for query in queries:
                if query.kind != adapter.accepts_query_kind:
                    continue
                for db in databases:
                    entry: dict[str, Any] = {
                        "method": adapter.method, "query": query.identifier,
                        "database": db.label,
                    }
                    try:
                        found = adapter.search(
                            query, db, workdir,
                            max_evalue=retention.max_evalue,
                            allow_network=ctx.policy.allow_network,
                        )
                    except ToolUnavailableError as exc:
                        entry.update(status="tool_unavailable", detail=str(exc))
                        errors.append(str(exc))
                        result.add_flag(
                            "search_tool_unavailable", Severity.WARN, str(exc),
                            subject=adapter.method,
                        )
                        log.append(entry)
                        break           # same binary missing for every database
                    except (SearchExecutionError, OSError, subprocess.SubprocessError) as exc:
                        entry.update(status="failed", detail=str(exc))
                        errors.append(f"{adapter.method}/{query.identifier}: {exc}")
                        result.add_flag(
                            "search_failed", Severity.WARN,
                            f"{adapter.method} on {query.identifier} vs {db.label}: {exc}",
                            subject=adapter.method,
                        )
                        log.append(entry)
                        continue
                    entry.update(status="ok", n_hits=len(found))
                    log.append(entry)
                    hits.extend(found)
                else:
                    continue
                break                   # propagate the tool-unavailable break
        return hits, log, errors

    def _load_databases(
        self, databases: Sequence[SequenceDatabase], cache_dir: Path | None,
    ) -> tuple[dict[str, dict[str, FastaEntry]], dict[str, list[str]]]:
        """Read each database FASTA once, returning per-database indices."""
        indices: dict[str, dict[str, FastaEntry]] = {}
        duplicates: dict[str, list[str]] = {}
        for db in databases:
            entries = read_fasta(db.resolve_fasta(cache_dir))
            index, dups = index_fasta(entries)
            indices[db.label] = index
            duplicates[db.label] = dups
        return indices, duplicates

    def _filter_hits(
        self,
        hits: Sequence[SearchHit],
        indices: Mapping[str, dict[str, FastaEntry]],
        retention: RetentionPolicy,
        expectation: LengthExpectation,
    ) -> tuple[list[RetrievalRow], list[SequenceRecord], list[str]]:
        """De-duplicate by sequence hash, apply QC, and build the records.

        Order of operations matters for honesty: hits are first collapsed by
        ``sequence_sha256`` (the pipeline's join key -- two accessions with
        identical residues are one protein), the *best* hit per sequence is kept
        as its retrieval provenance, and only then is the single retention
        decision made. There is exactly one place where a sequence is accepted
        or rejected, so there is nowhere for a second, looser path to appear.
        """
        best: dict[str, SearchHit] = {}
        entry_for: dict[str, FastaEntry] = {}
        methods: dict[str, set[str]] = {}
        queries: dict[str, set[str]] = {}
        unresolved: list[str] = []

        for hit in hits:
            index = indices.get(f"{hit.database_name}@{hit.database_version}", {})
            entry = index.get(hit.subject_id)
            if entry is None:
                unresolved.append(f"{hit.database_name}:{hit.subject_id}")
                continue
            key = entry.sequence_sha256
            methods.setdefault(key, set()).add(hit.search_method)
            queries.setdefault(key, set()).add(hit.query_id)
            incumbent = best.get(key)
            if incumbent is None or _hit_is_better(hit, incumbent):
                best[key] = hit
                # The accession that travels with the record is the one of the
                # best hit, not of whichever duplicate was parsed last.
                entry_for[key] = entry

        rows: list[RetrievalRow] = []
        records: list[SequenceRecord] = []
        seen_ids: dict[str, str] = {}
        for key in sorted(best):
            hit = best[key]
            entry = entry_for[key]
            seq = entry.sequence
            nonstandard = sorted(set(seq) - STANDARD_AA)
            length_issue = expectation.classify(len(seq))
            evidence, evidence_detail = _family_evidence(hit, retention)
            reasons = _retention_reasons(hit, retention, evidence)
            if nonstandard:
                reasons.append(f"nonstandard_residues:{''.join(nonstandard)}")
            if length_issue:
                reasons.append(f"{length_issue}:len={len(seq)} "
                               f"window={expectation.min_length}-{expectation.max_length}")

            row = RetrievalRow(
                sequence_sha256=key,
                accession=entry.identifier,
                length=len(seq),
                seed_accession=hit.query_id,
                search_method=hit.search_method,
                percent_identity=hit.percent_identity,
                query_coverage=hit.query_coverage,
                evalue=hit.evalue,
                bitscore=hit.bitscore,
                source_database=hit.database_name,
                database_version=hit.database_version,
                family_evidence=evidence,
                family_template_id=hit.family_template_id,
                n_methods=len(methods[key]),
                n_queries=len(queries[key]),
                all_queries=",".join(sorted(queries[key])),
                status="retained" if not reasons else "excluded",
                reason="; ".join(reasons) if reasons else evidence_detail,
                is_fragment=(length_issue == "fragment"),
                has_nonstandard_residues=bool(nonstandard),
            )
            rows.append(row)
            if reasons:
                continue
            candidate_id = _candidate_id(key)
            if candidate_id in seen_ids:
                raise FabricationGuardError(
                    f"candidate id {candidate_id} collides between two distinct "
                    f"sequences ({seen_ids[candidate_id]} and {key}); two "
                    f"proteins sharing an id would merge downstream"
                )
            seen_ids[candidate_id] = key
            row.candidate_id = candidate_id
            records.append(SequenceRecord(
                candidate_id=candidate_id,
                sequence=seq,
                sequence_sha256=key,
                accession=entry.identifier,
                source_database=hit.database_name,
                database_version=hit.database_version,
                organism=None,
                description=entry.description or None,
                seed_accession=hit.query_id,
                search_method=hit.search_method,
                percent_identity=hit.percent_identity,
                query_coverage=hit.query_coverage,
                evalue=hit.evalue,
                is_fragment=False,
                annotation_confidence=EvidenceStrength.ANNOTATION_ONLY,
            ))
        return rows, records, unresolved

    def _write_pool_fasta(self, ctx: RunContext,
                          records: Sequence[SequenceRecord]) -> Path:
        """Write ``candidate_sequences.fasta``, with retrieval facts in the header."""
        entries = [
            FastaEntry(
                r.candidate_id,
                " ".join([
                    f"accession={r.accession}",
                    f"db={r.source_database}@{r.database_version}",
                    f"seed={r.seed_accession}",
                    f"method={r.search_method}",
                    f"pid={'' if r.percent_identity is None else f'{r.percent_identity:.1f}'}",
                    f"cov={'' if r.query_coverage is None else f'{r.query_coverage:.3f}'}",
                    f"evalue={'' if r.evalue is None else f'{r.evalue:.3g}'}",
                ]),
                r.sequence,
            )
            for r in records
        ]
        return write_fasta(entries, ctx.path("mine_sequences", "candidate_sequences.fasta"))

    def _write_provenance_tsv(self, ctx: RunContext,
                              rows: Sequence[RetrievalRow]) -> Path:
        return _write_tsv(
            ctx.path("mine_sequences", "retrieval_provenance.tsv"),
            RetrievalRow.columns,
            [r.as_row() for r in rows],
        )

    def _report_coverage(
        self,
        ctx: RunContext,
        result: ToolResult,
        records: Sequence[SequenceRecord],
        rows: Sequence[RetrievalRow],
        clustering: ClusteringReport,
        search_errors: Sequence[str],
        dropped_seeds: Sequence[Mapping[str, str]],
    ) -> None:
        """State the pool size against the target, and why it fell short.

        The only honest responses to a short pool are to say so and to name the
        actionable causes. Relaxing the retention policy to reach the number
        would convert a reporting problem into a data-quality problem, so no
        branch here touches the policy.
        """
        target = int(ctx.task.budget.initial_sequence_target)
        n = len(records)
        excluded = [r for r in rows if r.status != "retained"]
        by_reason: dict[str, int] = {}
        for r in excluded:
            head = (r.reason.split(":")[0].split(";")[0] or "excluded").strip()
            by_reason[head] = by_reason.get(head, 0) + 1
        coverage = {
            "retained": n,
            "target": target,
            "fraction_of_target": (n / target) if target else None,
            "examined": len(rows),
            "excluded": len(excluded),
            "excluded_by_reason": by_reason,
            "n_clusters": len(clustering.clusters),
            "n_search_errors": len(search_errors),
            "n_seeds_dropped": len(dropped_seeds),
            "policy_relaxation_performed": False,
        }
        result.data["coverage"] = coverage

        if n == 0:
            result.status = Status.FAILED
            result.message = ("no sequence passed retrieval QC; the pool is empty")
            result.add_flag("empty_pool", Severity.BLOCKER, result.message)
        elif n < target:
            result.status = Status.PARTIAL
            causes: list[str] = []
            if search_errors:
                causes.append(f"{len(search_errors)} search(es) did not run")
            if by_reason:
                causes.append("excluded: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(by_reason.items())))
            if dropped_seeds:
                causes.append(f"{len(dropped_seeds)} seed(s) lacked experimental support")
            result.message = (
                f"pool is {n} of a target {target} "
                f"({n / target:.0%}); reported as-is. "
                + ("Causes: " + "; ".join(causes) if causes else
                   "The configured databases and seeds simply yield this many.")
            )
            result.add_flag("coverage_shortfall", Severity.WARN, result.message)
            result.add_uncertainty(
                "pool_coverage",
                f"Is a pool of {n} sufficient for this family, or should further "
                f"seeds, databases or a profile search be added? The retention "
                f"policy was not and must not be loosened to reach {target}.",
                affects=[self.name],
                resolvable_by="add seeds / add a database snapshot / add an HMM profile",
            )
            result.add_next(
                "mine_sequences",
                "Widen the *inputs*, not the thresholds: more seeds, another "
                "database snapshot, or a family HMM profile",
                {"add_profiles": True}, requires_human=True,
            )
        else:
            result.message = f"pool is {n} sequences against a target of {target}"

        result.add_next(
            "annotate_family",
            "Family membership is not established by retrieval; combine signals",
            {"n_sequences": n},
        )


def _hit_is_better(candidate: SearchHit, incumbent: SearchHit) -> bool:
    """Rank two hits on the same sequence.

    Profile hits win over pairwise hits because a profile match is family
    evidence in its own right, and that is the property the downstream step
    needs. Within a kind, the smaller e-value wins; a hit with no e-value never
    displaces one that has it, since "not reported" is not "better".
    """
    if candidate.query_kind != incumbent.query_kind:
        return candidate.query_kind == "profile"
    if candidate.evalue is None:
        return False
    if incumbent.evalue is None:
        return True
    if candidate.evalue != incumbent.evalue:
        return candidate.evalue < incumbent.evalue
    return (candidate.bitscore or 0.0) > (incumbent.bitscore or 0.0)


def _family_evidence(hit: SearchHit, retention: RetentionPolicy) -> tuple[str, str]:
    """Classify what family evidence, if any, this retrieval route carries.

    Three outcomes, kept distinct because they license different claims:

    * ``profile_hmm`` -- the sequence matched a family profile. Family-level
      evidence on its own.
    * ``seed_homology`` -- the sequence is similar to a characterised seed that
      belongs to a family template. Evidence about the *seed's* family that has
      to be confirmed per sequence; ``annotate_family`` is what confirms it.
    * ``none`` -- the route says nothing about family membership.
    """
    if hit.query_kind == "profile" and hit.family_template_id:
        return "profile_hmm", f"profile {hit.query_id}"
    if hit.family_template_id:
        floor = retention.family_evidence_min_identity
        if floor is None:
            return "seed_homology", (
                f"homology to characterised seed {hit.query_id}; family "
                f"membership unconfirmed (no family_evidence_min_identity set)"
            )
        if hit.percent_identity is not None and hit.percent_identity >= floor:
            return "seed_homology", (
                f"{hit.percent_identity:.1f}% identity to seed {hit.query_id} "
                f">= configured {floor}%"
            )
        return "none", (
            f"identity to seed {hit.query_id} is "
            f"{'unmeasured' if hit.percent_identity is None else f'{hit.percent_identity:.1f}%'}"
            f", below the configured family-evidence level of {floor}%"
        )
    return "none", f"seed {hit.query_id} carries no family template id"


def _retention_reasons(hit: SearchHit, retention: RetentionPolicy,
                       family_evidence: str) -> list[str]:
    """Reasons this hit must not enter the pool. Empty list means retained."""
    reasons: list[str] = []
    if retention.require_family_evidence and family_evidence == "none":
        reasons.append("family_evidence_absent")
    if hit.evalue is not None and hit.evalue > retention.max_evalue:
        reasons.append(f"evalue:{hit.evalue:.3g}>{retention.max_evalue:.3g}")
    if retention.min_query_coverage is not None:
        if hit.query_coverage is None:
            reasons.append("query_coverage_unmeasured")
        elif hit.query_coverage < retention.min_query_coverage:
            reasons.append(f"coverage:{hit.query_coverage:.3f}<"
                           f"{retention.min_query_coverage:.3f}")
    if retention.min_percent_identity is not None:
        if hit.percent_identity is None:
            # A profile hit has no identity. It is not "0% identity", so it is
            # not failed against an identity floor; it is simply not comparable.
            if hit.query_kind != "profile":
                reasons.append("percent_identity_unmeasured")
        elif hit.percent_identity < retention.min_percent_identity:
            reasons.append(f"identity:{hit.percent_identity:.1f}<"
                           f"{retention.min_percent_identity:.1f}")
    if retention.max_percent_identity is not None and hit.percent_identity is not None \
            and hit.percent_identity > retention.max_percent_identity:
        reasons.append(f"identity:{hit.percent_identity:.1f}>"
                       f"{retention.max_percent_identity:.1f}")
    return reasons


def _candidate_id(sequence_sha256: str) -> str:
    """Deterministic candidate id derived from the sequence hash.

    Derived rather than counted so that re-running the step, or merging two
    runs, gives the same sequence the same id instead of renumbering the pool.
    """
    digest = sequence_sha256.split(":")[-1]
    return f"cand_{digest[:12]}"
