"""Run the sequence-search adapters against real HMMER, in process.

WHY THIS EXISTS
===============
``mine_sequences`` has adapters for blastp, MMseqs2 and hmmsearch, and the
roadmap records that **none of them had ever been run against the real
program**: every test drives them through a fake runner that writes output in
the documented shape. That is a reasonable test seam and a poor guarantee. A
fake's output is written from the format documentation by the same person who
wrote the parser, so the two agree by construction and agree equally about any
column the documentation describes loosely. The failure mode it cannot catch is
the only one that matters: the real tool writing something the parser does not
expect.

``pyhmmer`` closes that for one of the three without a binary, root or network.
:class:`PyhmmerRunner` is a :class:`~eagent.tools.mine_sequences.CommandRunner`
that takes the argv :class:`~eagent.tools.mine_sequences.HmmsearchAdapter`
builds, runs the search in process, and writes **HMMER's own**
``--domtblout``. The adapter and its parser then run unchanged on real output.

WHAT IT IS NOT
==============
* **Not a substitute for the binary.** A deployment with hmmsearch on PATH
  should use :class:`~eagent.tools.mine_sequences.SubprocessRunner`; this
  exists so that an environment without it can still run a real search, and so
  that the parser is held to real output in CI.
* **Not a permissive argv interpreter.** It implements exactly the invocation
  the shipped adapter builds. An option it does not implement is **refused**,
  with a non-zero return and a message naming itself, rather than being
  silently dropped -- a dropped ``-E`` would change which hits came back and
  the hits would look just as plausible.
* **Not a sequence database.** It searches the FASTA the caller names. The
  mining seam is still open for want of a database to search, which is a
  reachability and licensing question, not this module's.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import EAgentError, ToolUnavailableError
from ..science.alignment_similarity import hmmer_version
from .mine_sequences import CommandResult

__all__ = [
    "PyhmmerError",
    "PyhmmerRunner",
    "build_profile_hmm",
    "hmmer_version",
    "pyhmmer_available",
]


class PyhmmerError(EAgentError):
    """pyhmmer was asked for something it cannot honestly do here."""


def pyhmmer_available() -> bool:
    """Whether ``pyhmmer`` can be imported. Guarded: it is the ``search`` extra."""
    try:
        import pyhmmer  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover - the absent-extra leg
        return False
    return True


def _require() -> None:
    if not pyhmmer_available():
        raise ToolUnavailableError(
            "hmmsearch (pyhmmer)",
            hint="pip install 'eagent[search]'; pyhmmer ships HMMER3 as a "
                 "library, so no external binary is needed")


def build_profile_hmm(sequences: Sequence[tuple[str, str]], out_path: str | Path,
                      *, name: str, seed: int = 42) -> dict[str, Any]:
    """Write an HMM for a family, from one seed sequence or from an alignment.

    ``sequences`` are ``(identifier, residues)`` pairs. One pair builds a
    single-sequence profile, which is what ``phmmer`` does internally. Several
    pairs are treated as a **multiple alignment** and are required to be one:
    equal lengths, with ``-`` or ``.`` for gaps. Unaligned sequences are
    refused rather than aligned here, because an alignment produced as a side
    effect of building a profile is an alignment nobody inspected, and the
    profile would carry its mistakes with no record that it had been made.

    Returns a provenance record: what was built, from how many sequences, with
    which builder settings, and the HMMER version that wrote the file.
    """
    _require()
    from pyhmmer.easel import Alphabet, TextMSA, TextSequence
    from pyhmmer.plan7 import Background, Builder

    rows = [(i, s.strip().upper()) for i, s in sequences]
    if not rows:
        raise PyhmmerError("no sequences to build a profile from")
    for ident, residues in rows:
        if not residues:
            raise PyhmmerError(f"{ident!r} has no residues")
    alphabet = Alphabet.amino()
    builder = Builder(alphabet, seed=seed)
    background = Background(alphabet)
    out_path = Path(out_path)

    if len(rows) == 1:
        ident, residues = rows[0]
        if set("-.") & set(residues):
            raise PyhmmerError(
                "a single sequence carrying gap characters is an alignment of "
                "one; strip the gaps or supply the whole alignment")
        seed_seq = TextSequence(name=ident.encode(), sequence=residues)
        hmm, _p, _o = builder.build(seed_seq.digitize(alphabet), background)
        built_from = "single sequence"
    else:
        widths = {len(r) for _, r in rows}
        if len(widths) != 1:
            raise PyhmmerError(
                f"these {len(rows)} sequences have {len(widths)} different "
                f"lengths, so they are not an alignment; this module does not "
                f"align them -- supply one seed sequence, or an aligned set")
        msa = TextMSA(
            name=name.encode(),
            sequences=[TextSequence(name=i.encode(), sequence=r)
                       for i, r in rows])
        hmm, _p, _o = builder.build_msa(msa.digitize(alphabet), background)
        built_from = f"alignment of {len(rows)} sequences, width {widths.pop()}"

    hmm.name = name.encode()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    hmm.write(buffer)
    payload = buffer.getvalue()
    out_path.write_bytes(payload)
    return {
        "path": str(out_path),
        "name": name,
        "built_from": built_from,
        "n_sequences": len(rows),
        "profile_length": int(hmm.M),
        "hmmer_version": payload.split(b"\n", 1)[0].decode("utf-8", "replace").strip(),
        "builder_settings": {"score_matrix": str(builder.score_matrix),
                             "gap_open": builder.popen,
                             "gap_extend": builder.pextend,
                             "seed": seed},
    }


@dataclass(frozen=True)
class _Invocation:
    """The parts of an hmmsearch command line this runner implements."""

    profile: Path
    target: Path
    domtblout: Path
    max_evalue: float
    cpus: int


#: Options the shipped adapter passes that this runner honours. ``-o`` is
#: accepted and its value discarded: the adapter sends the human-readable
#: report to /dev/null, and nothing reads it.
_WITH_VALUE = {"--domtblout", "-E", "--cpu", "-o"}


class PyhmmerRunner:
    """A ``CommandRunner`` that executes an ``hmmsearch`` argv with pyhmmer.

    Pass it where :class:`~eagent.tools.mine_sequences.SubprocessRunner` would
    go. The adapter's own driver, argv builder, failure handling and
    ``--domtblout`` parser all run unchanged; only the process boundary is
    replaced, and what lands in the output file is what HMMER wrote.

    ``executable_finder`` on the adapter still has to answer: the adapter
    resolves ``hmmsearch`` on PATH before running anything. With no binary
    installed, pass ``executable_finder=lambda name: "hmmsearch (pyhmmer)"`` so
    the resolution step records honestly what is going to run.
    """

    def __init__(self, *, cpus: int | None = None) -> None:
        _require()
        self.cpus = cpus
        #: Every invocation this runner has executed, for a test or an audit to
        #: read back. A runner that cannot say what it ran is not a seam.
        self.invocations: list[_Invocation] = []

    def __call__(self, argv: Sequence[str], *, cwd: Path | None = None,
                 timeout: float | None = None) -> CommandResult:
        args = tuple(argv)
        try:
            invocation = self._parse(args, cwd)
        except PyhmmerError as exc:
            return CommandResult(args, 2, "", f"PyhmmerRunner: {exc}")
        for path, what in ((invocation.profile, "profile"),
                           (invocation.target, "sequence database")):
            if not path.exists():
                return CommandResult(
                    args, 1, "",
                    f"PyhmmerRunner: no such {what}: {path}")
        try:
            written = self._search(invocation)
        except PyhmmerError as exc:
            return CommandResult(args, 1, "", f"PyhmmerRunner: {exc}")
        self.invocations.append(invocation)
        return CommandResult(args, 0, f"{written} target(s) reported", "")

    # -- internals ---------------------------------------------------------
    def _parse(self, args: Sequence[str], cwd: Path | None) -> _Invocation:
        base = Path(cwd) if cwd else Path.cwd()
        options: dict[str, str] = {}
        positional: list[str] = []
        rest = list(args[1:])  # args[0] is the resolved executable
        while rest:
            token = rest.pop(0)
            if token in _WITH_VALUE:
                if not rest:
                    raise PyhmmerError(f"option {token} needs a value")
                options[token] = rest.pop(0)
            elif token.startswith("-"):
                raise PyhmmerError(
                    f"option {token} is not implemented here; this runner "
                    f"implements the invocation HmmsearchAdapter builds and "
                    f"refuses the rest rather than ignoring it, because a "
                    f"dropped option changes which hits come back")
            else:
                positional.append(token)
        if len(positional) != 2:
            raise PyhmmerError(
                f"expected a profile and a target, got {len(positional)} "
                f"positional argument(s)")
        if "--domtblout" not in options:
            raise PyhmmerError(
                "no --domtblout; this runner writes the per-domain table and "
                "nothing else, because that is what the adapter parses")
        try:
            max_evalue = float(options.get("-E", "10.0"))
        except ValueError as exc:
            raise PyhmmerError(f"-E {options['-E']!r} is not a number") from exc
        try:
            cpus = int(options.get("--cpu", "1"))
        except ValueError as exc:
            raise PyhmmerError(f"--cpu {options['--cpu']!r} is not a number") from exc
        return _Invocation(
            profile=_resolve(positional[0], base),
            target=_resolve(positional[1], base),
            domtblout=_resolve(options["--domtblout"], base),
            max_evalue=max_evalue,
            cpus=self.cpus if self.cpus is not None else cpus)

    def _search(self, invocation: _Invocation) -> int:
        from pyhmmer.easel import Alphabet, SequenceFile
        from pyhmmer.hmmer import hmmsearch
        from pyhmmer.plan7 import HMMFile

        alphabet = Alphabet.amino()
        with HMMFile(invocation.profile) as handle:
            profiles = list(handle)
        if not profiles:
            raise PyhmmerError(f"{invocation.profile} holds no profile")
        with SequenceFile(invocation.target, digital=True,
                          alphabet=alphabet) as handle:
            targets = handle.read_block()
        invocation.domtblout.parent.mkdir(parents=True, exist_ok=True)
        reported = 0
        with invocation.domtblout.open("wb") as out:
            for index, hits in enumerate(
                    hmmsearch(profiles, targets, E=invocation.max_evalue,
                              cpus=invocation.cpus)):
                # HMMER writes the table; nothing here reformats it. The header
                # goes in once, as hmmsearch does for a multi-profile file.
                hits.write(out, format="domains", header=index == 0)
                reported += len(hits)
        return reported


def _resolve(token: str, base: Path) -> Path:
    path = Path(token)
    return path if path.is_absolute() else base / path
