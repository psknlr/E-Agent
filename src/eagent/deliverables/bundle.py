"""Assembling the research package for one run, with nothing silently absent.

A deliverables bundle is read by people who were not in the room: a reviewer
six months later, a second site reproducing the round, a curator asked why a
particular gene was ordered. Three failure modes make such a package worse
than useless, and every rule in this module exists for one of them.

*A package that looks complete and is not.* A directory with fourteen of
eighteen files reads as a finished package; nobody counts. So the bundle
manifest declares the full standard set up front and records a status for
every item, and :attr:`BundleManifest.complete` is false while any item is
missing or partial. A missing item is written down with the reason and with
what a curator must supply -- it is never omitted, because an omission is
indistinguishable from an item that was never meant to exist.

*A derived file standing in for the record it was derived from.* A structure
converted to PDB has lost the things mmCIF carries: full chain and entity
identifiers, author versus label numbering, and anything past residue 9999 or
chain Z. The mmCIF is therefore the primary record, a PDB beside it is kept
as a *validated extra* (see :func:`_validate_derived_structure`), and a
directory holding only converted files is reported as missing its primary
records. The same applies to confidence: a per-residue pLDDT/PAE file is the
record, and a screenshot of it or a single averaged number is not a
substitute -- :data:`CONFIDENCE_POLICY` rejects an image-only directory by
name rather than accepting it as "present".

*A filename that states a number the file does not contain.* A batch file
called ``selected_batch_96.csv`` holding 71 constructs becomes "we screened
96" at the next meeting. :func:`_count_batch_constructs` reads the actual row
count and the file is named from it, with the renaming recorded.

:func:`verify_bundle` closes the loop: every declared file must exist and
hash to what the manifest recorded, and a file inside a declared directory
that the manifest never saw is reported too, because an unrecorded file in a
traceable package is either a tampered one or a provenance gap.

Nothing here computes a score, a ranking or a quality number. The report it
generates cites an artifact for every quantitative claim it makes, and when
the artifact is absent the report says the number is unavailable instead of
estimating it.
"""

from __future__ import annotations

import csv
import enum
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..provenance import RunManifest, sha256_file, utc_now

__all__ = [
    "BATCH_ITEM_NAME",
    "BUNDLE_FORMAT",
    "BUNDLE_ITEMS",
    "BUNDLE_MANIFEST_NAME",
    "BundleEntry",
    "BundleItem",
    "BundleManifest",
    "BundleResult",
    "BundleVerification",
    "CONFIDENCE_POLICY",
    "DirectoryPolicy",
    "FileEntry",
    "ItemStatus",
    "RESEARCH_REPORT_NAME",
    "RUN_MANIFEST_NAME",
    "STRUCTURE_POLICY",
    "assemble_bundle",
    "bundle_summary_lines",
    "render_research_report",
    "verify_bundle",
]

#: Name of the manifest inside the bundle. Fixed, because
#: :func:`verify_bundle` is the only thing a recipient can run against a
#: package they did not build, and it has to know where to look.
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"

#: Name of the generated narrative.
RESEARCH_REPORT_NAME = "research_report.md"

#: Name of the run manifest, both in the run directory and in the bundle.
RUN_MANIFEST_NAME = "run_manifest.json"

#: Bundle format identifier, written into the manifest so a future reader can
#: tell which rules the package was assembled under.
BUNDLE_FORMAT = "eagent-research-package/1"

#: Declared name of the batch order form. The file is written under the count
#: it actually holds, so the declared name is only how it is looked up.
BATCH_ITEM_NAME = "selected_batch_96.csv"


class ItemStatus(str, enum.Enum):
    """What happened to one declared deliverable.

    ``PARTIAL`` is kept apart from ``PRESENT`` because a structures directory
    that holds only converted PDB files, or a confidence directory holding
    only pictures, is materially different from one holding the records --
    and folding it into "present" is exactly how a package comes to look
    complete.
    """

    PRESENT = "present"
    PARTIAL = "partial"
    MISSING = "missing"

    @property
    def counts_as_delivered(self) -> bool:
        return self is ItemStatus.PRESENT


@dataclass(frozen=True)
class DirectoryPolicy:
    """Which files in a directory are the record, and which only accompany it.

    ``primary_suffixes`` are the files the claims rest on. ``derived_suffixes``
    are conversions kept for convenience and validated against a primary.
    ``rejected_substitute_suffixes`` are formats that are sometimes offered in
    place of the record -- a PNG of a pLDDT plot, a PDF of a figure -- and that
    must never make a directory count as delivered.
    """

    primary_suffixes: tuple[str, ...] = ()
    derived_suffixes: tuple[str, ...] = ()
    rejected_substitute_suffixes: tuple[str, ...] = ()
    primary_required: bool = False
    primary_description: str = ""
    substitute_refusal: str = ""

    def role_of(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in self.primary_suffixes:
            return "primary"
        if suffix in self.derived_suffixes:
            return "derived"
        if suffix in self.rejected_substitute_suffixes:
            return "rejected_substitute"
        return "companion"


#: Coordinates. mmCIF is the record; a PDB beside it is a validated extra.
STRUCTURE_POLICY = DirectoryPolicy(
    primary_suffixes=(".cif", ".mmcif"),
    derived_suffixes=(".pdb", ".ent"),
    rejected_substitute_suffixes=(".png", ".jpg", ".jpeg", ".svg", ".gif"),
    primary_required=True,
    primary_description="mmCIF coordinate files",
    substitute_refusal=(
        "a rendered image is a picture of a structure, not the coordinates "
        "every distance in this run was measured on"),
)

#: Predictor confidence. The per-residue/per-pair files are the record.
CONFIDENCE_POLICY = DirectoryPolicy(
    primary_suffixes=(".json", ".tsv", ".csv", ".txt", ".cif", ".npz"),
    derived_suffixes=(),
    rejected_substitute_suffixes=(".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf"),
    primary_required=True,
    primary_description="per-structure confidence files (pLDDT/PAE)",
    substitute_refusal=(
        "a screenshot of a confidence plot, or a single averaged score, cannot "
        "be re-read per residue; the pocket is what matters and a chain mean "
        "hides a disordered loop"),
)

#: Free-form supporting directory: everything in it is kept, nothing in it is
#: the sole basis of a claim, so no primary format is demanded.
SUPPORTING_POLICY = DirectoryPolicy()


@dataclass(frozen=True)
class BundleItem:
    """One deliverable of the standard package.

    ``artifact_keys`` are the envelope keys the run would have produced it
    under; ``source_names`` are the paths it lands at in a run directory. Both
    are listed because a bundle must be assemblable from a run directory alone
    -- a manifest that was never written is itself one of the things the
    bundle has to report.
    """

    name: str
    kind: str                            # "file" | "directory" | "generated"
    why: str
    artifact_keys: tuple[str, ...] = ()
    source_names: tuple[str, ...] = ()
    directory_policy: DirectoryPolicy = SUPPORTING_POLICY
    count_name_template: str | None = None
    curator_note: str = ""

    @property
    def is_directory(self) -> bool:
        return self.kind == "directory"

    @property
    def is_generated(self) -> bool:
        return self.kind == "generated"


#: The standard research package, declared in full so that absence is
#: reportable. Order is the order a reader walks it: what the task was, what
#: was found, what was built, what was measured, what was ordered, what came
#: back, and the record that ties it together.
BUNDLE_ITEMS: tuple[BundleItem, ...] = (
    BundleItem(
        name="reaction_spec.yaml", kind="file",
        artifact_keys=("reaction_spec",),
        source_names=("reaction_spec.yaml", "normalize_reaction/reaction_spec.yaml"),
        why=("the chemistry the whole package is about, with the fields that "
             "are still unresolved; without it nothing downstream can be read "
             "as being about a particular molecule"),
        curator_note=("run the normalize_reaction step, or supply the operator's "
                      "confirmed substrate and product structures"),
    ),
    BundleItem(
        name="evidence_records.jsonl", kind="file",
        artifact_keys=("evidence_records",),
        source_names=("evidence_records.jsonl",
                      "retrieve_evidence/evidence_records.jsonl"),
        why=("one row per retrieved record, with the source, the strength and "
             "the licence it was carried under"),
        curator_note="run the retrieve_evidence step",
    ),
    BundleItem(
        name="candidate_sequences.fasta", kind="file",
        artifact_keys=("candidate_sequences",),
        source_names=("mine_sequences/candidate_sequences.fasta",
                      "candidate_sequences.fasta"),
        why="the sequences every later claim is about",
        curator_note="run the mine_sequences step",
    ),
    BundleItem(
        name="sequence_annotations.tsv", kind="file",
        artifact_keys=("sequence_annotations",),
        source_names=("annotate_family/sequence_annotations.tsv",
                      "sequence_annotations.tsv"),
        why="the family call per sequence and the signals it rests on",
        curator_note="run the annotate_family step",
    ),
    BundleItem(
        name="family_analysis", kind="directory",
        artifact_keys=("family_analysis_dir",),
        source_names=("annotate_family/family_analysis", "family_analysis"),
        why=("alignments, motif hits and cluster assignments: the working that "
             "a family call can be rechecked from"),
        directory_policy=SUPPORTING_POLICY,
        curator_note="run the annotate_family step",
    ),
    BundleItem(
        name="structures", kind="directory",
        artifact_keys=("structures_dir",),
        source_names=("prepare_structures/structures", "structures"),
        why=("the coordinates every geometric measurement was made on, stored "
             "as mmCIF"),
        directory_policy=STRUCTURE_POLICY,
        curator_note=("run the prepare_structures step, or deposit the chosen "
                      "mmCIF files"),
    ),
    BundleItem(
        name="complexes", kind="directory",
        artifact_keys=("complexes_dir",),
        source_names=("model_complexes/complexes", "complexes"),
        why=("the enzyme-substrate-cofactor poses, with the ligand coordinates "
             "retained rather than summarised"),
        directory_policy=STRUCTURE_POLICY,
        curator_note="run the model_complexes step",
    ),
    BundleItem(
        name="confidence_metrics", kind="directory",
        artifact_keys=("confidence_metrics_dir",),
        source_names=("prepare_structures/confidence_metrics", "confidence_metrics"),
        why=("the predictor's own confidence output, per structure, so a "
             "pocket-local figure can be recomputed instead of quoted"),
        directory_policy=CONFIDENCE_POLICY,
        curator_note=("copy the predictor's confidence files; a plot image or a "
                      "single mean is not a substitute"),
    ),
    BundleItem(
        name="residue_atom_mapping.tsv", kind="file",
        artifact_keys=("residue_atom_mapping",),
        source_names=("prepare_structures/residue_atom_mapping.tsv",
                      "residue_atom_mapping.tsv"),
        why=("candidate index to author numbering, residue by residue: the "
             "file that makes a mutation refer to the residue it was meant to"),
        curator_note="run the prepare_structures step",
    ),
    BundleItem(
        name="catalytic_geometry.tsv", kind="file",
        artifact_keys=("catalytic_geometry",),
        source_names=("evaluate_catalysis/catalytic_geometry.tsv",
                      "catalytic_geometry.tsv"),
        why=("every measurement against every template window, with which "
             "windows were restrained during modelling and so are not "
             "independent evidence"),
        curator_note="run the evaluate_catalysis step",
    ),
    BundleItem(
        name="candidate_scorecards.tsv", kind="file",
        artifact_keys=("candidate_scorecards",),
        source_names=("evaluate_catalysis/candidate_scorecards.tsv",
                      "candidate_scorecards.tsv"),
        why=("per-candidate gates and ordinal axis levels; there is no total "
             "column and none may be derived from these"),
        curator_note="run the evaluate_catalysis step",
    ),
    BundleItem(
        name="candidate_explanations.txt", kind="file",
        artifact_keys=("candidate_explanations",),
        source_names=("evaluate_catalysis/candidate_explanations.txt",
                      "candidate_explanations.txt"),
        why=("the per-candidate justification a reviewer reads instead of a "
             "score; the research report is built from these"),
        curator_note="run the evaluate_catalysis step",
    ),
    BundleItem(
        name=BATCH_ITEM_NAME, kind="file",
        artifact_keys=("selected_batch",),
        source_names=("select_batch/selected_batch_96.csv",),
        count_name_template="selected_batch_{n}.csv",
        why=("the order form: one row per construct, with the role the slot "
             "fills and why it was spent"),
        curator_note="run the select_batch step behind a recorded batch approval",
    ),
    BundleItem(
        name="mutation_proposals.tsv", kind="file",
        artifact_keys=("mutation_proposals",),
        source_names=("propose_mutations/mutation_proposals.tsv",
                      "mutation_proposals.tsv"),
        why=("each proposed substitution with the axis it is expected to move "
             "and what it may cost, stated before testing"),
        curator_note="run the propose_mutations step",
    ),
    BundleItem(
        name="experiment_plan.yaml", kind="file",
        artifact_keys=("experiment_plan",),
        source_names=("select_batch/experiment_plan.yaml", "experiment_plan.yaml"),
        why=("roles, quotas, controls and the measurement footprint, with the "
             "positive criterion recorded before any data exist"),
        curator_note="run the select_batch step",
    ),
    BundleItem(
        name="assay_results_template.csv", kind="file",
        artifact_keys=("assay_results_template",),
        source_names=("select_batch/assay_results_template.csv",
                      "assay_results_template.csv"),
        why="one row per well, in the columns the ingest step reads back",
        curator_note="run the select_batch step",
    ),
    BundleItem(
        name=RUN_MANIFEST_NAME, kind="file",
        artifact_keys=(),
        source_names=(RUN_MANIFEST_NAME,),
        why=("input hashes, database and model versions, seeds, costs and "
             "approvals: the file a claim is traced back through"),
        curator_note=("the run must write its manifest; without it nothing in "
                      "this package can be recomputed"),
    ),
    BundleItem(
        name=RESEARCH_REPORT_NAME, kind="generated",
        why=("the narrative, generated from the artifacts in this package; "
             "every quantitative claim in it names the file it came from"),
    ),
)


# ==========================================================================
# manifest records
# ==========================================================================

@dataclass(frozen=True)
class FileEntry:
    """One file inside the bundle, with the hash verification checks."""

    path: str                 # relative to the bundle root
    sha256: str
    n_bytes: int
    role: str = "record"      # record | primary | derived | companion | rejected_substitute
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256,
                "n_bytes": self.n_bytes, "role": self.role, "note": self.note}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FileEntry":
        return cls(path=str(raw["path"]), sha256=str(raw["sha256"]),
                   n_bytes=int(raw.get("n_bytes", 0)),
                   role=str(raw.get("role", "record")),
                   note=str(raw.get("note", "")))


@dataclass
class BundleEntry:
    """What became of one declared deliverable."""

    name: str                      # the name it was declared under
    kind: str
    status: ItemStatus
    why: str
    path: str | None = None        # relative path inside the bundle, as written
    source: str | None = None      # where it was copied from
    reason: str = ""               # why it is missing or partial
    curator_note: str = ""
    n_files: int | None = None
    n_records: int | None = None
    files: list[FileEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def renamed(self) -> bool:
        return bool(self.path) and Path(str(self.path)).name != self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status.value,
            "why": self.why,
            "path": self.path,
            "renamed_from": self.name if self.renamed else None,
            "source": self.source,
            "reason": self.reason,
            "curator_note": self.curator_note,
            "n_files": self.n_files,
            "n_records": self.n_records,
            "files": [f.to_dict() for f in self.files],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BundleEntry":
        return cls(
            name=str(raw["name"]), kind=str(raw.get("kind", "file")),
            status=ItemStatus(raw.get("status", ItemStatus.MISSING.value)),
            why=str(raw.get("why", "")), path=raw.get("path"),
            source=raw.get("source"), reason=str(raw.get("reason", "")),
            curator_note=str(raw.get("curator_note", "")),
            n_files=raw.get("n_files"), n_records=raw.get("n_records"),
            files=[FileEntry.from_dict(f) for f in raw.get("files", [])],
            notes=[str(n) for n in raw.get("notes", [])],
        )


@dataclass
class BundleManifest:
    """The bundle's own index: every declared item, present or not."""

    run_id: str | None
    task_id: str | None
    source_run_dir: str
    created_at: str = field(default_factory=utc_now)
    bundle_format: str = BUNDLE_FORMAT
    entries: list[BundleEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- queries -----------------------------------------------------------
    def entry(self, name: str) -> BundleEntry | None:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    @property
    def missing(self) -> list[BundleEntry]:
        return [e for e in self.entries if e.status is ItemStatus.MISSING]

    @property
    def partial(self) -> list[BundleEntry]:
        return [e for e in self.entries if e.status is ItemStatus.PARTIAL]

    @property
    def present(self) -> list[BundleEntry]:
        return [e for e in self.entries if e.status is ItemStatus.PRESENT]

    @property
    def complete(self) -> bool:
        """True only when every declared item is present in full.

        A partial item counts against completeness. A directory of converted
        structures with no mmCIF, or of confidence screenshots with no
        confidence files, is a package that cannot support the claims made
        from it, and calling that "complete" is the failure this field exists
        to prevent.
        """
        return bool(self.entries) and not self.missing and not self.partial

    def counts(self) -> dict[str, int]:
        return {"declared": len(self.entries), "present": len(self.present),
                "partial": len(self.partial), "missing": len(self.missing)}

    # -- io ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_format": self.bundle_format,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "source_run_dir": self.source_run_dir,
            "complete": self.complete,
            "counts": self.counts(),
            "missing": [
                {"name": e.name, "reason": e.reason,
                 "curator_must_supply": e.curator_note, "why_it_matters": e.why}
                for e in self.missing
            ],
            "partial": [
                {"name": e.name, "reason": e.reason,
                 "curator_must_supply": e.curator_note}
                for e in self.partial
            ],
            "items": [e.to_dict() for e in self.entries],
            "notes": list(self.notes),
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False,
                                default=str), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "BundleManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(
            run_id=raw.get("run_id"), task_id=raw.get("task_id"),
            source_run_dir=str(raw.get("source_run_dir", "")),
            created_at=str(raw.get("created_at", "")),
            bundle_format=str(raw.get("bundle_format", BUNDLE_FORMAT)),
            notes=[str(n) for n in raw.get("notes", [])],
        )
        manifest.entries = [BundleEntry.from_dict(e) for e in raw.get("items", [])]
        return manifest


@dataclass
class BundleResult:
    """What :func:`assemble_bundle` produced."""

    bundle_dir: Path
    manifest: BundleManifest
    manifest_path: Path
    report_path: Path | None = None

    @property
    def complete(self) -> bool:
        return self.manifest.complete

    @property
    def missing_names(self) -> list[str]:
        return [e.name for e in self.manifest.missing]

    @property
    def partial_names(self) -> list[str]:
        return [e.name for e in self.manifest.partial]


@dataclass
class BundleVerification:
    """Result of re-checking a bundle against its own manifest."""

    bundle_dir: Path
    problems: list[str] = field(default_factory=list)
    n_files_checked: int = 0
    n_items_checked: int = 0
    complete: bool = False

    @property
    def ok(self) -> bool:
        """No file is absent, altered, or present without being recorded."""
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {"bundle_dir": str(self.bundle_dir), "ok": self.ok,
                "complete": self.complete, "problems": list(self.problems),
                "n_files_checked": self.n_files_checked,
                "n_items_checked": self.n_items_checked}


# ==========================================================================
# counting helpers -- a number in this package always came from a file
# ==========================================================================

def _count_fasta(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return sum(1 for line in text.splitlines() if line.startswith(">"))


def _count_jsonl(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return sum(1 for line in text.splitlines() if line.strip())


def _read_delimited(path: Path) -> tuple[list[str], list[list[str]]] | None:
    """Header and data rows of a CSV/TSV, or ``None`` when it cannot be read."""
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.reader(fh, delimiter=delimiter) if r]
    except (OSError, csv.Error, UnicodeDecodeError):
        return None
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _count_delimited(path: Path) -> int | None:
    parsed = _read_delimited(path)
    if parsed is None:
        return None
    return len(parsed[1])


def _count_records(path: Path) -> int | None:
    """Rows in a file whose format states how to count them, else ``None``.

    ``None`` is returned rather than a guess: a record count printed in a
    report must be re-derivable from the file, and "number of lines" is not a
    record count for a format whose records span lines.
    """
    suffix = path.suffix.lower()
    if suffix in (".fasta", ".fa", ".faa"):
        return _count_fasta(path)
    if suffix == ".jsonl":
        return _count_jsonl(path)
    if suffix in (".tsv", ".csv", ".tab"):
        return _count_delimited(path)
    return None


def _count_batch_constructs(path: Path) -> tuple[int | None, int | None]:
    """``(constructs, rows)`` in a batch order form.

    Constructs are the rows that are not controls, which is what the filename
    must state: a control well is not a gene that was ordered under a
    candidate's name. Returns ``(None, None)`` when the file cannot be parsed,
    so the caller reports "unknown" instead of naming the file after a count
    it did not read.
    """
    parsed = _read_delimited(path)
    if parsed is None:
        return None, None
    header, rows = parsed
    if not header:
        return None, 0
    lowered = [h.strip().lower() for h in header]
    if "kind" in lowered:
        index = lowered.index("kind")
        constructs = sum(1 for r in rows
                         if len(r) > index and r[index].strip().lower() != "control")
        return constructs, len(rows)
    if "role" in lowered:
        index = lowered.index("role")
        constructs = sum(1 for r in rows
                         if len(r) > index and r[index].strip().lower() != "control")
        return constructs, len(rows)
    return len(rows), len(rows)


# ==========================================================================
# source resolution
# ==========================================================================

def _artifact_paths(manifest: RunManifest | None,
                    keys: Sequence[str]) -> list[tuple[str, str]]:
    """``(key, path)`` for the latest recording of each key, newest step first."""
    if manifest is None or not keys:
        return []
    wanted = set(keys)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record in reversed(manifest.steps):
        for artifact in record.artifacts:
            key = str(artifact.get("key") or "")
            path = artifact.get("path")
            if key in wanted and key not in seen and path:
                seen.add(key)
                found.append((key, str(path)))
    return found


def _recorded_path_candidates(raw: str, run_dir: Path) -> list[Path]:
    """Where a path recorded in a manifest might actually be.

    A manifest records the path a step wrote, which was relative to whatever
    directory the run was started from. Resolving it only against the run
    directory would report a file that is plainly there as absent -- and
    "artifact missing" is a far more alarming sentence than it deserves.

    Run-directory-anchored readings are tried before the current working
    directory, because a bundler invoked from somewhere else could otherwise
    match a same-named file belonging to a different run, and a package that
    quietly contains another run's artifact is worse than one that is short
    of it.
    """
    path = Path(raw)
    if path.is_absolute():
        return [path]
    options = [run_dir / path]
    parts = path.parts
    if parts and parts[0] == run_dir.name and len(parts) > 1:
        options.append(run_dir / Path(*parts[1:]))
    options.append(path)
    return options


def _resolve_source(item: BundleItem, run_dir: Path,
                    manifest: RunManifest | None) -> tuple[Path | None, str, list[str]]:
    """Locate one deliverable. Returns ``(path, how, notes)``.

    An artifact the manifest recorded but that is not on disk is a distinct
    and serious state -- the run believed it wrote the file -- so it is noted
    explicitly rather than silently falling through to "not produced".
    """
    notes: list[str] = []
    for key, raw in _artifact_paths(manifest, item.artifact_keys):
        for candidate in _recorded_path_candidates(raw, run_dir):
            if candidate.exists():
                return candidate, f"run manifest artifact '{key}' at {raw}", notes
        notes.append(
            f"the run manifest records artifact '{key}' at {raw}, which is not "
            f"on disk at any path this bundler could resolve it to; the "
            f"recorded path was not used")
    for name in item.source_names:
        candidate = run_dir / name
        if candidate.exists():
            return candidate, f"{name} in the run directory", notes
    if item.count_name_template is not None:
        # The producing step names the file from its own count, so the declared
        # name is only one of the possibilities; find whichever was written.
        pattern = item.count_name_template.replace("{n}", "*")
        for parent in {Path(n).parent for n in item.source_names} or {Path(".")}:
            matches = sorted((run_dir / parent).glob(pattern)) \
                if (run_dir / parent).is_dir() else []
            if matches:
                return matches[0], f"{matches[0].relative_to(run_dir)} in the " \
                                   f"run directory", notes
    return None, "", notes


# ==========================================================================
# copying
# ==========================================================================

def _copy_file(src: Path, dest: Path) -> FileEntry:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return FileEntry(path="", sha256=sha256_file(dest), n_bytes=dest.stat().st_size)


def _validate_derived_structure(primary: Path | None, derived: Path) -> str:
    """Check a converted coordinate file against the record it came from.

    A PDB is kept only as an extra, and an extra that disagrees with the
    record is worse than no extra at all: it will be opened by somebody and
    measured. The check is deliberately coarse -- atom counts -- because that
    is what a conversion silently loses (residues past 9999, chains past Z,
    alternate locations) and because a full comparison would need the
    chemistry this module must not contain.
    """
    if primary is None:
        return ("no mmCIF record with the same name; a converted PDB may not "
                "stand in for the primary record")
    try:
        from ..science.structure_io import read_mmcif, read_pdb
    except Exception as exc:                        # pragma: no cover - defensive
        return f"not validated: the structure reader could not be imported ({exc})"
    try:
        reference = read_mmcif(primary)
        converted = read_pdb(derived)
    except Exception as exc:
        return f"not validated: {type(exc).__name__}: {exc}"
    n_ref = len(list(reference.atoms()))
    n_conv = len(list(converted.atoms()))
    if n_ref != n_conv:
        return (f"disagrees with the mmCIF record: {n_conv} atom(s) against "
                f"{n_ref}; kept, but not usable as a substitute")
    return f"validated against the mmCIF record: {n_ref} atom(s) in both"


def _copy_directory(src: Path, dest: Path, policy: DirectoryPolicy,
                    ) -> tuple[list[FileEntry], list[str], dict[str, int]]:
    """Copy a directory, classifying every file and validating the conversions."""
    entries: list[FileEntry] = []
    notes: list[str] = []
    counts = {"primary": 0, "derived": 0, "companion": 0,
              "rejected_substitute": 0, "skipped": 0}
    dest.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in src.rglob("*") if p.is_file() or p.is_symlink())
    primaries = {p.stem: p for p in sources
                 if p.is_file() and p.suffix.lower() in policy.primary_suffixes}
    for path in sources:
        relative = path.relative_to(src)
        if path.is_symlink() and not path.exists():
            counts["skipped"] += 1
            notes.append(f"{relative}: a broken symbolic link, not copied")
            continue
        if not path.is_file():
            counts["skipped"] += 1
            continue
        role = policy.role_of(path)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        note = ""
        if role == "derived":
            note = _validate_derived_structure(primaries.get(path.stem), path)
        elif role == "rejected_substitute":
            note = "kept as an illustration only"
            if policy.substitute_refusal:
                note += "; " + policy.substitute_refusal
        counts[role] = counts.get(role, 0) + 1
        entries.append(FileEntry(
            path=str(relative).replace("\\", "/"),
            sha256=sha256_file(target), n_bytes=target.stat().st_size,
            role=role, note=note))
    return entries, notes, counts


# ==========================================================================
# assembly
# ==========================================================================

def _load_run_manifest(run_dir: Path) -> tuple[RunManifest | None, str]:
    path = run_dir / RUN_MANIFEST_NAME
    if not path.exists():
        return None, (f"no {RUN_MANIFEST_NAME} in {run_dir}; the bundle was "
                      f"assembled from the files on disk alone, so nothing in "
                      f"it can be traced back to the step that produced it")
    try:
        return RunManifest.load(path), ""
    except Exception as exc:
        return None, (f"{RUN_MANIFEST_NAME} could not be read "
                      f"({type(exc).__name__}: {exc}); the bundle was assembled "
                      f"from the files on disk alone")


def _assemble_file_item(item: BundleItem, src: Path, bundle_dir: Path,
                        entry: BundleEntry) -> None:
    """Copy one file deliverable, naming it from its contents where required."""
    name = item.name
    if item.count_name_template is not None:
        constructs, rows = _count_batch_constructs(src)
        if constructs is None:
            entry.notes.append(
                f"the construct count could not be read from {src.name}, so the "
                f"file keeps the name the run gave it rather than a count this "
                f"bundle did not verify")
            name = src.name
        else:
            name = item.count_name_template.format(n=constructs)
            entry.n_records = constructs
            if rows is not None and rows != constructs:
                entry.notes.append(
                    f"{rows} row(s) in the order form, of which {constructs} are "
                    f"constructs; the remainder are controls")
            if name != item.name:
                entry.notes.append(
                    f"named from the actual construct count; declared as "
                    f"'{item.name}', written as '{name}' because a short batch "
                    f"labelled as a full one is read as a full one later")
    dest = bundle_dir / name
    copied = _copy_file(src, dest)
    entry.path = name
    entry.files = [FileEntry(path=name, sha256=copied.sha256,
                             n_bytes=copied.n_bytes, role="record")]
    entry.n_files = 1
    if entry.n_records is None:
        entry.n_records = _count_records(dest)
    entry.status = ItemStatus.PRESENT


def _assemble_directory_item(item: BundleItem, src: Path, bundle_dir: Path,
                             entry: BundleEntry) -> None:
    """Copy one directory deliverable and judge whether the records are there."""
    dest = bundle_dir / item.name
    files, notes, counts = _copy_directory(src, dest, item.directory_policy)
    entry.path = item.name
    entry.files = [FileEntry(path=f"{item.name}/{f.path}", sha256=f.sha256,
                             n_bytes=f.n_bytes, role=f.role, note=f.note)
                   for f in files]
    entry.n_files = len(entry.files)
    entry.notes.extend(notes)
    policy = item.directory_policy
    if not entry.files:
        entry.status = ItemStatus.MISSING
        entry.reason = f"{src} exists but holds no files"
        return
    if policy.primary_required and counts.get("primary", 0) == 0:
        entry.status = ItemStatus.PARTIAL
        substitutes = counts.get("rejected_substitute", 0)
        derived = counts.get("derived", 0)
        detail = []
        if derived:
            detail.append(f"{derived} converted file(s) are present")
        if substitutes:
            detail.append(f"{substitutes} image file(s) are present")
        entry.reason = (
            f"no {policy.primary_description} in this directory"
            + (f" ({'; '.join(detail)})" if detail else "")
            + (f"; {policy.substitute_refusal}" if policy.substitute_refusal else ""))
        return
    if counts.get("derived", 0):
        unvalidated = [f.path for f in entry.files
                       if f.role == "derived" and not f.note.startswith("validated")]
        if unvalidated:
            entry.status = ItemStatus.PARTIAL
            entry.reason = (
                f"{len(unvalidated)} converted file(s) could not be validated "
                f"against their mmCIF record: {', '.join(sorted(unvalidated))}")
            return
    entry.status = ItemStatus.PRESENT


def assemble_bundle(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    run_manifest: RunManifest | None = None,
    candidates: Sequence[Any] | None = None,
    overwrite: bool = False,
    write_report: bool = True,
) -> BundleResult:
    """Collect one run's deliverables into a traceable package.

    Every item of :data:`BUNDLE_ITEMS` gets an entry whatever happened to it,
    which is the only way the recipient of a package can tell the difference
    between "this run did not get that far" and "somebody forgot to copy it".

    ``candidates`` is optional: when the caller still has the
    :class:`~eagent.schemas.candidate.Candidate` objects in memory, the report
    regenerates their justifications through
    :func:`eagent.science.scorecard.explain`; otherwise it quotes the
    ``candidate_explanations.txt`` artifact, which the same function produced.
    Nothing is written about a candidate from any other source.
    """
    source = Path(run_dir)
    if not source.is_dir():
        raise FileNotFoundError(
            f"run directory {source} does not exist; a bundle is assembled "
            f"from a run, not from nothing")
    bundle_dir = Path(out_dir) if out_dir is not None else source / "bundle"
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        if not overwrite and not (bundle_dir / BUNDLE_MANIFEST_NAME).exists():
            raise FileExistsError(
                f"{bundle_dir} is not empty and does not look like a bundle "
                f"this tool wrote; refusing to mix a new package into it. Pass "
                f"overwrite=True to replace its contents")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest_obj = run_manifest
    manifest_note = ""
    if manifest_obj is None:
        manifest_obj, manifest_note = _load_run_manifest(source)

    bundle = BundleManifest(
        run_id=getattr(manifest_obj, "run_id", None),
        task_id=getattr(manifest_obj, "task_id", None),
        source_run_dir=str(source.resolve()),
    )
    if manifest_note:
        bundle.notes.append(manifest_note)
    bundle.notes.append(
        "every item of the standard package is listed below; an item that is "
        "absent is recorded as missing with the reason, never left out")

    for item in BUNDLE_ITEMS:
        entry = BundleEntry(name=item.name, kind=item.kind,
                            status=ItemStatus.MISSING, why=item.why,
                            curator_note=item.curator_note)
        bundle.entries.append(entry)
        if item.is_generated:
            continue                     # written after everything it reports on
        found, how, notes = _resolve_source(item, source, manifest_obj)
        entry.notes.extend(notes)
        if found is None:
            entry.reason = (
                f"not produced by this run and not present in {source}"
                + (f"; {'; '.join(notes)}" if notes else ""))
            continue
        entry.source = how
        if item.is_directory:
            if not found.is_dir():
                entry.reason = (f"{found} is a file; this deliverable is a "
                                f"directory of records")
                continue
            _assemble_directory_item(item, found, bundle_dir, entry)
        else:
            if not found.is_file():
                entry.reason = f"{found} is not a readable file"
                continue
            _assemble_file_item(item, found, bundle_dir, entry)

    report_path: Path | None = None
    report_entry = bundle.entry(RESEARCH_REPORT_NAME)
    if write_report:
        # Marked present *before* rendering, so the counts the report prints
        # about its own package are the counts the manifest will carry. A
        # report that lists itself as a missing item is a report nobody can
        # trust about anything else either.
        if report_entry is not None:
            report_entry.status = ItemStatus.PRESENT
            report_entry.path = RESEARCH_REPORT_NAME
            report_entry.source = "generated from the items in this bundle"
            report_entry.n_files = 1
        text = render_research_report(bundle_dir, bundle, source,
                                      run_manifest=manifest_obj,
                                      candidates=candidates)
        report_path = bundle_dir / RESEARCH_REPORT_NAME
        report_path.write_text(text, encoding="utf-8")
        if report_entry is not None:
            report_entry.files = [FileEntry(
                path=RESEARCH_REPORT_NAME, sha256=sha256_file(report_path),
                n_bytes=report_path.stat().st_size, role="record")]
    elif report_entry is not None:
        report_entry.reason = "report generation was switched off by the caller"

    manifest_path = bundle.write(bundle_dir / BUNDLE_MANIFEST_NAME)
    return BundleResult(bundle_dir=bundle_dir, manifest=bundle,
                        manifest_path=manifest_path, report_path=report_path)


# ==========================================================================
# the report
# ==========================================================================

def _fact(statement: str, source: str) -> str:
    """A sentence carrying a number, with the file the number came from.

    Refuses an empty source. A quantitative claim with no artifact behind it
    is the thing this report is not allowed to contain, and making that a
    programming error is cheaper than reviewing for it.
    """
    if not str(source).strip():
        raise ValueError(
            f"quantitative claim without a source artifact: {statement!r}. "
            f"If the artifact is absent, state that the number is unavailable "
            f"instead of printing one")
    return f"{statement} [source: {source}]"


def _unavailable(statement: str, reason: str) -> str:
    return f"{statement}: unavailable -- {reason}"


def _split_explanations(text: str) -> list[str]:
    """Split ``candidate_explanations.txt`` back into its per-candidate blocks."""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if set(line.strip()) == {"="} and len(line.strip()) >= 10:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def _candidate_blocks(bundle_dir: Path, bundle: BundleManifest,
                      candidates: Sequence[Any] | None,
                      ) -> tuple[list[tuple[str, str]], str, str]:
    """``([(candidate_id, text)], source_name, note)`` for the report.

    Prefers regenerating from live :class:`Candidate` objects, because that
    runs the same :func:`eagent.science.scorecard.explain` the artifact was
    written with and cannot drift from the current schema. Falls back to the
    stored artifact, and says which was used.
    """
    if candidates:
        try:
            from ..science.scorecard import explain
        except Exception as exc:                    # pragma: no cover - defensive
            return [], "", (f"the scorecard explainer could not be imported "
                            f"({type(exc).__name__}: {exc})")
        blocks = [(str(getattr(c, "candidate_id", f"candidate_{i}")), explain(c))
                  for i, c in enumerate(candidates)]
        return blocks, "eagent.science.scorecard.explain, run over the " \
                       "candidates supplied to the bundler", ""
    entry = bundle.entry("candidate_explanations.txt")
    if entry is None or entry.status is ItemStatus.MISSING or not entry.path:
        return [], "", ("no candidate explanations are in this bundle, so this "
                        "report states none; see the missing-items section")
    path = bundle_dir / entry.path
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], "", f"{entry.path} could not be read ({exc})"
    blocks = []
    for block in _split_explanations(text):
        first = block.splitlines()[0].strip()
        identifier = first[len("Candidate "):].strip() if first.startswith("Candidate ") \
            else first
        blocks.append((identifier, block))
    return blocks, entry.path, ""


def _reaction_spec_lines(bundle_dir: Path, bundle: BundleManifest) -> list[str]:
    entry = bundle.entry("reaction_spec.yaml")
    if entry is None or entry.status is ItemStatus.MISSING or not entry.path:
        return ["- " + _unavailable(
            "the reaction this package is about",
            "no reaction_spec.yaml in this bundle; every statement below is "
            "therefore about an unconfirmed molecule")]
    try:
        import yaml
        document = yaml.safe_load((bundle_dir / entry.path).read_text(
            encoding="utf-8"))
    except Exception as exc:
        return [f"- reaction_spec.yaml could not be parsed "
                f"({type(exc).__name__}: {exc}); it is in the bundle unchanged"]
    if not isinstance(document, Mapping):
        return ["- reaction_spec.yaml did not parse into a mapping"]
    reaction = document.get("reaction") or {}
    substrate = (reaction.get("substrate") or {}) if isinstance(reaction, Mapping) else {}
    product = (reaction.get("product") or {}) if isinstance(reaction, Mapping) else {}
    unsupplied = "null -- not supplied by an operator"
    reaction_class = reaction.get("reaction_class", "unrecorded") \
        if isinstance(reaction, Mapping) else "unrecorded"
    substrate_smiles = substrate.get("isomeric_smiles") or unsupplied
    product_smiles = product.get("isomeric_smiles") or unsupplied
    configuration = product.get("target_stereochemistry", "unrecorded")
    lines = [
        f"- reaction class: {reaction_class} [source: {entry.path}]",
        f"- substrate: {substrate_smiles} [source: {entry.path}]",
        f"- product: {product_smiles} [source: {entry.path}]",
        f"- target configuration: {configuration} [source: {entry.path}]",
    ]
    checks = document.get("checks")
    if isinstance(checks, Mapping):
        unresolved = checks.get("unresolved_for_gate")
        if isinstance(unresolved, (list, tuple)) and unresolved:
            lines.append(_fact(
                f"- {len(unresolved)} field(s) still unresolved for the "
                f"reaction_spec_confirmed gate: {', '.join(str(u) for u in unresolved)}",
                entry.path))
        elif isinstance(unresolved, Mapping) and unresolved:
            for gate, paths in unresolved.items():
                lines.append(_fact(
                    f"- gate {gate}: {len(list(paths))} unresolved field(s) "
                    f"({', '.join(str(p) for p in paths)})", entry.path))
    return lines


def _run_lines(bundle: BundleManifest, run_manifest: RunManifest | None) -> list[str]:
    if run_manifest is None:
        return ["- " + _unavailable(
            "what this run executed",
            f"no {RUN_MANIFEST_NAME} was available, so step count, costs and "
            f"approvals cannot be stated")]
    lines = [_fact(f"- {len(run_manifest.steps)} step(s) recorded",
                   RUN_MANIFEST_NAME)]
    for step in run_manifest.steps:
        lines.append(f"  - {step.step_id}: {step.status}"
                     + (f" -- {step.message}" if step.message else "")
                     + f" [source: {RUN_MANIFEST_NAME}]")
    if run_manifest.cost_total:
        for key, value in sorted(run_manifest.cost_total.items()):
            lines.append(_fact(f"- cost {key}: {value}", RUN_MANIFEST_NAME))
    else:
        lines.append("- cost: nothing was recorded by any step; this is not a "
                     "statement that the run was free")
    if run_manifest.approvals:
        for approval in run_manifest.approvals:
            lines.append(
                f"- gate {approval.get('gate')}: {approval.get('decision')} by "
                f"{approval.get('actor')} at {approval.get('at')} "
                f"[source: {RUN_MANIFEST_NAME}]")
    else:
        lines.append("- approvals: none recorded; no human decision is on file "
                     "for any gate in this run")
    databases = run_manifest.databases or {}
    if databases:
        for name, version in sorted(databases.items()):
            lines.append(f"- database {name}: {version} "
                         f"[source: {RUN_MANIFEST_NAME}]")
    else:
        lines.append("- databases: none recorded by any step")
    return lines


def _batch_lines(bundle_dir: Path, bundle: BundleManifest) -> list[str]:
    entry = None
    for candidate_entry in bundle.entries:
        if candidate_entry.name == BATCH_ITEM_NAME:
            entry = candidate_entry
            break
    if entry is None or entry.status is ItemStatus.MISSING or not entry.path:
        return ["- " + _unavailable(
            "the composition of the ordered batch",
            "no batch order form is in this bundle; no construct count may be "
            "quoted")]
    lines = []
    if entry.n_records is not None:
        lines.append(_fact(f"- {entry.n_records} construct(s) in the batch",
                           entry.path))
    else:
        lines.append(f"- the construct count could not be read from "
                     f"{entry.path}; it is not estimated here")
    if entry.renamed:
        lines.append(f"- the file is named `{entry.path}`, not "
                     f"`{entry.name}`: it carries the count it actually holds")
    parsed = _read_delimited(bundle_dir / entry.path)
    if parsed is not None:
        header, rows = parsed
        lowered = [h.strip().lower() for h in header]
        if "role" in lowered:
            index = lowered.index("role")
            counts: dict[str, int] = {}
            for row in rows:
                if len(row) > index:
                    counts[row[index]] = counts.get(row[index], 0) + 1
            for role, n in sorted(counts.items()):
                lines.append(_fact(f"  - role {role or 'unlabelled'}: {n} row(s)",
                                   entry.path))
    return lines


def _missing_lines(bundle: BundleManifest) -> list[str]:
    lines: list[str] = []
    for entry in bundle.missing:
        lines.append(f"- **{entry.name}** -- MISSING. {entry.reason or 'no reason recorded'}.")
        lines.append(f"  - what it is for: {entry.why}")
        if entry.curator_note:
            lines.append(f"  - to supply it: {entry.curator_note}")
    for entry in bundle.partial:
        lines.append(f"- **{entry.name}** -- PARTIAL. {entry.reason}")
        if entry.curator_note:
            lines.append(f"  - to complete it: {entry.curator_note}")
    if not lines:
        lines.append("- nothing: every declared item of the standard package is "
                     "present in full.")
    return lines


def _backing_tables_line(bundle: BundleManifest) -> str:
    """Where the measurements behind the justifications are -- or are not.

    Naming a backing table that is not in the package would invite a reader
    to go and check a file that is not there, and an unchecked citation reads
    as a checked one.
    """
    wanted = ("catalytic_geometry.tsv", "candidate_scorecards.tsv")
    here, absent = [], []
    for name in wanted:
        entry = bundle.entry(name)
        (here if entry is not None and entry.status is not ItemStatus.MISSING
         else absent).append(name)
    parts = []
    if here:
        parts.append("The measurements behind these lines are in "
                     + ", ".join(f"`{n}`" for n in here) + ".")
    if absent:
        parts.append("Not in this package, so the lines below cannot be "
                     "checked against them: "
                     + ", ".join(f"`{n}`" for n in absent)
                     + " (see \"What is missing\").")
    return " ".join(parts)


def render_research_report(bundle_dir: Path, bundle: BundleManifest,
                           run_dir: Path | None = None, *,
                           run_manifest: RunManifest | None = None,
                           candidates: Sequence[Any] | None = None) -> str:
    """Generate ``research_report.md`` from the package that was just assembled.

    Written from the artifacts rather than from memory of the run: a report
    that states something no file in the bundle supports cannot be rechecked
    by its reader, and will outlive the run.
    """
    counts = bundle.counts()
    out: list[str] = []
    out.append(f"# Research report -- run {bundle.run_id or 'unidentified'}")
    out.append("")
    out.append(f"Task: `{bundle.task_id or 'unrecorded'}`  ")
    out.append(f"Assembled: {bundle.created_at}  ")
    out.append(f"Source run directory: `{bundle.source_run_dir}`")
    out.append("")
    out.append("## How to read this report")
    out.append("")
    out.append("- Every quantitative claim below names the file it came from in "
               "square brackets. A number with no such citation is a defect in "
               "this generator, not a finding.")
    out.append("- There is no total score anywhere in this package, and none "
               "may be derived from the scorecard columns: the axes have "
               "different units and no calibrated exchange rate.")
    out.append("- An absent artifact is reported as absent. Nothing here is "
               "estimated, interpolated or filled in from a typical value.")
    if not bundle.complete:
        out.append(f"- **This package is incomplete**: "
                   f"{counts['missing']} declared item(s) missing and "
                   f"{counts['partial']} partial, of {counts['declared']} "
                   f"[source: {BUNDLE_MANIFEST_NAME}]. See "
                   f"\"What is missing\" below before quoting anything from it.")
    else:
        out.append(f"- All {counts['declared']} declared items are present "
                   f"[source: {BUNDLE_MANIFEST_NAME}].")
    out.append("")

    out.append("## The reaction")
    out.append("")
    out.extend(_reaction_spec_lines(bundle_dir, bundle))
    out.append("")

    out.append("## What the run did")
    out.append("")
    out.extend(_run_lines(bundle, run_manifest))
    out.append("")

    out.append("## Candidates")
    out.append("")
    blocks, block_source, block_note = _candidate_blocks(
        bundle_dir, bundle, candidates)
    if block_note:
        out.append(f"- {block_note}")
    if blocks:
        out.append(_fact(f"{len(blocks)} candidate justification(s) follow.",
                         block_source))
        out.append("")
        out.append("Each one answers, in order: why the sequence was retrieved, "
                   "which family it is and on what evidence, what supports the "
                   "target reaction specifically, how the substrate and cofactor "
                   "are positioned and by what authority, what is still "
                   "uncertain, and why it deserves an experimental slot.")
        out.append("")
        out.append(_backing_tables_line(bundle))
        out.append("")
        for identifier, text in blocks:
            out.append(f"### {identifier}")
            out.append("")
            out.append(f"_Generated by `eagent.science.scorecard.explain`; "
                       f"source: {block_source}._")
            out.append("")
            out.append("```")
            out.append(text)
            out.append("```")
            out.append("")
    else:
        out.append("- No per-candidate justification is available, so this "
                   "report makes no claim about any candidate.")
        out.append("")

    out.append("## The batch")
    out.append("")
    out.extend(_batch_lines(bundle_dir, bundle))
    out.append("")

    out.append("## What is missing")
    out.append("")
    out.extend(_missing_lines(bundle))
    out.append("")

    out.append("## Package contents")
    out.append("")
    out.append("| item | status | written as | files | sha256 (file items) |")
    out.append("| --- | --- | --- | --- | --- |")
    for entry in bundle.entries:
        if len(entry.files) == 1 and entry.files[0].sha256:
            digest = entry.files[0].sha256[:16] + "..."
        elif entry.status is ItemStatus.MISSING:
            digest = "-"
        else:
            digest = f"see {BUNDLE_MANIFEST_NAME}"
        out.append(f"| {entry.name} | {entry.status.value} | "
                   f"{entry.path or '-'} | "
                   f"{'-' if entry.n_files is None else entry.n_files} | "
                   f"{digest} |")
    out.append("")
    out.append(f"Hashes for every file, including those inside the directories, "
               f"are in `{BUNDLE_MANIFEST_NAME}`; `eagent.deliverables.bundle."
               f"verify_bundle` re-checks them.")
    out.append("")
    return "\n".join(out)


# ==========================================================================
# verification
# ==========================================================================

def verify_bundle(path: str | Path) -> BundleVerification:
    """Re-check a bundle against its own manifest.

    Three things are checked, and the third is the one people forget: a
    declared file must exist, its content must still hash to what was
    recorded, and no extra file may be sitting inside a declared directory
    without being recorded. An unrecorded file in a traceable package is
    either an edit nobody logged or a provenance gap, and both make the
    package unusable as a record.
    """
    bundle_dir = Path(path)
    if bundle_dir.is_file() and bundle_dir.name == BUNDLE_MANIFEST_NAME:
        bundle_dir = bundle_dir.parent
    verification = BundleVerification(bundle_dir=bundle_dir)
    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    if not manifest_path.exists():
        verification.problems.append(
            f"no {BUNDLE_MANIFEST_NAME} in {bundle_dir}; a package without its "
            f"manifest cannot be verified and must not be treated as one")
        return verification
    try:
        manifest = BundleManifest.load(manifest_path)
    except Exception as exc:
        verification.problems.append(
            f"{BUNDLE_MANIFEST_NAME} could not be read "
            f"({type(exc).__name__}: {exc})")
        return verification

    verification.complete = manifest.complete
    recorded: set[Path] = {manifest_path.resolve()}
    for entry in manifest.entries:
        verification.n_items_checked += 1
        if entry.status is ItemStatus.MISSING:
            if entry.path and (bundle_dir / entry.path).exists():
                verification.problems.append(
                    f"{entry.name}: recorded as missing, but "
                    f"{entry.path} exists in the bundle")
            continue
        if not entry.files:
            verification.problems.append(
                f"{entry.name}: recorded as {entry.status.value} but the "
                f"manifest lists no file for it, so nothing can be checked")
            continue
        for file_entry in entry.files:
            verification.n_files_checked += 1
            target = bundle_dir / file_entry.path
            recorded.add(target.resolve())
            if not target.is_file():
                verification.problems.append(
                    f"{file_entry.path}: declared in the manifest, absent from "
                    f"the bundle")
                continue
            actual = sha256_file(target)
            if actual != file_entry.sha256:
                verification.problems.append(
                    f"{file_entry.path}: content does not match the manifest "
                    f"(recorded {file_entry.sha256[:16]}..., found "
                    f"{actual[:16]}...); the file has changed since the bundle "
                    f"was assembled")

    for present in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        if present.resolve() not in recorded:
            verification.problems.append(
                f"{present.relative_to(bundle_dir)}: present in the bundle but "
                f"not recorded in {BUNDLE_MANIFEST_NAME}; an unrecorded file "
                f"has no provenance")
    return verification


def bundle_summary_lines(result: BundleResult) -> list[str]:
    """Plain-text summary of a bundle, for a terminal with no colour."""
    manifest = result.manifest
    counts = manifest.counts()
    lines = [
        f"bundle: {result.bundle_dir}",
        f"run: {manifest.run_id or 'unidentified'}  "
        f"task: {manifest.task_id or 'unrecorded'}",
        f"items: {counts['present']} present, {counts['partial']} partial, "
        f"{counts['missing']} missing, of {counts['declared']} declared",
        f"complete: {'yes' if manifest.complete else 'NO'}",
    ]
    for entry in manifest.entries:
        mark = {ItemStatus.PRESENT: "[ok]     ",
                ItemStatus.PARTIAL: "[partial]",
                ItemStatus.MISSING: "[MISSING]"}[entry.status]
        written = entry.path
        suffix = f" -> written as {written}" \
            if written and written != entry.name else ""
        lines.append(f"  {mark} {entry.name}{suffix}")
        if entry.status is not ItemStatus.PRESENT and entry.reason:
            lines.append(f"            reason: {entry.reason}")
            if entry.curator_note:
                lines.append(f"            supply: {entry.curator_note}")
        for note in entry.notes:
            lines.append(f"            note: {note}")
    return lines
