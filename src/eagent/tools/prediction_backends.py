"""Real adapters behind the two ``model_complexes`` seams.

``model_complexes`` defines two Protocols -- :class:`DockingRunner` (route A)
and :class:`ComplexPredictor` (route B) -- and ships only "absent, and loudly
so" defaults. This module is the first thing that can actually *be* one of
those, and it is written to be refused rather than trusted:

* **Each CLI shape was read, not remembered.** The Boltz command line, input
  YAML, output directory layout, confidence-JSON keys and ligand naming were
  read from the released ``boltz`` 2.2.1 wheel (``boltz/main.py``,
  ``boltz/data/parse/schema.py``, ``boltz/data/write/writer.py``). The gnina
  command line and licence were read from its README; its SD-file property
  names (``minimizedAffinity``, ``CNNscore``, ``CNNaffinity``, ``CNN_VS``) come
  from the gnina documentation as quoted by the ``biobb_vs`` wrapper -- *not*
  from running the binary, which is not installed here. Neither adapter has
  been run against a real binary. They have been run against fakes that write
  output in the documented shape, which tests the adapter's handling and
  nothing about the engine.
* **Nothing is fabricated when output is missing.** A non-zero exit, a missing
  output file, or an unparsable one raises :class:`BackendRunError` carrying
  the tool's own stderr. A property the engine did not write is ``None``, not
  zero.
* **An engine never claims what it did not do.** Neither adapter can apply a
  catalytic distance restraint, so a job that asks for one is refused with
  :class:`UnsupportedJobError`. Quietly ignoring it would produce a pose that
  the pipeline then records as restrained (the pipeline is conservative) while
  the engine never restrained it, which loses information in one direction and
  is the first step toward claiming it in the other.
* **A sequence leaves the machine only if the adapter says so.** Boltz's
  ``--use_msa_server`` sends the query sequence to a remote MSA service. The
  adapter never passes it unless constructed with ``use_msa_server=True``, and
  then reports ``runs_remotely`` so that ``guard_sequence_submission`` demands
  the named authorisation. Without it the run is single-sequence (``msa:
  empty``) or uses an MSA the caller supplies, and the result says which.

WHAT THE POSES ARE NOT
======================
A gnina pose is a placement under a scoring function inside a box the pipeline
chose from trusted site atoms. ``minimizedAffinity`` is comparable only within
one scoring function, receptor preparation and box, and ``CNNaffinity`` is a
network's pK estimate for *binding*, not turnover. A Boltz sample is a
hypothesis about placement, and ``confidence_score`` orders structural
plausibility. Neither is activity; ``model_complexes`` says so in the pose
index, and these adapters add nothing that could be read otherwise.

ATOM NAMES ARE THE ENGINE'S, NOT THE REACTION'S
===============================================
The role atoms of a catalytic template (``substrate.electrophile``) are
defined by the atom-mapped reaction. A docked or co-folded ligand carries atom
names its tool assigned: gnina's SD output has none, so this adapter names
atoms ``<element><ordinal in file order>``; Boltz names a SMILES ligand's atoms
``<ELEMENT><canonical rank + 1>`` (RDKit canonical ranking over the molecule
with hydrogens). Mapping reaction roles onto those names needs a substructure
match this module does not attempt, so every pose records its naming
convention in ``extra`` and a :class:`PoseBinding` for it must be built by
something that knows the correspondence. A pose whose binding was guessed from
atom order is a pose whose geometry is about the wrong atom.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Sequence

import yaml

from ..errors import EAgentError, ToolUnavailableError
from ..provenance import sha256_file
from ..science.structure_io import (
    Atom,
    Chain,
    Residue,
    Structure,
    StructureParseError,
    read_structure,
    write_pdb,
)
from .mine_sequences import CommandResult, CommandRunner, SubprocessRunner
from .model_complexes import DockingJob, JointPredictionJob, RawPose
from .prepare_structures import write_mmcif_text

__all__ = [
    "BackendRunError",
    "UnsupportedJobError",
    "SdfPose",
    "parse_sdf",
    "merge_ligand_into_receptor",
    "rename_residues",
    "GninaDockingRunner",
    "BoltzComplexPredictor",
]


class BackendRunError(EAgentError):
    """An engine ran and did not produce usable output.

    Carries the engine's own stderr: a bare "docking failed" would discard the
    only line that says why (a missing CUDA device, a malformed ligand).
    """


class UnsupportedJobError(EAgentError):
    """The job asks for something this engine cannot honour.

    Raised before the engine runs, because the alternative -- running anyway
    and dropping the request -- yields a pose that looks like it answered it.
    """


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


# ==========================================================================
# SD files
# ==========================================================================

@dataclass(frozen=True)
class SdfPose:
    """One molecule from an SD file: coordinates, elements, data items.

    Bonds are parsed past but not kept: nothing here needs them, and keeping a
    half-interpreted bond table would invite use as chemistry.
    """

    title: str
    elements: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]
    properties: Mapping[str, str]

    def number(self, name: str) -> float | None:
        """A data item as a float, or ``None`` if absent or not a number."""
        raw = self.properties.get(name)
        if raw is None:
            return None
        try:
            return float(raw.strip().split()[0])
        except (ValueError, IndexError):
            return None


def parse_sdf(text: str) -> list[SdfPose]:
    """Read a V2000 SD file.

    V3000 is refused rather than half-read, and a truncated record (counts that
    promise atoms the file does not contain) raises instead of yielding a
    shorter molecule: a pose with missing atoms would be measured as if whole.
    """
    poses: list[SdfPose] = []
    for block in re.split(r"^\$\$\$\$\s*$", text, flags=re.MULTILINE):
        lines = block.strip("\n").split("\n") if block.strip() else []
        if not lines:
            continue
        if len(lines) < 4:
            raise BackendRunError("an SD record has fewer than four lines")
        counts = lines[3]
        if "V3000" in counts:
            raise BackendRunError("V3000 SD records are not supported; ask the "
                                  "engine for V2000 output")
        try:
            n_atoms = int(counts[0:3])
            n_bonds = int(counts[3:6])
        except ValueError as exc:
            raise BackendRunError(f"unreadable SD counts line: {counts!r}") from exc
        atom_lines = lines[4:4 + n_atoms]
        if len(atom_lines) < n_atoms:
            raise BackendRunError(
                f"SD record promises {n_atoms} atoms and holds {len(atom_lines)}")
        elements: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for line in atom_lines:
            try:
                coords.append((float(line[0:10]), float(line[10:20]),
                               float(line[20:30])))
            except ValueError as exc:
                raise BackendRunError(f"unreadable SD atom line: {line!r}") from exc
            symbol = line[31:34].strip()
            if not symbol:
                raise BackendRunError(f"SD atom line has no element: {line!r}")
            elements.append(symbol[0].upper() + symbol[1:].lower())
        rest = lines[4 + n_atoms + n_bonds:]
        props: dict[str, str] = {}
        name: str | None = None
        values: list[str] = []
        for line in rest:
            match = re.match(r"^>\s*<([^>]+)>", line)
            if match:
                if name is not None:
                    props[name] = "\n".join(values).strip()
                name, values = match.group(1), []
            elif line.startswith("M  END"):
                continue
            elif name is not None and line.strip():
                values.append(line)
        if name is not None:
            props[name] = "\n".join(values).strip()
        poses.append(SdfPose(title=lines[0].strip(), elements=tuple(elements),
                             coords=tuple(coords), properties=props))
    return poses


# ==========================================================================
# assembling a complex from parts
# ==========================================================================

def merge_ligand_into_receptor(
    receptor: Structure, ligand: SdfPose, resname: str, *,
    chain_id: str | None = None, structure_id: str = "complex",
) -> tuple[Structure, dict[str, Any]]:
    """Receptor coordinates plus one ligand pose as a new residue.

    Returns the structure and the *selector* a binding needs to find the ligand
    again (chain, resname, resseq) with the atom naming convention used. Refuses
    a receptor that already contains a component of the same name: two copies
    would be indistinguishable to every later step, and the one that got
    measured might be the crystal ligand rather than the pose.
    """
    resname = resname.strip().upper()
    for chain in receptor.chains:
        for residue in chain.residues:
            if residue.resname.strip().upper() == resname:
                raise BackendRunError(
                    f"the receptor already contains a component named {resname} "
                    f"({residue}); adding the docked ligand under the same name "
                    f"would make the two indistinguishable")
    target = next((c for c in receptor.chains
                   if chain_id is not None and c.chain_id == chain_id), None)
    if target is None:
        if chain_id is not None:
            raise BackendRunError(f"the receptor has no chain {chain_id!r}")
        target = receptor.chains[0] if receptor.chains else None
    if target is None:
        raise BackendRunError("the receptor has no chains")
    resseq = max((r.resseq for r in target.residues), default=0) + 1
    serial = max((a.serial for c in receptor.chains for a in c.atoms()),
                 default=0)
    counts: dict[str, int] = {}
    atoms: list[Atom] = []
    for element, (x, y, z) in zip(ligand.elements, ligand.coords):
        counts[element] = counts.get(element, 0) + 1
        serial += 1
        atoms.append(Atom(
            serial=serial, name=f"{element.upper()}{counts[element]}",
            element=element, resname=resname, chain=target.chain_id,
            resseq=resseq, icode="", altloc="", x=x, y=y, z=z,
            is_hetatm=True))
    new_chains = [Chain(chain_id=c.chain_id, residues=list(c.residues))
                  for c in receptor.chains]
    for c in new_chains:
        if c.chain_id == target.chain_id:
            c.residues.append(Residue(chain=c.chain_id, resname=resname,
                                      resseq=resseq, icode="", atoms=atoms,
                                      is_hetatm=True))
    merged = Structure(structure_id=structure_id, chains=new_chains,
                       source_format="mmcif",
                       models_present=list(receptor.models_present),
                       model_selected=receptor.model_selected,
                       parse_notes=list(receptor.parse_notes) + [
                           f"ligand {resname} added from an SD pose by "
                           f"prediction_backends"])
    selector = {"chain": target.chain_id, "resname": resname, "resseq": resseq,
                "atom_naming": "element symbol + ordinal of that element in "
                               "the engine's output order (C1, C2, O1, ...)",
                "atom_names_follow_reaction_atom_map": False}
    return merged, selector


def rename_residues(structure: Structure, mapping: Mapping[str, str],
                    structure_id: str | None = None) -> Structure:
    """A copy of ``structure`` with component names replaced.

    Used because a co-folding tool names a SMILES ligand by its own scheme
    (Boltz: ``LIG1``) while the pipeline looks for the component code the
    request named. Coordinates, atom names and per-atom confidence are kept.
    """
    upper = {k.upper(): v.upper() for k, v in mapping.items()}
    chains: list[Chain] = []
    for chain in structure.chains:
        residues: list[Residue] = []
        for res in chain.residues:
            new = upper.get(res.resname.strip().upper())
            if new is None:
                residues.append(res)
                continue
            residues.append(Residue(
                chain=res.chain, resname=new, resseq=res.resseq,
                icode=res.icode, is_hetatm=res.is_hetatm,
                atoms=[replace(a, resname=new) for a in res.atoms]))
        chains.append(Chain(chain_id=chain.chain_id, residues=residues))
    return Structure(
        structure_id=structure_id or structure.structure_id, chains=chains,
        source_format=structure.source_format, source_path=structure.source_path,
        models_present=list(structure.models_present),
        model_selected=structure.model_selected,
        parse_notes=list(structure.parse_notes) + [
            "component names rewritten: "
            + ", ".join(f"{a}->{b}" for a, b in sorted(upper.items()))])


# ==========================================================================
# shared plumbing
# ==========================================================================

class _Backend:
    """Binary resolution and execution, in the one way the repo does it."""

    binary: ClassVar[str] = ""
    install_hint: ClassVar[str] = ""

    def __init__(self, runner: CommandRunner | None = None,
                 executable_finder: Callable[[str], str | None] | None = None,
                 timeout_s: float | None = 6 * 3600.0) -> None:
        self.runner: CommandRunner = runner or SubprocessRunner()
        self.executable_finder = executable_finder or shutil.which
        self.timeout_s = timeout_s
        self._version: str | None = None

    def is_available(self) -> bool:
        return self.executable_finder(self.binary) is not None

    def _executable(self) -> str:
        found = self.executable_finder(self.binary)
        if not found:
            raise ToolUnavailableError(self.binary, self.install_hint)
        return found

    def _run(self, argv: Sequence[str], cwd: Path | None = None) -> CommandResult:
        try:
            result = self.runner(argv, cwd=cwd, timeout=self.timeout_s)
        except FileNotFoundError as exc:
            raise ToolUnavailableError(self.binary, self.install_hint) from exc
        if result.returncode != 0:
            raise BackendRunError(
                f"{self.binary} exited with status {result.returncode}: "
                f"{_tail(result.stderr) or _tail(result.stdout) or 'no output'}")
        return result

    def _read_version(self, argv: Sequence[str]) -> str:
        if self._version is None:
            try:
                out = self.runner(argv, cwd=None, timeout=60.0)
                text = (out.stdout or out.stderr).strip()
                self._version = text.splitlines()[0] if (
                    out.returncode == 0 and text) else "unreported"
            except Exception:                     # noqa: BLE001 - unknown, say so
                self._version = "unreported"
        return self._version


# ==========================================================================
# route A: gnina
# ==========================================================================

class GninaDockingRunner(_Backend):
    """Template-guided pocket docking with gnina (``DockingRunner``).

    One gnina run per conformer file, inside the box the job carries; every
    pose of every run is returned, in the engine's own order, because
    ``model_complexes`` keeps them all. The ligand pose is merged into the
    receptor (cofactor included, since it is part of the rigid receptor the
    pipeline supplied) and written as mmCIF so the pipeline can read the
    complete catalytic system.

    Reads from the output only what the gnina documentation names:
    ``minimizedAffinity`` (the empirical scoring function's value, kcal/mol,
    lower is better) as the score, and ``CNNscore`` / ``CNNaffinity`` /
    ``CNN_VS`` as confidence fields under explicit names. CNN scoring defaults
    to ``rescore``; pass ``cnn_scoring="none"`` for a weights-free run, which
    changes the registry keys the licence check demands.
    """

    binary = "gnina"
    install_hint = ("install gnina (https://github.com/gnina/gnina) and record "
                    "the version printed by `gnina --version`")
    name = "gnina"
    #: The properties the gnina documentation names in its output SD file.
    SCORE_PROPERTY: ClassVar[str] = "minimizedAffinity"
    CNN_PROPERTIES: ClassVar[Mapping[str, str]] = {
        "CNNscore": "cnn_pose_score", "CNNaffinity": "cnn_affinity_pK",
        "CNN_VS": "cnn_vs"}

    def __init__(self, runner: CommandRunner | None = None,
                 executable_finder: Callable[[str], str | None] | None = None,
                 timeout_s: float | None = 6 * 3600.0, *,
                 exhaustiveness: int = 8, num_modes: int = 9,
                 cnn_scoring: str = "rescore", cpus: int | None = None,
                 use_gpu: bool = False) -> None:
        super().__init__(runner, executable_finder, timeout_s)
        if cnn_scoring not in ("none", "rescore", "refinement", "all"):
            raise ValueError(f"unsupported cnn_scoring {cnn_scoring!r}")
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.cnn_scoring = cnn_scoring
        self.cpus = cpus
        self.use_gpu = use_gpu

    # -- the DockingRunner protocol ------------------------------------
    @property
    def version(self) -> str:
        exe = self.executable_finder(self.binary)
        if not exe:
            return "absent"
        return self._read_version([exe, "--version"])

    @property
    def uses_model_weights(self) -> bool:
        return self.cnn_scoring != "none"

    @property
    def registry_keys(self) -> tuple[str, ...]:
        keys = ["gnina.code"]
        if self.uses_model_weights:
            keys.append("gnina.model_weights")
        keys.append("gnina.output")
        return tuple(keys)

    def dock(self, job: DockingJob) -> Sequence[RawPose]:
        if job.restraints:
            raise UnsupportedJobError(
                f"gnina cannot apply the geometric restraint(s) "
                f"{list(job.restraint_names)}; the search box limits where the "
                f"ligand may go and nothing more. Run without restraints, or "
                f"use an engine that can enforce them and record them")
        if not job.conformer_paths:
            raise UnsupportedJobError("route A needs at least one conformer file")
        exe = self._executable()
        job.out_dir.mkdir(parents=True, exist_ok=True)
        receptor_file = self._receptor_file(job)
        try:
            receptor = read_structure(job.receptor_path,
                                      structure_id=f"{job.candidate_id}_receptor")
        except (StructureParseError, OSError) as exc:
            raise BackendRunError(f"receptor could not be read: {exc}") from exc

        raw: list[RawPose] = []
        for n, conformer in enumerate(job.conformer_paths):
            cpath = Path(conformer)
            if not cpath.is_file():
                raise BackendRunError(f"conformer file {cpath} does not exist")
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", cpath.stem) or f"conf{n}"
            out_sdf = job.out_dir / f"{stem}.poses.sdf"
            argv = self._argv(exe, job, receptor_file, cpath, out_sdf)
            self._run(argv, cwd=job.out_dir)
            if not out_sdf.is_file():
                raise BackendRunError(
                    f"gnina reported success but wrote no {out_sdf.name}")
            poses = parse_sdf(out_sdf.read_text(encoding="utf-8"))
            for k, pose in enumerate(poses, start=1):
                raw.append(self._raw_pose(job, receptor, pose, stem, k,
                                          out_sdf, argv))
        return raw

    # -- internals --------------------------------------------------------
    def _receptor_file(self, job: DockingJob) -> Path:
        path = Path(job.receptor_path)
        if path.suffix.lower() in (".pdb", ".ent"):
            return path
        try:
            structure = read_structure(path, structure_id="receptor")
        except (StructureParseError, OSError) as exc:
            raise BackendRunError(f"receptor could not be read: {exc}") from exc
        out = job.out_dir / "receptor_for_gnina.pdb"
        out.write_text(write_pdb(structure), encoding="utf-8")
        return out

    def _argv(self, exe: str, job: DockingJob, receptor: Path, conformer: Path,
              out_sdf: Path) -> list[str]:
        cx, cy, cz = job.box.center
        sx, sy, sz = job.box.size
        argv = [exe, "-r", str(receptor), "-l", str(conformer),
                "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}",
                "--center_z", f"{cz:.3f}",
                "--size_x", f"{sx:.3f}", "--size_y", f"{sy:.3f}",
                "--size_z", f"{sz:.3f}",
                "-o", str(out_sdf), "--seed", str(job.seed),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.num_modes),
                "--cnn_scoring", self.cnn_scoring]
        if self.cpus:
            argv += ["--cpu", str(self.cpus)]
        if not self.use_gpu:
            argv.append("--no_gpu")
        return argv

    def _raw_pose(self, job: DockingJob, receptor: Structure, pose: SdfPose,
                  stem: str, k: int, sdf_path: Path,
                  argv: Sequence[str]) -> RawPose:
        merged, selector = merge_ligand_into_receptor(
            receptor, pose, job.substrate_ligand_code,
            chain_id=job.receptor_chain,
            structure_id=f"{job.candidate_id}_{stem}_{k}")
        path = job.out_dir / f"{stem}.pose{k:03d}.cif"
        path.write_text(write_mmcif_text(merged, merged.structure_id),
                        encoding="utf-8")
        confidence = {label: v for prop, label in self.CNN_PROPERTIES.items()
                      if (v := pose.number(prop)) is not None}
        return RawPose(
            path=path, score=pose.number(self.SCORE_PROPERTY),
            score_function=(
                f"gnina {self.SCORE_PROPERTY} (empirical scoring; cnn_scoring="
                f"{self.cnn_scoring}; comparable only within one receptor "
                f"preparation and box)"),
            model_confidence=confidence, conformer_id=stem, sample_index=k,
            restrained_constraints=(),
            extra={
                "engine": "gnina", "engine_version": self.version,
                "raw_output": str(sdf_path), "raw_output_sha256": sha256_file(sdf_path),
                "argv": list(argv), "ligand_selector": selector,
                "mode_in_engine_order": k,
                "sd_properties": dict(pose.properties),
                "score_property_present": self.SCORE_PROPERTY in pose.properties,
                "restraints_enforced": False,
            })


# ==========================================================================
# route B: Boltz
# ==========================================================================

#: Keys the Boltz writer puts in ``confidence_<id>_model_<k>.json`` (read from
#: ``boltz/data/write/writer.py``). Mapped onto the names ``model_complexes``
#: indexes; ``confidence_score`` is Boltz's own ranking quantity.
BOLTZ_CONFIDENCE_KEYS: Mapping[str, str] = {
    "confidence_score": "ranking_score", "ptm": "ptm", "iptm": "iptm",
    "ligand_iptm": "ligand_iptm", "protein_iptm": "protein_iptm",
    "complex_plddt": "complex_plddt", "complex_iplddt": "complex_iplddt",
    "complex_pde": "complex_pde", "complex_ipde": "complex_ipde",
}

#: Residue name Boltz gives the first SMILES-defined ligand. Ligands are named
#: ``LIG<n>`` in the order they appear in the input (``schema.py``).
BOLTZ_FIRST_SMILES_LIGAND = "LIG1"


class BoltzComplexPredictor(_Backend):
    """Joint protein-substrate-cofactor prediction with Boltz-2 (``ComplexPredictor``).

    The input is one YAML per candidate: the protein as chain ``A``, the
    substrate as a SMILES ligand ``B``, the cofactor as a CCD ligand ``C`` (so
    its component name and atom names -- ``C4N`` for the nicotinamide hydride
    donor -- are the CCD's, not an invented scheme), and any metals as CCD
    ligands. MSA handling is explicit:

    * a path in ``job.params["msa_path"]`` (``.a3m``/``.csv``) is used as given;
    * otherwise the protein is run single-sequence (``msa: empty``), which
      Boltz itself warns is suboptimal -- and is recorded on every pose;
    * ``use_msa_server=True`` lets Boltz send the sequence to a remote service
      and flips ``runs_remotely``, engaging the submission gate.

    A SMILES ligand comes back as ``LIG1``; it is renamed to the component code
    the request named so the pipeline can find it, and the original is kept.
    """

    binary = "boltz"
    install_hint = ("pip install boltz (https://github.com/jwohlwend/boltz) and "
                    "record the release; the first run downloads model "
                    "checkpoints, which needs network access")
    name = "boltz"
    uses_model_weights = True

    def __init__(self, runner: CommandRunner | None = None,
                 executable_finder: Callable[[str], str | None] | None = None,
                 timeout_s: float | None = 6 * 3600.0, *,
                 accelerator: str = "gpu", recycling_steps: int = 3,
                 sampling_steps: int = 200, use_msa_server: bool = False,
                 model: str = "boltz2") -> None:
        super().__init__(runner, executable_finder, timeout_s)
        if accelerator not in ("gpu", "cpu", "tpu"):
            raise ValueError(f"unsupported accelerator {accelerator!r}")
        if model not in ("boltz1", "boltz2"):
            raise ValueError(f"unsupported model {model!r}")
        self.accelerator = accelerator
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps
        self.use_msa_server = use_msa_server
        self.model = model

    @property
    def version(self) -> str:
        exe = self.executable_finder(self.binary)
        if not exe:
            return "absent"
        return self._read_version([exe, "--version"])

    @property
    def runs_remotely(self) -> bool:
        return self.use_msa_server

    @property
    def registry_keys(self) -> tuple[str, ...]:
        keys = ["boltz.code", "boltz.model_weights"]
        if self.use_msa_server:
            keys.append("boltz.input_database")
        keys.append("boltz.output")
        return tuple(keys)

    # -- input -----------------------------------------------------------
    @staticmethod
    def build_input(job: JointPredictionJob) -> dict[str, Any]:
        """The Boltz YAML document for one job. Pure, so it is testable."""
        if not job.substrate_smiles:
            raise UnsupportedJobError(
                "Boltz needs the substrate as a SMILES string; this job has "
                "none, and a ligand named only by a code would be invented")
        sequences: list[dict[str, Any]] = []
        msa = job.params.get("msa_path") if job.params else None
        protein: dict[str, Any] = {"id": "A", "sequence": job.sequence}
        protein["msa"] = str(msa) if msa else "empty"
        sequences.append({"protein": protein})
        sequences.append({"ligand": {"id": "B", "smiles": job.substrate_smiles}})
        next_id = ord("C")
        if job.cofactor_ligand_code:
            sequences.append({"ligand": {"id": chr(next_id),
                                         "ccd": job.cofactor_ligand_code.upper()}})
            next_id += 1
        elif job.cofactor_smiles:
            sequences.append({"ligand": {"id": chr(next_id),
                                         "smiles": job.cofactor_smiles}})
            next_id += 1
        for metal in job.metals:
            sequences.append({"ligand": {"id": chr(next_id),
                                         "ccd": metal.strip().upper()}})
            next_id += 1
        return {"version": 1, "sequences": sequences}

    def predict(self, job: JointPredictionJob) -> Sequence[RawPose]:
        if job.restraints:
            raise UnsupportedJobError(
                f"this Boltz adapter does not translate the geometric "
                f"restraint(s) {list(job.restraint_names)} into Boltz "
                f"constraints; an untranslated restraint would be recorded "
                f"as enforced without having been")
        document = self.build_input(job)
        exe = self._executable()
        job.out_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", job.candidate_id) or "candidate"
        yaml_path = job.out_dir / f"{stem}.yaml"
        yaml_path.write_text(yaml.safe_dump(document, sort_keys=False),
                             encoding="utf-8")
        out_root = job.out_dir / "boltz"
        argv = [exe, "predict", str(yaml_path), "--out_dir", str(out_root),
                "--model", self.model, "--accelerator", self.accelerator,
                "--devices", "1", "--diffusion_samples", str(job.n_samples),
                "--recycling_steps", str(self.recycling_steps),
                "--sampling_steps", str(self.sampling_steps),
                "--seed", str(job.seed), "--output_format", "mmcif"]
        if self.use_msa_server:
            argv.append("--use_msa_server")
        self._run(argv, cwd=job.out_dir)

        predictions = out_root / f"boltz_results_{stem}" / "predictions" / stem
        models = sorted(
            (p for p in predictions.glob(f"{stem}_model_*.cif")
             if re.fullmatch(rf"{re.escape(stem)}_model_(\d+)\.cif", p.name)),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if not models:
            raise BackendRunError(
                f"boltz reported success but {predictions} holds no "
                f"{stem}_model_<k>.cif")
        msa_note = (f"MSA supplied by the caller: {document['sequences'][0]['protein']['msa']}"
                    if job.params and job.params.get("msa_path")
                    else "remote MSA server" if self.use_msa_server
                    else "single-sequence mode (msa: empty); Boltz warns "
                         "predictions are suboptimal without an MSA")
        raw: list[RawPose] = []
        for path in models:
            k = int(path.stem.rsplit("_", 1)[1])
            raw.append(self._raw_pose(job, stem, path, k, argv, msa_note))
        return raw

    def _raw_pose(self, job: JointPredictionJob, stem: str, path: Path, k: int,
                  argv: Sequence[str], msa_note: str) -> RawPose:
        try:
            structure = read_structure(path, structure_id=f"{stem}_model_{k}")
        except (StructureParseError, OSError) as exc:
            raise BackendRunError(f"{path.name} could not be read: {exc}") from exc
        present = {r.resname.strip().upper()
                   for c in structure.chains for r in c.residues}
        renamed: dict[str, str] = {}
        wanted = job.substrate_ligand_code.strip().upper()
        if wanted != BOLTZ_FIRST_SMILES_LIGAND:
            if BOLTZ_FIRST_SMILES_LIGAND not in present:
                raise BackendRunError(
                    f"{path.name} has no {BOLTZ_FIRST_SMILES_LIGAND} residue; "
                    f"the substrate cannot be identified (present: "
                    f"{sorted(present)})")
            renamed[BOLTZ_FIRST_SMILES_LIGAND] = wanted
        usable = path
        if renamed:
            usable = path.with_name(f"{path.stem}.renamed.cif")
            usable.write_text(
                write_mmcif_text(rename_residues(structure, renamed),
                                 structure.structure_id), encoding="utf-8")

        confidence: dict[str, Any] = {}
        conf_path = path.with_name(f"confidence_{path.stem}.json")
        confidence_present = conf_path.is_file()
        if confidence_present:
            try:
                payload = json.loads(conf_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise BackendRunError(
                    f"{conf_path.name} is not valid JSON: {exc}") from exc
            for src, dst in BOLTZ_CONFIDENCE_KEYS.items():
                if isinstance(payload.get(src), (int, float)):
                    confidence[dst] = float(payload[src])
            for src in ("chains_ptm", "pair_chains_iptm"):
                if src in payload:
                    confidence[src] = payload[src]
        return RawPose(
            path=usable, model_confidence=confidence, sample_index=None,
            restrained_constraints=(),
            extra={
                "engine": "boltz", "engine_version": self.version,
                "model": self.model, "rank_by_confidence": k,
                "raw_output": str(path), "raw_output_sha256": sha256_file(path),
                "confidence_file_present": confidence_present,
                "renamed_residues": renamed, "argv": list(argv),
                "msa": msa_note, "remote_msa_server_used": self.use_msa_server,
                "atom_naming": "SMILES ligand atoms are <ELEMENT><canonical "
                               "rank+1>; the cofactor and metals keep their "
                               "CCD atom names",
                "atom_names_follow_reaction_atom_map": False,
                "restraints_enforced": False,
            })
