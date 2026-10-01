"""Run manifest: the record that makes a claim recomputable.

Every step appends to a manifest holding input hashes, database snapshots,
software and model versions, seeds, parameters, residue-numbering maps and
cost. A candidate that reaches a synthesis order must be traceable back
through this file to the sequence record and the evidence that justified it.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .envelope import Provenance, ToolResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sequence_hash(seq: str) -> str:
    """Hash of a protein sequence, normalised to uppercase without whitespace.

    This is the join key for the data layer. Two records refer to the same
    protein only if this value matches; an accession alone is not enough,
    because accessions get re-annotated and isoforms share names.
    """
    norm = "".join(seq.split()).upper()
    if not norm:
        raise ValueError("empty sequence")
    return "sha256:" + sha256_text(norm)


def _git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd or os.getcwd()),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def capture_environment() -> dict[str, Any]:
    """Snapshot of the machine and interpreter, recorded once per run."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": _safe_user(),
        "cwd": os.getcwd(),
        "git_commit": _git_commit(),
    }


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


@dataclass
class StepRecord:
    """One executed interface call."""

    step_id: str
    interface: str
    status: str
    started_at: str
    finished_at: str
    provenance: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    qc_flags: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    """Append-only record of a single agent run."""

    run_id: str
    task_id: str
    created_at: str = field(default_factory=utc_now)
    environment: dict[str, Any] = field(default_factory=capture_environment)
    task_input_sha256: str | None = None
    global_seed: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    databases: dict[str, str] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    cost_total: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- mutation ---------------------------------------------------------
    def record(self, step_id: str, interface: str, result: ToolResult,
               started_at: str, finished_at: str | None = None) -> StepRecord:
        prov = result.provenance or Provenance(tool=interface)
        prov.started_at = prov.started_at or started_at
        prov.finished_at = prov.finished_at or (finished_at or utc_now())
        rec = StepRecord(
            step_id=step_id,
            interface=interface,
            status=result.status.value,
            started_at=prov.started_at,
            finished_at=prov.finished_at,
            provenance=prov.to_dict(),
            artifacts=[a.to_dict() for a in result.artifacts],
            qc_flags=[f.to_dict() for f in result.qc_flags],
            uncertainty=[u.to_dict() for u in result.uncertainty],
            message=result.message,
        )
        self.steps.append(rec)
        self.databases.update(prov.databases)
        self.models.update(prov.models)
        for k, v in (prov.cost or {}).items():
            if isinstance(v, (int, float)):
                self.cost_total[k] = self.cost_total.get(k, 0.0) + float(v)
        return rec

    def record_approval(self, gate: str, decision: str, actor: str,
                        detail: str = "") -> None:
        self.approvals.append({
            "gate": gate, "decision": decision, "actor": actor,
            "detail": detail, "at": utc_now(),
        })

    def approved(self, gate: str) -> bool:
        return any(a["gate"] == gate and a["decision"] == "approve"
                   for a in self.approvals)

    # -- io ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False,
                                default=str), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        steps = [StepRecord(**s) for s in raw.pop("steps", [])]
        m = cls(**raw)
        m.steps = steps
        return m


def derive_seed(global_seed: int, step_id: str) -> int:
    """Deterministic per-step seed, so one step's rerun does not shift others."""
    h = sha256_text(f"{global_seed}:{step_id}")
    return int(h[:8], 16)
