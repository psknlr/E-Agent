"""Run context handed to every scientific interface.

Holds the task, the working directory, the template library, the manifest and
the execution policy. Interfaces receive it rather than reaching for globals,
so a run is reproducible from the context alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .envelope import ToolResult
from .errors import ApprovalRequiredError
from .provenance import RunManifest, derive_seed, utc_now
from .schemas import TaskSpec


@dataclass
class ExecutionPolicy:
    """What the run is permitted to do."""

    allow_network: bool = False
    allow_external_binaries: bool = True
    allow_gpu_models: bool = True
    dry_run: bool = False
    strict: bool = True              # unresolved required field -> raise, never guess
    max_retries: int = 1
    cost_ceiling: dict[str, float] = field(default_factory=dict)
    allow_commercial_use: bool = False   # gates licence-restricted model weights


@dataclass
class RunContext:
    """Everything a step needs, and nothing it should not reach for."""

    task: TaskSpec
    workdir: Path
    manifest: RunManifest
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    templates: Any = None                     # TemplateLibrary, injected by the harness
    config: dict[str, Any] = field(default_factory=dict)
    logger: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        p = self.workdir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dir(self, *parts: str) -> Path:
        p = self.workdir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- determinism -------------------------------------------------------
    def seed_for(self, step_id: str) -> int:
        return derive_seed(self.manifest.global_seed, step_id)

    # -- gates -------------------------------------------------------------
    def require_approval(self, gate: str, detail: str = "") -> None:
        if self.task.approval.state(gate) or self.manifest.approved(gate):
            return
        raise ApprovalRequiredError(gate, detail)

    def require_fields(self, gate: str) -> None:
        self.task.require(gate)

    # -- logging -----------------------------------------------------------
    def log(self, message: str) -> None:
        if self.logger:
            self.logger(message)
        self.manifest.notes.append(f"{utc_now()} {message}")
