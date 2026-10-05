"""Experimental batch: candidate slots, their purpose, and the control plan."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BatchRole(str, enum.Enum):
    """Why a slot was spent."""

    HIGH_EVIDENCE = "high_evidence"           # most likely to succeed
    DIVERSITY = "diversity"                   # family / pocket coverage
    UNCERTAINTY_PROBE = "uncertainty_probe"   # mechanistically plausible, model unsure
    CONTROL = "control"
    VARIANT = "variant"                       # round-2 engineered variant


class BatchMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: int
    candidate_id: str
    role: BatchRole
    family: str | None = None
    sequence_cluster_id: str | None = None
    utility: float | None = None
    selection_reason: str = ""
    construct_notes: str | None = None


class ControlItem(BaseModel):
    """A control, and what it is actually capable of proving."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str                       # no_enzyme | empty_vector | no_cofactor | positive_enzyme
    demonstrates: str = Field(
        ..., description="'assay system works' and 'target substrate is turned over' "
                         "are different claims and must not be conflated"
    )
    requires_new_construct: bool = False
    occupies_batch_slot: bool = False


class BatchPlan(BaseModel):
    """A round of experiments, including its measurement footprint."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    round_number: int = 1
    members: list[BatchMember] = Field(default_factory=list)
    controls: list[ControlItem] = Field(default_factory=list)
    cofactor_conditions: int = 2
    replicates: int = 3
    requested_slots: int = 96
    family_quota: dict[str, int] = Field(default_factory=dict)
    shortfall_reason: str | None = Field(
        None,
        description="Set when fewer candidates were selected than requested. The "
                    "batch is reported short rather than padded with weak entries.",
    )

    @property
    def n_candidates(self) -> int:
        return len([m for m in self.members if m.role is not BatchRole.CONTROL])

    @property
    def measurement_units(self) -> int:
        """Wells, not genes: candidates x cofactor conditions x replicates."""
        return self.n_candidates * max(1, self.cofactor_conditions) * max(1, self.replicates)

    @property
    def control_units(self) -> int:
        return len(self.controls) * max(1, self.cofactor_conditions) * max(1, self.replicates)

    @property
    def total_units(self) -> int:
        return self.measurement_units + self.control_units

    @property
    def is_short(self) -> bool:
        return self.n_candidates < self.requested_slots

    def role_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.members:
            out[m.role.value] = out.get(m.role.value, 0) + 1
        return out

    @model_validator(mode="after")
    def _short_batch_must_explain(self) -> "BatchPlan":
        if self.is_short and not self.shortfall_reason:
            raise ValueError(
                "a batch smaller than requested must record why; padding to the "
                "target by lowering quality standards is not permitted"
            )
        new_constructs = [c for c in self.controls if c.requires_new_construct]
        occupying = [c for c in new_constructs if not c.occupies_batch_slot]
        if occupying:
            raise ValueError(
                f"controls needing new genes must occupy batch slots: "
                f"{[c.name for c in occupying]}"
            )
        return self
