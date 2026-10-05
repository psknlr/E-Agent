"""The three human decision points, and the block that makes them real.

A gate that is "checked" by reading a boolean out of a YAML file is not a
decision point. Someone sets ``synthesis_authorized: true`` while debugging,
forgets, and six weeks later ninety-six genes are ordered against a spec no
human ever looked at. The three gates here are therefore cleared only by a
recorded grant: a named actor, a timestamp, and the payload they were shown,
all appended to the :class:`~eagent.provenance.RunManifest` and persisted to
disk so the record survives the process that made it.

The three points, and why each one is a human's and not the agent's:

``reaction_spec_confirmed``
    Which molecule, which configuration, which conditions. Getting this wrong
    does not degrade the run, it redefines it, and every mined sequence after
    that point is evidence about a different project.

``synthesis_authorized``
    This is the step that spends money and lab time. The operator authorises a
    specific batch at a specific cost; "approve the pipeline" is not a thing
    they can do here.

``functional_criteria_confirmed``
    What counts as a hit, fixed before the data exist. Confirmed afterwards it
    is not a criterion, it is a description of whatever came back.

:func:`guard_batch_selection` is the hard block. It is deliberately stricter
than :meth:`eagent.context.RunContext.require_approval`, which also accepts
the task-file flag: the flag says somebody wrote ``true`` in a file, and the
manifest grant says a named person decided.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import ApprovalRequiredError
from ..provenance import RunManifest, sha256_obj, utc_now

__all__ = [
    "APPROVAL_GATES",
    "ApprovalQueue",
    "ApprovalRequest",
    "BATCH_GATE",
    "CRITERIA_GATE",
    "DECISION_POINTS",
    "Decision",
    "DecisionPoint",
    "REACTION_GATE",
    "RequestKind",
    "batch_cost_payload",
    "guard_batch_selection",
]

REACTION_GATE = "reaction_spec_confirmed"
BATCH_GATE = "synthesis_authorized"
CRITERIA_GATE = "functional_criteria_confirmed"

#: The only three gates. Matching the field names of
#: :class:`eagent.schemas.reaction.Approval` and the keys of
#: :data:`eagent.schemas.reaction.GATE_REQUIREMENTS` is load-bearing: a fourth
#: spelling of a gate would be a gate nothing ever checks.
APPROVAL_GATES: tuple[str, str, str] = (REACTION_GATE, BATCH_GATE, CRITERIA_GATE)


class Decision(str, enum.Enum):
    """State of one request.

    ``approve`` and ``deny`` are the strings :meth:`RunManifest.approved`
    matches on, so they are not renameable without breaking every manifest
    already written.
    """

    PENDING = "pending"
    APPROVE = "approve"
    DENY = "deny"

    @property
    def is_final(self) -> bool:
        return self is not Decision.PENDING


class RequestKind(str, enum.Enum):
    """Whether a request opens a gate or is an errand for the operator.

    Both queue, because both are things only a human can do, but only a
    ``gate`` grant is allowed to unblock a step. A curator fetching a structure
    file is not an authorisation to spend the synthesis budget.
    """

    GATE = "gate"
    OPERATOR_TASK = "operator_task"


@dataclass(frozen=True)
class DecisionPoint:
    """What a human is deciding, and what they must be shown to decide it."""

    gate: str
    question: str
    must_show: tuple[str, ...]
    consequence_if_wrong: str


DECISION_POINTS: dict[str, DecisionPoint] = {
    REACTION_GATE: DecisionPoint(
        gate=REACTION_GATE,
        question=("Is this the reaction the project is about: this substrate "
                  "structure, this product structure, this configuration?"),
        must_show=("reaction.substrate.isomeric_smiles",
                   "reaction.product.isomeric_smiles",
                   "reaction.product.target_stereochemistry",
                   "reaction.product.creates_new_stereocenter",
                   "reaction.atom_mapped_reaction_smiles",
                   "assumptions"),
        consequence_if_wrong=("every sequence mined, every complex modelled and "
                              "every gene ordered afterwards is evidence about "
                              "the wrong molecule"),
    ),
    BATCH_GATE: DecisionPoint(
        gate=BATCH_GATE,
        question=("Authorise this batch: these constructs, these wells, this "
                  "cost?"),
        must_show=("n_constructs", "n_candidate_wells", "n_control_wells",
                   "n_plates", "cofactor_conditions", "replicates",
                   "controls", "cost"),
        consequence_if_wrong=("money and lab time are spent on a round whose "
                              "composition nobody reviewed"),
    ),
    CRITERIA_GATE: DecisionPoint(
        gate=CRITERIA_GATE,
        question=("Which experimental result counts as supporting a functional "
                  "claim, decided before the data exist?"),
        must_show=("assay_template_id", "tier", "method",
                   "confirms_product_identity", "positive_criteria",
                   "limit_of_detection", "controls_required"),
        consequence_if_wrong=("the criterion gets fitted to whatever the plate "
                              "produced, and the round has no endpoint"),
    ),
}


def batch_cost_payload(
    *,
    n_constructs: int,
    n_candidate_wells: int,
    n_control_wells: int,
    n_plates: int,
    cofactor_conditions: int,
    replicates: int,
    controls: Iterable[str] = (),
    currency: str | None = None,
    cost_per_construct: float | None = None,
    cost_per_plate: float | None = None,
) -> dict[str, Any]:
    """Assemble what the operator must see before authorising a batch.

    Prices are optional and default to ``None`` with a note. This harness does
    not know what a gene costs at this site, and a plausible figure in an
    authorisation request is worse than no figure: it would be approved as
    though it had been checked.

    A total is computed only when every price it needs is present. Treating a
    missing plate price as zero would quote a total that is confidently too
    low, which is the shape of mistake an authorisation request must not make.
    """
    missing_prices = [
        label for label, price, count in
        (("per_construct", cost_per_construct, n_constructs),
         ("per_plate", cost_per_plate, n_plates))
        if price is None and count
    ]
    total: float | None = None
    if not missing_prices and (cost_per_construct is not None
                               or cost_per_plate is not None):
        total = ((cost_per_construct or 0.0) * n_constructs
                 + (cost_per_plate or 0.0) * n_plates)
    payload: dict[str, Any] = {
        "n_constructs": int(n_constructs),
        "n_candidate_wells": int(n_candidate_wells),
        "n_control_wells": int(n_control_wells),
        "n_wells_total": int(n_candidate_wells) + int(n_control_wells),
        "n_plates": int(n_plates),
        "cofactor_conditions": int(cofactor_conditions),
        "replicates": int(replicates),
        "controls": list(controls),
        "cost": {
            "currency": currency,
            "per_construct": cost_per_construct,
            "per_plate": cost_per_plate,
            "estimated_total": total,
        },
    }
    if total is None:
        needed = ", ".join(missing_prices) or "per-construct and per-plate"
        payload["cost"]["note"] = (
            f"unknown: this harness has no price list. A curator or the "
            f"operator must supply {needed} cost(s) before this request means "
            f"anything financially; the missing one is not zero.")
    return payload


@dataclass
class ApprovalRequest:
    """One thing a human must decide, and the decision once made."""

    request_id: str
    gate: str
    kind: RequestKind = RequestKind.GATE
    requested_by: str = "controller"
    requested_at: str = field(default_factory=utc_now)
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    decision: Decision = Decision.PENDING
    actor: str | None = None
    decided_at: str | None = None
    reason: str = ""

    @property
    def is_pending(self) -> bool:
        return self.decision is Decision.PENDING

    @property
    def is_granted(self) -> bool:
        return self.decision is Decision.APPROVE

    @property
    def payload_sha256(self) -> str:
        """Hash of what was decided, so a grant cannot migrate to other work.

        An approval is an approval *of something*. Without this, the only
        thing a later check can match on is the gate name, and every gate
        name is shared by every batch that will ever be proposed.
        """
        return sha256_obj(dict(self.payload))

    def covers(self, payload: Mapping[str, Any] | None) -> bool:
        """Whether this decision was made about ``payload``.

        ``None`` means the caller did not say what it is about, which cannot
        be matched against anything and so is never covered by a specific
        decision; the gate-level reading is left to the caller.
        """
        if payload is None:
            return False
        return self.payload_sha256 == sha256_obj(dict(payload))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["decision"] = self.decision.value
        return d

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ApprovalRequest":
        data = dict(raw)
        data["kind"] = RequestKind(data.get("kind", RequestKind.GATE.value))
        data["decision"] = Decision(data.get("decision", Decision.PENDING.value))
        known = {f for f in cls.__dataclass_fields__}      # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"approval queue entry carries unknown field(s) {sorted(unknown)}; "
                f"refusing to load a queue this version does not understand "
                f"rather than dropping a decision silently")
        return cls(**data)


def _request_id(gate: str, payload: Mapping[str, Any], requested_at: str) -> str:
    """Stable id over gate, payload and request time.

    The payload is part of the identity because an approval is an approval *of
    something*. Re-asking with a changed batch must produce a new request, not
    inherit the grant given for the old one.
    """
    return sha256_obj({"gate": gate, "payload": dict(payload),
                       "at": requested_at})[:16]


class ApprovalQueue:
    """Pending and decided human decisions, persisted to disk.

    Persisted because the three decisions outlive the process: a batch
    authorisation is requested on Monday and granted on Thursday, after the run
    has been stopped and resumed twice. A queue that lived in memory would
    quietly re-ask -- or worse, re-run the gate check against a task flag and
    pass.
    """

    def __init__(self, path: str | Path, manifest: RunManifest | None = None,
                 requests: Iterable[ApprovalRequest] = ()) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self._requests: list[ApprovalRequest] = list(requests)

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path,
             manifest: RunManifest | None = None) -> "ApprovalQueue":
        """Read an existing queue, or start an empty one at this path."""
        p = Path(path)
        if not p.exists():
            return cls(p, manifest)
        raw = json.loads(p.read_text(encoding="utf-8"))
        items = [ApprovalRequest.from_dict(r) for r in raw.get("requests", [])]
        return cls(p, manifest, items)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"requests": [r.to_dict() for r in self._requests]},
                       indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        return self.path

    # -- queries -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self):
        return iter(self._requests)

    def all(self) -> list[ApprovalRequest]:
        return list(self._requests)

    def get(self, request_id: str) -> ApprovalRequest | None:
        for r in self._requests:
            if r.request_id == request_id:
                return r
        return None

    def pending(self, gate: str | None = None) -> list[ApprovalRequest]:
        return [r for r in self._requests
                if r.is_pending and (gate is None or r.gate == gate)]

    def for_gate(self, gate: str) -> list[ApprovalRequest]:
        return [r for r in self._requests if r.gate == gate]

    def latest_for(self, gate: str,
                   payload: Mapping[str, Any] | None = None) -> ApprovalRequest | None:
        """The most recent decided request for this gate, or for this payload.

        The most recent decision is what governs. Reading "has this gate ever
        been granted" instead lets a grant given for a small pilot release a
        later batch that was explicitly refused.
        """
        matches = [r for r in self._requests
                   if r.gate == gate and r.decision.is_final
                   and (payload is None or r.covers(payload))]
        return matches[-1] if matches else None

    def is_granted(self, gate: str,
                   payload: Mapping[str, Any] | None = None) -> bool:
        """Whether the work described by ``payload`` is approved at this gate.

        With a payload, the decision must have been made about that exact
        payload: a grant for one batch does not carry to a different one, and
        a changed construct list, well count, budget or criterion produces a
        different payload and therefore needs its own decision.

        Without a payload this falls back to the gate-level reading, but a
        later refusal anywhere on the gate still shuts it. An older grant
        outranking a newer denial is how an explicitly refused batch gets
        released.
        """
        if payload is not None:
            decided = self.latest_for(gate, payload)
            if decided is not None:
                return decided.is_granted
            # No decision about this payload. A grant recorded for other work
            # says nothing about it, and neither does the manifest.
            return False

        latest = self.latest_for(gate)
        if latest is not None:
            return latest.is_granted
        return bool(self.manifest is not None and self.manifest.approved(gate))

    def is_denied(self, gate: str,
                  payload: Mapping[str, Any] | None = None) -> bool:
        """Whether the latest decision on this gate, or payload, was a refusal."""
        decided = self.latest_for(gate, payload)
        return decided is not None and decided.decision is Decision.DENY

    # -- mutation ----------------------------------------------------------
    def request(self, gate: str, *, requested_by: str = "controller",
                detail: str = "", payload: Mapping[str, Any] | None = None,
                kind: RequestKind | str = RequestKind.GATE) -> ApprovalRequest:
        """Queue a decision for a human.

        Re-requesting a gate that already has an identical pending request
        returns the existing one rather than piling up duplicates, which is
        what a controller re-entering a stage after a retry would otherwise do.
        """
        kind_enum = RequestKind(kind)
        if kind_enum is RequestKind.GATE and gate not in APPROVAL_GATES:
            raise ValueError(
                f"'{gate}' is not one of the three approval gates "
                f"{APPROVAL_GATES}; queue an operator task instead of "
                f"inventing a fourth gate")
        body = dict(payload or {})
        at = utc_now()
        existing = [r for r in self._requests
                    if r.gate == gate and r.is_pending and r.payload == body]
        if existing:
            return existing[0]
        req = ApprovalRequest(
            request_id=_request_id(gate, body, at),
            gate=gate, kind=kind_enum, requested_by=requested_by,
            requested_at=at, detail=detail, payload=body,
        )
        self._requests.append(req)
        if self.manifest is not None:
            self.manifest.notes.append(
                f"{at} approval requested at gate '{gate}' "
                f"(request {req.request_id}): {detail}")
        self.save()
        return req

    def grant(self, request_id_or_gate: str, actor: str,
              reason: str = "") -> ApprovalRequest:
        """Record a human's approval, with who they were and when.

        An empty ``actor`` is refused: "approved" with nobody's name on it is
        the state this whole module exists to prevent.
        """
        return self._decide(request_id_or_gate, Decision.APPROVE, actor, reason)

    def deny(self, request_id_or_gate: str, actor: str,
             reason: str = "") -> ApprovalRequest:
        """Record a refusal. A denied gate stays shut until a new request."""
        return self._decide(request_id_or_gate, Decision.DENY, actor, reason)

    def _decide(self, key: str, decision: Decision, actor: str,
                reason: str) -> ApprovalRequest:
        if not (actor or "").strip():
            raise ValueError(
                f"a {decision.value} decision must name the actor; an "
                f"anonymous approval is not a human decision point")
        req = self.get(key)
        if req is None:
            pending = self.pending(key)
            if not pending:
                raise KeyError(
                    f"no pending approval request for '{key}'; a decision must "
                    f"answer a request that states what was being decided")
            req = pending[-1]
        if req.decision.is_final:
            raise ValueError(
                f"request {req.request_id} ({req.gate}) was already "
                f"{req.decision.value}d by {req.actor}; re-deciding would "
                f"overwrite the record of what was agreed")
        req.decision = decision
        req.actor = actor.strip()
        req.decided_at = utc_now()
        req.reason = reason
        if self.manifest is not None:
            detail = reason or req.detail
            self.manifest.record_approval(req.gate, decision.value, req.actor,
                                          req.reason or req.detail,
                                          payload_sha256=req.payload_sha256,
                                          request_id=req.request_id)
        self.save()
        return req

    # -- enforcement -------------------------------------------------------
    def require(self, gate: str, detail: str = "",
                payload: Mapping[str, Any] | None = None) -> None:
        """Raise unless the work described by ``payload`` is approved here.

        Pass the payload wherever one exists. Requiring only the gate accepts
        any decision ever recorded under that name, which is not the question
        a batch about to be ordered is asking.
        """
        if self.is_granted(gate, payload):
            return
        if payload is not None and self.is_denied(gate, payload):
            raise ApprovalRequiredError(
                gate, f"{detail} this exact request was refused; a grant "
                      f"recorded for different work does not release it")
        if payload is not None:
            wanted = sha256_obj(dict(payload))
            others = [r for r in self._requests
                      if r.gate == gate and r.is_granted
                      and r.payload_sha256 != wanted]
            if others:
                # Without this the operator is told "no request has even been
                # raised" immediately after they approved something, and the
                # natural reading is that the tool lost their decision rather
                # than that the work changed under it.
                raise ApprovalRequiredError(
                    gate,
                    f"{detail}a grant exists for DIFFERENT work "
                    f"(request {others[-1].request_id}, approved by "
                    f"{others[-1].actor}), which does not carry over to this "
                    f"one. What is being decided changed, so it needs its own "
                    f"decision.")
        pending = self.pending(gate)
        where = (f" request {pending[-1].request_id} is still pending"
                 if pending else " no request has even been raised")
        raise ApprovalRequiredError(gate, detail + where)

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path),
                "requests": [r.to_dict() for r in self._requests]}


def guard_batch_selection(manifest: RunManifest,
                          queue: ApprovalQueue | None = None,
                          *, gate: str = BATCH_GATE,
                          payload: Mapping[str, Any] | None = None) -> None:
    """Hard block in front of batch selection.

    Called by the controller before it will route to ``select_batch`` at all,
    so there is no path from "the pipeline ran" to "genes were ordered" that
    does not pass a recorded grant. The check is on the manifest, not on
    :class:`~eagent.schemas.reaction.Approval`: the task flag can be set by
    editing a file, and this gate has to mean that a named person said yes to
    a specific batch at a specific cost.

    Raises :class:`~eagent.errors.ApprovalRequiredError`.
    """
    wanted = sha256_obj(dict(payload)) if payload is not None else None
    decisions = [a for a in manifest.approvals
                 if a.get("gate") == gate
                 and str(a.get("actor") or "").strip()
                 and (wanted is None or a.get("payload_sha256") == wanted)]
    # The LATEST decision governs. Scanning for any grant ever recorded lets
    # an approval given for a small pilot release a later, different batch
    # that a human explicitly refused.
    if decisions and decisions[-1].get("decision") == "approve":
        return
    denied = [a for a in decisions if a.get("decision") == "deny"]
    detail = (
        "batch selection spends the construct budget. The manifest carries no "
        "approval with an actor for this gate")
    if wanted is not None:
        detail += f" and this exact batch (payload {wanted[:12]})"
        other = [a for a in manifest.approvals
                 if a.get("gate") == gate and a.get("decision") == "approve"
                 and a.get("payload_sha256") not in (None, wanted)]
        if other:
            detail += ("; a grant exists for a DIFFERENT batch, which does not "
                       "carry over: re-request approval for this one")
    if denied:
        detail += (f"; it was denied by {denied[-1].get('actor')} "
                   f"({denied[-1].get('detail') or 'no reason recorded'})")
    elif queue is not None and queue.pending(gate):
        detail += (f"; request {queue.pending(gate)[-1].request_id} is waiting "
                   f"for a decision")
    else:
        detail += "; no request has been raised"
    raise ApprovalRequiredError(gate, detail)
