"""A language model in the controller's loop, with nothing it can break.

The controller already has the places where judgement lives and the controller
is "deliberately unable to answer for itself": ``widen`` ("which other terms
should be searched?"), and the moments a round ends or an escalation is raised
("what does this mean and what would test it?"). This module is the first thing
that puts a language model behind them. It is designed around what the model is
*not* allowed to do, because that is what makes it safe to wire in:

* **It never executes anything.** Its output is a proposal. The planner applies
  exactly one kind of proposal -- additional search terms for
  ``retrieve_evidence`` -- and records everything else (hypotheses, an
  explanation of an escalation, questions for the operator) as a proposal the
  report labels unapproved.
* **It cannot name an argument outside a short allowlist.** The accepted keys
  are ``families``, ``substrate_synonyms`` and ``engineering_keywords`` of
  ``retrieve_evidence``. Any other key -- ``allow_single_seed``,
  ``submit_to``, a threshold, a tool object -- rejects the *whole* proposal,
  not just that key: a model that reaches for a gate has told you something
  about the rest of its answer. "Widening means more seeds, another database, a
  further family -- never a lowered threshold" is the controller's rule; this
  is where it is enforced for a model.
* **Nothing unpublished goes in the prompt.** Sequences are scanned for and
  refused; the substrate name is withheld unless the operator opts in; the
  substrate structure is never sent. A client that runs remotely is not called
  at all when ``allow_network`` is off, because a prompt cannot be unsent.
* **Every number it states is bound to a cell.** The response goes through
  :class:`~eagent.harness.llm.NumericGuard` against the run's own artifact
  index. A proposal containing an uncited or contradicted quantity is rejected.
* **Hypotheses must be falsifiable, and must cite evidence that exists.**
  ``validate_turn`` rejects a hypothesis without a test and a refuting result;
  an ``evidence_refs`` entry naming an artifact the run never wrote rejects it.
* **Every exchange is audited.** The prompt and the raw response are stored,
  hashed, with the decision and the reasons, whether the proposal was accepted
  or not, so a reviewer can see what the model was shown and what it said.
* **It can fail without failing the run.** An exception from the client, a
  spent call budget, a cost-ceiling breach: each is recorded and the planner
  reports "nothing changed", which sends the controller down the path it would
  have taken with no model at all.
* **Temperature zero, seeded by the run.** Replaying a run replays the request;
  it does not make the answer reproducible across a provider's model updates,
  which is why the response itself is stored.

What none of this does is make a proposal *correct*. A query term the model
suggests is a guess about what the literature calls something; it can be
irrelevant, and the retrieval step decides whether it finds anything.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..connectors.base import AccessPolicy
from ..errors import EAgentError
from ..provenance import sha256_obj, utc_now
from .citation import ArtifactIndex
from .controller import (
    STAGE_INTERFACE, ControllerHooks, Escalation, ResearchController, Stage,
)
from .llm import (
    MEASUREMENT_PATTERN, SYSTEM_PROMPT, LLMClient, ModelTurn, NumericGuard,
    parse_turn, validate_turn,
)

__all__ = [
    "PlannerError",
    "WIDENABLE_ARGUMENTS",
    "Exchange",
    "LLMPlanner",
    "scan_for_unpublished_content",
]


class PlannerError(EAgentError):
    """The planner was misconfigured. Model misbehaviour is a rejection, not this."""


#: The only arguments a model may add to a step, by interface. Anything not
#: listed here is refused by name, and the refusal rejects the whole proposal.
WIDENABLE_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "retrieve_evidence": frozenset({
        "families", "substrate_synonyms", "engineering_keywords"}),
}

#: A run of twenty or more amino-acid letters: a sequence, in any prompt.
_SEQUENCE_RUN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{20,}")

#: What an added search term may look like. Deliberately narrow: a query term
#: is a few words, not a sentence, a URL, a command or a structure.
_TERM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ,;:()/+\-'.]{0,79}$")

MAX_TERMS_PER_KEY = 10

PLANNER_ADDENDUM = """\

You are being asked for a PROPOSAL, not an action. Nothing you write is executed.
Reply with the single JSON object described above. For widening, use one
tool_call to "retrieve_evidence" whose "arguments" may contain only the keys
families, substrate_synonyms and engineering_keywords, each a list of short
search terms (at most 10, at most 80 characters each). Do not propose thresholds,
approvals, seeds, tool paths or any other argument. If you cannot propose
anything useful, answer with "questions" instead.
"""


# ==========================================================================
# the audit trail
# ==========================================================================

@dataclass(frozen=True)
class Exchange:
    """One model call, or one refusal to make it, and what became of it."""

    exchange_id: int
    purpose: str
    client: str
    decision: str                      # accepted | rejected | error | not_called
    reasons: tuple[str, ...]
    prompt_sha256: str | None = None
    response_sha256: str | None = None
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "reasons": list(self.reasons)}


def scan_for_unpublished_content(text: str, *, withheld: Iterable[str] = ()
                                 ) -> list[str]:
    """Why ``text`` must not be sent to a model, or ``[]``.

    Two checks, both on the text as it will be sent: no run of twenty amino-acid
    letters (a sequence, whether or not it is on any authorisation list -- the
    planner never has one to give), and none of the strings the caller says are
    withheld (a substrate structure, a name the operator has not released).
    """
    problems: list[str] = []
    if _SEQUENCE_RUN.search(text.upper()):
        problems.append("the prompt contains what looks like a protein "
                        "sequence")
    lowered = text.lower()
    for secret in withheld:
        secret = (secret or "").strip()
        if len(secret) >= 3 and secret.lower() in lowered:
            problems.append("the prompt contains a string the operator has "
                            "not released to the model")
    return problems


# ==========================================================================
# the planner
# ==========================================================================

class LLMPlanner:
    """Attach a model to a controller's widening and commentary hooks.

    ``disclose_substrate_name`` is off by default: a substrate named in a prompt
    has been disclosed, and for a confidential intermediate that is the whole
    secret. With it off the model is told the name is withheld and can still
    propose family and engineering terms, which do not depend on it.
    """

    def __init__(self, client: LLMClient, *, audit_dir: str | Path | None = None,
                 max_calls: int = 6, disclose_substrate_name: bool = False,
                 access_policy: AccessPolicy | None = None) -> None:
        if max_calls < 1:
            raise PlannerError("max_calls must be at least 1")
        self.client = client
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.max_calls = max_calls
        self.disclose_substrate_name = disclose_substrate_name
        self.access_policy = access_policy
        self.exchanges: list[Exchange] = []
        self.added_terms: dict[str, dict[str, list[str]]] = {}
        self._calls = 0

    # -- wiring ------------------------------------------------------------
    def hooks(self, base: ControllerHooks | None = None) -> ControllerHooks:
        """A hooks object with this planner behind widen and the observers.

        Whatever ``base`` supplies still runs. The planner's widening runs
        only after the base hook declines (returns False or is absent), and its
        added terms are *merged into* whatever arguments the base provides
        rather than replacing them.
        """
        base = base or ControllerHooks()
        planner = self

        def widen(controller: ResearchController, attempt: Any) -> bool:
            if base.widen is not None and base.widen(controller, attempt):
                return True
            return planner.widen(controller, attempt)

        def arguments(controller: ResearchController, interface: str):
            supplied = (base.arguments(controller, interface)
                        if base.arguments is not None else None)
            if supplied is None:
                supplied = controller.static_arguments.get(interface, {})
            return planner.merge_arguments(interface, supplied)

        def on_escalation(controller: ResearchController, esc: Escalation) -> None:
            if base.on_escalation is not None:
                base.on_escalation(controller, esc)
            planner.explain_escalation(controller, esc)

        def on_round_complete(controller: ResearchController) -> None:
            if base.on_round_complete is not None:
                base.on_round_complete(controller)
            planner.interpret_round(controller)

        return ControllerHooks(
            arguments=arguments, repair=base.repair, widen=widen,
            batch_payload=base.batch_payload, results_ready=base.results_ready,
            branch=base.branch, hit_found=base.hit_found,
            on_escalation=on_escalation, on_round_complete=on_round_complete)

    def merge_arguments(self, interface: str, supplied: Mapping[str, Any]
                        ) -> dict[str, Any]:
        """``supplied`` plus the terms the model added, as a union, never a replacement."""
        merged = dict(supplied)
        for key, terms in self.added_terms.get(interface, {}).items():
            existing = [str(t) for t in (merged.get(key) or ())]
            seen = {t.lower() for t in existing}
            merged[key] = existing + [t for t in terms if t.lower() not in seen]
        return merged

    # -- the model call, behind every guard ----------------------------------
    def _ask(self, controller: ResearchController, purpose: str, user_text: str
             ) -> tuple[ModelTurn | None, Exchange]:
        ctx = controller.ctx
        client_name = getattr(self.client, "name", type(self.client).__name__)

        def refuse(reason: str, decision: str = "not_called",
                   prompt_sha: str | None = None) -> tuple[None, Exchange]:
            return None, self._record(purpose, client_name, decision, [reason],
                                      prompt_sha=prompt_sha)

        if self._calls >= self.max_calls:
            return refuse(f"the planner's call budget ({self.max_calls}) is spent")
        breaches = controller.budget_breaches()
        if breaches:
            return refuse("cost ceiling reached: " + "; ".join(breaches))
        withheld = []
        if not self.disclose_substrate_name:
            sub = ctx.task.reaction.substrate
            withheld += [getattr(sub, "name", None) or "",
                         getattr(sub, "isomeric_smiles", None) or "",
                         getattr(ctx.task.reaction.product, "name", None) or ""]
        system = SYSTEM_PROMPT + PLANNER_ADDENDUM
        messages = [{"role": "user", "content": user_text}]
        prompt_text = system + "\n\n" + user_text
        prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        problems = scan_for_unpublished_content(prompt_text, withheld=withheld)
        if problems:
            return refuse("; ".join(problems), "rejected", prompt_sha)
        if getattr(self.client, "runs_remotely", True):
            if not ctx.policy.allow_network:
                return refuse(
                    f"{client_name} runs off this machine and "
                    f"ctx.policy.allow_network is False; nothing was sent",
                    "not_called", prompt_sha)
            policy = self.access_policy or AccessPolicy.from_execution_policy(
                ctx.policy)
            try:
                policy.check_outbound(f"llm:{client_name}",
                                      {"system": system, "user": user_text})
            except EAgentError as exc:
                return refuse(str(exc), "rejected", prompt_sha)

        self._calls += 1
        seed = ctx.seed_for(f"llm:{purpose}:{self._calls}")
        try:
            raw = self.client.complete(system, messages, None, 0.0, seed)
        except Exception as exc:                      # noqa: BLE001
            return None, self._record(
                purpose, client_name, "error",
                [f"{type(exc).__name__}: {exc}"], prompt_sha=prompt_sha,
                prompt_text=prompt_text)
        response_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        guard = NumericGuard(strict=True,
                             index=ArtifactIndex.from_manifest(ctx.manifest))
        report = guard.inspect(raw)
        if not report.clean:
            reasons = [f"uncited quantity: {q}" for q in report.uncited_quantities]
            reasons += [f"broken citation: {c}" for c in report.broken_citations]
            return None, self._record(
                purpose, client_name, "rejected", reasons, prompt_sha=prompt_sha,
                response_sha=response_sha, prompt_text=prompt_text, raw=raw)
        turn = parse_turn(raw)
        problems = validate_turn(turn, controller.registry.names()
                                 if hasattr(controller.registry, "names")
                                 else [])
        if problems:
            return None, self._record(
                purpose, client_name, "rejected", problems,
                prompt_sha=prompt_sha, response_sha=response_sha,
                prompt_text=prompt_text, raw=raw)
        return turn, self._record(
            purpose, client_name, "accepted", [], prompt_sha=prompt_sha,
            response_sha=response_sha, prompt_text=prompt_text, raw=raw)

    def _record(self, purpose: str, client: str, decision: str,
                reasons: Sequence[str], *, prompt_sha: str | None = None,
                response_sha: str | None = None, prompt_text: str | None = None,
                raw: str | None = None) -> Exchange:
        exchange = Exchange(
            exchange_id=len(self.exchanges) + 1, purpose=purpose, client=client,
            decision=decision, reasons=tuple(reasons), prompt_sha256=prompt_sha,
            response_sha256=response_sha)
        self.exchanges.append(exchange)
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            with (self.audit_dir / "llm_exchanges.jsonl").open(
                    "a", encoding="utf-8") as handle:
                handle.write(json.dumps(exchange.to_dict()) + "\n")
            stem = f"{exchange.exchange_id:03d}_{purpose}"
            if prompt_text is not None:
                (self.audit_dir / f"{stem}_prompt.txt").write_text(
                    prompt_text, encoding="utf-8")
            if raw is not None:
                (self.audit_dir / f"{stem}_response.txt").write_text(
                    raw, encoding="utf-8")
        return exchange

    # -- what the model is shown ------------------------------------------
    def _context(self, controller: ResearchController, headline: str) -> str:
        ctx = controller.ctx
        lines = [headline, "",
                 f"reaction class: {ctx.task.reaction.reaction_class.value}"]
        if self.disclose_substrate_name:
            name = getattr(ctx.task.reaction.substrate, "name", None)
            lines.append(f"substrate name: {name or 'unnamed'}")
        else:
            lines.append("substrate: withheld by the operator (do not guess it)")
        if controller.attempts:
            last = controller.attempts[-1]
            lines += [f"last step: {last.interface} at {last.stage.value}, "
                      f"outcome {last.failure.value}, status {last.status}",
                      f"message: {last.message[:300]}"]
        result = controller.results.get(
            controller.attempts[-1].interface) if controller.attempts else None
        if result is not None:
            for flag in list(result.qc_flags)[:8]:
                lines.append(f"flag {flag.code} [{flag.severity.value}]: "
                             f"{flag.message[:200]}")
        index = ArtifactIndex.from_manifest(ctx.manifest)
        if index.keys:
            lines += ["", "artifacts you may cite (key, sha256 prefix):"]
            for key in index.keys[:20]:
                entry = index.resolve(key)
                lines.append(f"  {key} sha256={(entry.sha256 or '')[:12]}")
        args = controller.arguments_for("retrieve_evidence")
        merged = self.merge_arguments("retrieve_evidence", args)
        for key in sorted(WIDENABLE_ARGUMENTS["retrieve_evidence"]):
            vals = merged.get(key)
            if isinstance(vals, (list, tuple)):
                lines.append(f"current {key}: {', '.join(map(str, vals)) or '(none)'}")
        return "\n".join(lines)

    # -- widening: the one proposal that is applied -------------------------
    def widen(self, controller: ResearchController, attempt: Any) -> bool:
        """Ask for more search terms; apply those that survive validation.

        Returns True only if at least one new term was accepted, which is the
        controller's own test for "something changed". Declines (False) for any
        stage this planner cannot widen, so the controller carries the shortfall
        forward as exploration probes exactly as it would without a model.
        """
        target = controller._widen_target or Stage.RETRIEVE_EVIDENCE
        interface = STAGE_INTERFACE.get(target)
        if interface not in WIDENABLE_ARGUMENTS:
            self._record("widen", getattr(self.client, "name", "?"), "not_called",
                         [f"no model-proposable widening exists for "
                          f"{interface or target.value}; the controller's own "
                          f"widening applies"])
            return False
        headline = (f"The search at {target.value} found too little to act on. "
                    f"Propose additional search terms for {interface}.")
        turn, exchange = self._ask(controller, "widen",
                                   self._context(controller, headline))
        if turn is None:
            return False
        proposal = self._widening_from(turn, interface, controller)
        if isinstance(proposal, str):
            self.exchanges[-1] = Exchange(
                exchange.exchange_id, exchange.purpose, exchange.client,
                "rejected", (proposal,), exchange.prompt_sha256,
                exchange.response_sha256, exchange.at)
            self._rewrite_audit(self.exchanges[-1])
            controller.llm_proposals.append({
                "kind": "widen_search", "status": "rejected",
                "reason": proposal, "exchange": exchange.exchange_id})
            return False
        if not proposal:
            controller.llm_proposals.append({
                "kind": "widen_search", "status": "no_new_terms",
                "exchange": exchange.exchange_id})
            return False
        bucket = self.added_terms.setdefault(interface, {})
        for key, terms in proposal.items():
            have = bucket.setdefault(key, [])
            have += [t for t in terms if t.lower() not in {h.lower() for h in have}]
        controller.llm_proposals.append({
            "kind": "widen_search", "status": "applied", "terms": proposal,
            "exchange": exchange.exchange_id,
            "note": "search terms only; retrieval decides whether they find "
                    "anything"})
        controller.manifest.notes.append(
            f"{utc_now()} language-model widening applied to {interface}: "
            f"{json.dumps(proposal)} (exchange {exchange.exchange_id}, "
            f"prompt {exchange.prompt_sha256})")
        return True

    def _widening_from(self, turn: ModelTurn, interface: str,
                       controller: ResearchController
                       ) -> dict[str, list[str]] | str:
        """The new terms in ``turn``, or a string saying why it is rejected."""
        calls = [c for c in turn.tool_calls]
        if not calls:
            return "the response proposed no tool_call to widen with"
        allowed = WIDENABLE_ARGUMENTS[interface]
        existing = self.merge_arguments(
            interface, controller.arguments_for(interface))
        out: dict[str, list[str]] = {}
        for call in calls:
            if call.interface != interface:
                return (f"the response called {call.interface!r}; only "
                        f"{interface!r} may be widened by a model")
            for key, value in call.arguments.items():
                if key not in allowed:
                    return (f"argument {key!r} is not one a model may set "
                            f"(allowed: {sorted(allowed)}); the whole proposal "
                            f"is rejected")
                if not isinstance(value, list) or not all(
                        isinstance(t, str) for t in value):
                    return f"{key} must be a list of strings"
                if len(value) > MAX_TERMS_PER_KEY:
                    return (f"{key} has {len(value)} terms; the cap is "
                            f"{MAX_TERMS_PER_KEY}")
                have = {str(t).lower() for t in (existing.get(key) or ())}
                for term in value:
                    term = term.strip()
                    if not _TERM.match(term):
                        return (f"{key} term {term!r} is not a short plain "
                                f"search phrase")
                    if _SEQUENCE_RUN.search(term.upper()) \
                            or MEASUREMENT_PATTERN.search(term):
                        return (f"{key} term {term!r} contains a sequence or "
                                f"a quantity, which a search term must not")
                    if term.lower() in have or term.lower() in {
                            t.lower() for t in out.get(key, [])}:
                        continue
                    out.setdefault(key, []).append(term)
        return out

    def _rewrite_audit(self, exchange: Exchange) -> None:
        if self.audit_dir is None:
            return
        path = self.audit_dir / "llm_exchanges.jsonl"
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        for i, row in enumerate(lines):
            if row["exchange_id"] == exchange.exchange_id:
                lines[i] = exchange.to_dict()
        path.write_text("".join(json.dumps(r) + "\n" for r in lines),
                        encoding="utf-8")

    # -- commentary: recorded, never applied --------------------------------
    def explain_escalation(self, controller: ResearchController,
                           esc: Escalation) -> None:
        headline = (f"The run escalated at {esc.stage.value} ({esc.interface}), "
                    f"failure kind {esc.kind.value}: {esc.reason[:300]}. "
                    f"Explain in plain terms what the operator must decide, "
                    f"as questions.")
        turn, exchange = self._ask(controller, "escalation",
                                   self._context(controller, headline))
        if turn is None:
            return
        controller.llm_proposals.append({
            "kind": "escalation_explanation", "status": "proposal_unapproved",
            "reasoning": turn.reasoning, "questions": list(turn.questions),
            "exchange": exchange.exchange_id})

    def interpret_round(self, controller: ResearchController) -> None:
        # The hook fires before the run leaves the stage that ended the round:
        # local engineering means a hit, the diagnosis stage means none.
        hit = controller.stage is Stage.LOCAL_ENGINEERING
        headline = ("The round has ended " + ("with a confirmed hit." if hit
                    else "with no confirmed hit.")
                    + " Propose hypotheses that could explain the outcome. "
                      "Each needs a test and the result that would refute it, "
                      "and may cite only artifacts listed below.")
        turn, exchange = self._ask(controller, "round",
                                   self._context(controller, headline))
        if turn is None:
            return
        index = ArtifactIndex.from_manifest(controller.ctx.manifest)
        accepted: list[dict[str, Any]] = []
        for h in turn.hypotheses:
            missing = [r for r in h.evidence_refs if r not in index]
            if missing:
                controller.llm_proposals.append({
                    "kind": "hypothesis", "status": "rejected",
                    "reason": f"cites artifact(s) the run never wrote: {missing}",
                    "statement": h.statement, "exchange": exchange.exchange_id})
                continue
            accepted.append(h.__dict__)
        controller.llm_proposals.append({
            "kind": "round_interpretation", "status": "proposal_unapproved",
            "reasoning": turn.reasoning, "hypotheses": accepted,
            "questions": list(turn.questions), "exchange": exchange.exchange_id})

    # -- summary -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {"client": getattr(self.client, "name", "?"),
                "calls": self._calls, "max_calls": self.max_calls,
                "exchanges": [e.to_dict() for e in self.exchanges],
                "digest": sha256_obj([e.to_dict() for e in self.exchanges])}
