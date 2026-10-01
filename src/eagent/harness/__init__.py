"""Orchestration: one research controller, deterministic tools, an independent verifier.

Deliberately not a committee of role-playing agents arguing with each other. A
single controller plans and interprets; everything measurable is computed by
code or by a dedicated scientific model; a separate verifier checks the result
without reusing the producing step's conclusions.
"""

from __future__ import annotations

from .llm import (
    CallbackClient, EchoClient, GuardReport, Hypothesis, LLMClient, ModelTurn,
    NumericGuard, SYSTEM_PROMPT, ToolCall, parse_turn, validate_turn,
)

__all__ = [
    "CallbackClient", "EchoClient", "GuardReport", "Hypothesis", "LLMClient",
    "ModelTurn", "NumericGuard", "SYSTEM_PROMPT", "ToolCall", "parse_turn",
    "validate_turn",
]
