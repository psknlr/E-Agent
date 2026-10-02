"""Carrying candidates between steps without losing their types.

Every interface writes its output into ``ToolResult.data`` so the manifest can
serialise it. Serialising a :class:`~eagent.schemas.candidate.Candidate` turns
it into a plain mapping, and the next interface in the chain is typed against
the model, not the mapping. Wiring two steps together the obvious way
therefore hands a dict to code that expects a model, and the failure surfaces
somewhere deep inside the consumer as a missing attribute rather than at the
boundary where the mistake was made.

This module is that boundary. Consumers call :func:`as_candidates` on whatever
they were given and get models back, or a clear error naming the step that
produced the bad payload. The alternative, letting each consumer do its own
``isinstance`` dance, is how the two representations drift apart.

Nothing here invents data. A mapping that does not validate against the model
is an error, never a partially-filled object: a candidate silently missing its
family call or its catalytic mapping would be scored as though those were
genuinely absent rather than lost in transit.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from pydantic import ValidationError

from ..errors import EAgentError
from ..schemas.candidate import Candidate

__all__ = [
    "HandoffError",
    "as_candidate",
    "as_candidates",
    "serialise_candidates",
    "CANDIDATES_KEY",
]

#: The agreed key under which every interface publishes its candidate set.
CANDIDATES_KEY = "candidates"


class HandoffError(EAgentError):
    """A payload passed between steps was not a usable candidate set."""


def as_candidate(value: Any, *, source: str = "", index: int | None = None) -> Candidate:
    """Coerce one model-or-mapping into a :class:`Candidate`.

    Raises rather than returning a best-effort object, because a candidate that
    lost a field in transit still scores, and scores wrongly.
    """
    if isinstance(value, Candidate):
        return value
    where = f"{source or 'handoff'}"
    if index is not None:
        where += f"[{index}]"
    if isinstance(value, Mapping):
        try:
            return Candidate.model_validate(dict(value))
        except ValidationError as exc:
            raise HandoffError(
                f"{where}: mapping does not validate as a Candidate. A payload "
                f"that lost fields in serialisation must not be scored as "
                f"though those fields were genuinely absent.\n{exc}"
            ) from exc
    raise HandoffError(
        f"{where}: expected a Candidate or a mapping, got {type(value).__name__}"
    )


def as_candidates(value: Any, *, source: str = "") -> list[Candidate]:
    """Coerce a candidate payload into models.

    Accepts a sequence of models or mappings, or a ``ToolResult``-style data
    mapping carrying them under :data:`CANDIDATES_KEY`. ``None`` yields an
    empty list, because "this step produced no candidates" is a legitimate
    state that the consumer reports itself; a malformed payload is not.
    """
    if value is None:
        return []
    if isinstance(value, Candidate):
        return [value]
    if isinstance(value, Mapping):
        if CANDIDATES_KEY in value:
            return as_candidates(value[CANDIDATES_KEY], source=source)
        return [as_candidate(value, source=source)]
    if isinstance(value, (str, bytes)):
        raise HandoffError(
            f"{source or 'handoff'}: expected candidates, got a "
            f"{type(value).__name__}; a path or a JSON string is not a payload"
        )
    if isinstance(value, Iterable):
        return [as_candidate(v, source=source, index=i)
                for i, v in enumerate(value)]
    raise HandoffError(
        f"{source or 'handoff'}: cannot read candidates from "
        f"{type(value).__name__}"
    )


def serialise_candidates(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    """Render candidates for the manifest, in the form :func:`as_candidates` reads."""
    return [c.model_dump(mode="json") for c in candidates]
