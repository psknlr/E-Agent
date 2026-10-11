"""The agents that run in a browser, behind one JSON call surface.

WHAT THIS IS
============
A thin, bounded layer over code the project already has and already tests: the
pure-Python science functions and the task/gate machinery. It exists so that a
page can run them on the visitor's own CPU -- the interpreter is CPython
compiled to WebAssembly -- while a language model, reached through the visitor's
own API key, only decides *which* agent to call and with what. The model never
runs code. It names an agent and passes strings; this package validates them,
runs one fixed function, and hands back JSON.

WHY ONE STRING IN, ONE STRING OUT
=================================
:func:`call` takes an agent name and a JSON string and returns a JSON string.
Nothing else crosses the Python/JavaScript boundary. That keeps the boundary
trivial to test (the same call can be made natively and in the browser and the
two answers compared) and means no live Python object ever escapes into page
code.

WHAT A RESULT CARRIES
=====================
Every successful result carries ``cite``: the artifact key, a SHA-256 over the
canonical JSON of the result's value, and a row id derived from that hash. The
page registers it with the same numeric guard that checks every number a model
writes with a unit, so a model can only quote a computed number by citing the
result that computed it, and the guard checks the number, and the unit it is
written in, against that result. A field names its unit (``_percent``,
``_angstrom``, ``_degrees``); one with none is a count, a fraction or a score.
The guard does not check that a result answers the question asked.
A refusal carries no ``cite`` and no value: an agent that cannot produce a value
says so rather than producing a plausible one.

WHAT IS NOT HERE
================
This package's own code starts no process, opens no socket, reads nothing outside
the project tree and runs no model-supplied code; a test over its syntax trees
holds that. The one agent that runs the project's controller (``run_pipeline``)
calls project code the engine does not control: that code records the machine it
runs on and, natively, asks ``git`` for the current commit (a failure there is
caught). None of it appears in a result, the run is given no network, and in a
browser there is no process to start. The pipeline's external programs
(sequence search, structure prediction, docking, inverse folding) cannot run in
a browser and are not faked: :mod:`eagent.engine.pipeline` states which of the
ten interfaces can run here and which need a program, from the same table the
documentation uses.

Importing this package registers every agent. The three modules:

* :mod:`~eagent.engine.core` -- registry, argument validation, result envelope.
* :mod:`~eagent.engine.agents` -- the science agents.
* :mod:`~eagent.engine.pipeline` -- the pipeline capability map and gate report.
"""

from __future__ import annotations

from . import agents as _agents  # noqa: F401  (registers the science agents)
from . import pipeline as _pipeline  # noqa: F401  (registers the pipeline agents)
from .core import (
    AGENTS, ARTIFACT, ENGINE_SCHEMA, AgentRefusal, AgentSpec, call, catalog,
    static_catalog,
)

__all__ = [
    "AGENTS", "ARTIFACT", "ENGINE_SCHEMA", "AgentRefusal", "AgentSpec", "call",
    "catalog", "static_catalog",
]
