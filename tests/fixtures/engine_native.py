"""Run engine calls natively and print the replies, for the browser parity test.

Reads a JSON list of [agent, arguments_json] pairs on stdin and writes the list
of reply strings on stdout. The Node test makes the same calls inside the
WebAssembly runtime and compares the two.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eagent.engine import call  # noqa: E402

json.dump([call(agent, arguments) for agent, arguments in json.load(sys.stdin)], sys.stdout)
