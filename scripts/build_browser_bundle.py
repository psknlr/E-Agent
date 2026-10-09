"""Export deterministic browser evidence by running the actual reference readers.

Run this at build time; the published chat needs no Python process. --check
refuses a stale checked-in bundle without changing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eagent.errors import EAgentError
from eagent.harness.toolloop import SYSTEM_PROMPT, reference_tools


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def numeric_fields(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
        if value is not None and not math.isfinite(value):
            raise ValueError("A reference reader returned a non-finite numeric value")
        yield path, value
    elif isinstance(value, dict):
        for name, item in value.items():
            if name != "cite":
                yield from numeric_fields(item, f"{path}.{name}" if path else name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from numeric_fields(item, f"{path}[{index}]")


def field_names(value: Any, path: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for name, item in value.items():
            if name != "cite":
                yield from field_names(item, f"{path}.{name}" if path else name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from field_names(item, f"{path}[{index}]")
    else:
        yield path


def build_bundle(root: Path = ROOT) -> dict[str, Any]:
    tools = reference_tools(root / "configs" / "references" / "kred_calibration" / "v0.1")
    by_name = {tool.name: tool for tool in tools}
    results: dict[str, dict[str, Any]] = {tool.name: {} for tool in tools}
    rows: dict[str, dict[str, Any]] = {}
    conflicts: set[tuple[str, str, str]] = set()

    def export(name: str, arguments: dict[str, str]) -> Any:
        try:
            value = by_name[name].run(**arguments)
            outcome = {"ok": True, "value": value, "refusal": ""}
        except EAgentError as exc:
            value = None
            outcome = {"ok": False, "value": None, "refusal": str(exc)}
        results[name][canonical(arguments)] = outcome
        if isinstance(value, dict) and isinstance(value.get("cite"), dict):
            citation = value["cite"]
            artifact, row_id = citation["artifact"], citation["row"]
            row = rows.setdefault(artifact, {}).setdefault(row_id, {
                "sha256": citation["sha256"], "fields": {}, "field_names": []})
            row["field_names"] = sorted(set(row["field_names"]) | set(field_names(value)))
            if row["sha256"] != citation["sha256"]:
                raise ValueError("One exported citation row has conflicting data versions")
            for field, number in numeric_fields(value):
                # Printed zeros on censored rows remain visible as source
                # metadata but cannot license a claimed point measurement.
                if value.get("status") in {"below_detection", "not_assayed"} and field.endswith("_as_printed"):
                    number = None
                marker = (artifact, row_id, field)
                if marker in conflicts:
                    continue
                if field in row["fields"] and row["fields"][field] != number:
                    del row["fields"][field]
                    conflicts.add(marker)
                else:
                    row["fields"][field] = number
        return value

    for tool in tools:
        if not tool.parameters:
            export(tool.name, {})
    kinetics = export("list_kinetic_records", {})
    export("list_kinetic_records", {"tier": ""})
    for tier in ("core", "secondary", "sensitivity", "nd"):
        export("list_kinetic_records", {"tier": tier})
    for label_id in kinetics["label_ids"]:
        export("kinetic_record", {"label_id": label_id})
    for entry in results["list_structure_entries"]["{}"]["value"]["entries"]:
        export("structure_entry", {"pdb_id": entry["pdb_id"]})
    constructs = results["list_activity_constructs"]["{}"]["value"]
    for enzyme_id in constructs["enzyme_ids"]:
        for substrate_id in constructs["substrate_ids"]:
            export("activity_endpoint", {"enzyme_id": enzyme_id, "substrate_id": substrate_id})
    digest = results["reference_summary"]["{}"]["value"]["cite"]["sha256"]
    bundle = {"format": "eagent.browser-reference-tools", "schema_version": 1,
              "system_prompt": SYSTEM_PROMPT, "reference_digest": digest,
              "source": "eagent.harness.toolloop.reference_tools",
              "schemas": [tool.schema() for tool in tools],
              "results": results, "citation_rows": rows}
    bundle["bundle_digest"] = hashlib.sha256(canonical(bundle).encode("utf-8")).hexdigest()
    return bundle


def serialize(bundle: dict[str, Any]) -> str:
    return canonical(bundle) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the checked-in export without writing")
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "reference-data.json")
    args = parser.parse_args(argv)
    expected = serialize(build_bundle())
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != expected:
            print("Browser reference bundle is stale; run scripts/build_browser_bundle.py", file=sys.stderr)
            return 1
        print("Browser reference bundle matches the actual readers.")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(expected, encoding="utf-8")
        print(f"Exported browser reference bundle ({len(expected.encode('utf-8'))} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
