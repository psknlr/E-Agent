"""The browser export must preserve actual loader values and refusals."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from eagent.harness.toolloop import SYSTEM_PROMPT, reference_tools

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("browser_bundle", ROOT / "scripts" / "build_browser_bundle.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def bundle():
    return json.loads((ROOT / "web" / "reference-data.json").read_text())


def test_checked_in_bundle_is_current_deterministic_and_versioned():
    current = bundle()
    assert current["schema_version"] == 1
    assert current["system_prompt"] == SYSTEM_PROMPT
    assert builder.serialize(builder.build_bundle()) == (ROOT / "web" / "reference-data.json").read_text()
    payload = {key: value for key, value in current.items() if key != "bundle_digest"}
    assert hashlib.sha256(builder.canonical(payload).encode()).hexdigest() == current["bundle_digest"]
    assert (ROOT / "web" / "reference-data.json").stat().st_size < 1_000_000
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_browser_bundle.py"), "--check"], check=True)


def test_check_rejects_stale_bundle_without_modifying_it(tmp_path):
    output = tmp_path / "stale.json"
    output.write_text('{"stale":true}\n')
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_browser_bundle.py"),
                             "--check", "--out", str(output)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "stale" in result.stderr
    assert output.read_text() == '{"stale":true}\n'


def test_every_exported_query_matches_the_actual_python_reader():
    current = bundle()
    actual = {tool.name: tool for tool in reference_tools(ROOT / "configs/references/kred_calibration/v0.1")}
    for name, queries in current["results"].items():
        for encoded_args, result in queries.items():
            if result["ok"]:
                assert result["value"] == actual[name].run(**json.loads(encoded_args))
                assert result["refusal"] == ""
            else:
                try:
                    actual[name].run(**json.loads(encoded_args))
                except Exception as exc:
                    assert str(exc) == result["refusal"]
                else:
                    raise AssertionError("Exported refusal did not occur in actual loader")


def test_export_has_every_valid_record_structure_and_endpoint():
    current = bundle()["results"]
    kinetics = current["list_kinetic_records"]["{}"]["value"]["label_ids"]
    structures = current["list_structure_entries"]["{}"]["value"]["entries"]
    constructs = current["list_activity_constructs"]["{}"]["value"]
    assert len(current["kinetic_record"]) == len(kinetics)
    assert len(current["structure_entry"]) == len(structures)
    assert len(current["activity_endpoint"]) == len(constructs["enzyme_ids"]) * len(constructs["substrate_ids"])
    assert {json.loads(key)["tier"] for key in current["list_kinetic_records"] if key != "{}"} == {
        "", "core", "secondary", "sensitivity", "nd"}


def test_withheld_bounds_censoring_and_independence_are_preserved():
    current = bundle()
    bounded = current["results"]["kinetic_record"][builder.canonical({
        "label_id": "PaHBDH_H150N_AAE_activity_only"})]["value"]
    assert bounded["km"] is None
    assert "bound" in bounded["refusals"]["km"].lower()
    endpoints = [outcome["value"] for outcome in current["results"]["activity_endpoint"].values()]
    censored = [row for row in endpoints if row.get("status") == "below_detection"]
    insoluble = [row for row in endpoints if row.get("status") == "not_assayed"]
    assert censored and insoluble
    for row in censored + insoluble:
        assert row["total_product_mM"] is None
        assert row["refusal"]
        citation_row = current["citation_rows"][row["cite"]["artifact"]][row["cite"]["row"]]
        assert citation_row["fields"]["total_product_mM"] is None
        assert citation_row["fields"]["product_r_mM_as_printed"] is None
    assert current["results"]["activity_independence_groups"]["{}"]["ok"]
    assert current["results"]["source_verification"]["{}"]["value"]["documents"]
