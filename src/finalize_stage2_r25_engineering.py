#!/usr/bin/env python3
"""Merge and register the two-shard R25 Stage-2 engineering audit."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def run(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    stage2 = root / "work/stage1_optimization_r25_final_20260914/stage2"
    records_path = stage2 / "engineering_records_32.jsonl"
    records = [json.loads(line) for line in records_path.open(encoding="utf-8") if line.strip()]
    audits = []
    for path in args.shards:
        audits.extend(json.loads(line) for line in Path(path).resolve().open(encoding="utf-8") if line.strip())
    if len(records) != 32 or len({row["record_id"] for row in records}) != 32:
        raise ValueError("engineering manifest must contain 32 unique records")
    by_id = {row["record_id"]: row for row in audits}
    expected = {row["record_id"] for row in records}
    if set(by_id) != expected:
        raise ValueError(f"audit IDs do not match manifest: missing={sorted(expected - set(by_id))[:3]}")
    merged = []
    updated = []
    totals = {
        "evidence_slots": 0,
        "unique_evidence_objects": 0,
        "json_parse_successes": 0,
        "json_parse_failures": 0,
        "empty_outputs": 0,
        "span_over_limit": 0,
        "roi_invalid": 0,
        "text_evidence_slots": 0,
        "image_evidence_slots": 0,
    }
    for record in records:
        audit = deepcopy(by_id[record["record_id"]])
        # Preserve repeated evidence IDs as explicit slots.  The second
        # occurrence is the same immutable artifact, so reuse its audit while
        # marking the duplication instead of silently dropping the slot.
        seen: dict[str, dict] = {}
        expanded = []
        for evidence_id in record["evidence_ids"]:
            match = next((item for item in audit.get("evidence_audits", []) if item["evidence_id"] == evidence_id), None)
            if match is None:
                raise ValueError(f"{record['record_id']}: missing evidence audit {evidence_id}")
            item = deepcopy(match)
            if evidence_id in seen:
                item["duplicate_evidence_reference"] = True
            seen[evidence_id] = item
            expanded.append(item)
        audit["evidence_audits"] = expanded
        merged.append(audit)
        parse_failures = sum(not item["generation"]["json_parse_ok"] for item in expanded)
        empty_outputs = sum(item["generation"]["empty_output"] for item in expanded)
        span_over = sum(not item["span_within_limit"] for item in expanded)
        roi_invalid = sum(item.get("roi_valid") is False for item in expanded)
        for item in expanded:
            totals["evidence_slots"] += 1
            totals["unique_evidence_objects"] += int(not item.get("duplicate_evidence_reference"))
            totals["json_parse_successes"] += int(item["generation"]["json_parse_ok"])
            totals["json_parse_failures"] += int(not item["generation"]["json_parse_ok"])
            totals["empty_outputs"] += int(item["generation"]["empty_output"])
            totals["span_over_limit"] += int(not item["span_within_limit"])
            totals["roi_invalid"] += int(item.get("roi_valid") is False)
            totals[f"{item['asset_type']}_evidence_slots"] += 1
        row = dict(record)
        row.update(
            {
                "completion_parse_status": "executed_parse_failed" if parse_failures else "executed_parse_ok",
                "engineering_audit_status": "executed",
                "evidence_slots_audited": len(expanded),
                "json_parse_failures": parse_failures,
                "empty_outputs": empty_outputs,
                "span_over_limit": span_over,
                "roi_invalid": roi_invalid,
            }
        )
        updated.append(row)
    merged.sort(key=lambda row: row["record_id"])
    updated.sort(key=lambda row: row["record_id"])
    output = stage2 / "S2_ENGINEERING_AUDIT.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8")
    records_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated), encoding="utf-8")
    receipt = {
        "format_version": 1,
        "module": "S2-engineering-audit",
        "status": "complete_with_parse_failures" if totals["json_parse_failures"] else "complete",
        "records": len(updated),
        "records_executed": sum(row["engineering_audit_status"] == "executed" for row in updated),
        "records_with_parse_failures": sum(row["completion_parse_status"] == "executed_parse_failed" for row in updated),
        "limits": {"max_span_tokens": 192, "max_generation_tokens": 64},
        "totals": totals,
        "audit_output": str(output.resolve()),
        "engineering_manifest": str(records_path.resolve()),
    }
    (stage2 / "S2_ENGINEERING_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--shards", nargs="+", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
