#!/usr/bin/env python3
"""Merge the two generator shards of the R25 Stage-2 pilot and emit its receipt."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> dict:
    shards = [read_rows(Path(path)) for path in args.shard]
    rows = [row for shard in shards for row in shard]
    keys = [(row["generator_id"], row["condition"], row["query_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("S2 pilot shards contain duplicate generator/condition/query rows")
    expected_queries = 64
    expected_generators = {"B13", "raw-Qwen"}
    expected_conditions = {"Real", "NoE-fill"}
    query_ids = {row["query_id"] for row in rows}
    if len(query_ids) != expected_queries:
        raise ValueError(f"Expected {expected_queries} unique queries, found {len(query_ids)}")
    if {row["generator_id"] for row in rows} != expected_generators:
        raise ValueError("S2 pilot must contain both B13 and raw-Qwen generators")
    if {row["condition"] for row in rows} != expected_conditions:
        raise ValueError("S2 pilot must contain Real and NoE-fill conditions")
    expected_rows = expected_queries * len(expected_generators) * len(expected_conditions)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    complete = sum(row.get("status") == "complete" for row in rows)
    effective_counts = [
        int(row.get("result", {}).get("input_candidate_count", 0))
        for row in rows
        if isinstance(row.get("result"), dict) and "input_candidate_count" in row["result"]
    ]
    receipt = {
        "format_version": 1,
        "module": "S2",
        # A failed model call is an observed S2 outcome, not a missing row.  The
        # execution is complete once every pre-registered cell is represented;
        # failed_rows remains explicit for the engineering/quality report.
        "status": "complete" if len(rows) == expected_rows else "partial",
        "generators": sorted(expected_generators),
        "conditions": sorted(expected_conditions),
        "queries": len(query_ids),
        "rows": len(rows),
        "complete_rows": complete,
        "failed_rows": len(rows) - complete,
        "candidate_budget": 50,
        "effective_candidate_counts": sorted(set(effective_counts)),
        "recovery_budget": 10,
        "outputs": [str(Path(path).resolve()) for path in args.shard],
        "merged_output": str(output),
    }
    receipt_path = output.parent / "S2_COMPLETION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (output.parent / "STATUS.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "module": "S2",
                "status": receipt["status"],
                "pilot_receipt": str(receipt_path.resolve()),
                "merged_output": str(output),
                "failed_rows": receipt["failed_rows"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # Promote the previously prepared manifest only after all generator/condition
    # rows have actually been executed; keep model_input and evaluation_metadata
    # untouched so the audit can distinguish inputs from outcomes.
    manifest = output.parent / "pilot_queries_64.jsonl"
    if len(rows) == expected_rows and manifest.is_file():
        updated = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item["status"] = "completed"
            item.pop("reason", None)
            item["stage2_checkpoint"] = str((output.parent / "r25_b13_column_scorer.pt").resolve())
            item["stage2_checkpoint_sha256"] = _sha256(Path(item["stage2_checkpoint"]))
            updated.append(item)
        manifest.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in updated) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
