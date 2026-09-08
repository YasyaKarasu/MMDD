#!/usr/bin/env python
"""Sample source-diverse train-fit queries for the R11 attribute audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mmdd_stage1.retrieval import checkpoint_fingerprint


def _priority(seed: int, *values: str) -> str:
    return hashlib.sha256(
        (str(seed) + "\0" + "\0".join(values)).encode("utf-8")
    ).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_by_query = {}
    with Path(args.query_tables).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_by_query[str(row["table_id"])] = str(row["source_table_id"])

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(args.target_lists).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("positive_evidence_by_target"):
                continue
            query_id = str(row["query_id"])
            by_source[source_by_query[query_id]].append(row)
    for source_id, rows in by_source.items():
        rows.sort(key=lambda row: _priority(args.seed, source_id, str(row["query_id"])))
    source_ids = sorted(
        by_source, key=lambda source_id: _priority(args.seed, source_id)
    )
    selected = []
    offset = 0
    while len(selected) < args.queries:
        added = 0
        for source_id in source_ids:
            rows = by_source[source_id]
            if offset < len(rows):
                selected.append(rows[offset])
                added += 1
                if len(selected) == args.queries:
                    break
        if not added:
            break
        offset += 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    payload = {
        "format_version": 1,
        "seed": args.seed,
        "policy": "source-diverse hash order; one query per source before reuse",
        "requested_queries": args.queries,
        "selected_queries": len(selected),
        "source_groups": len(
            {source_by_query[str(row["query_id"])] for row in selected}
        ),
        "input": str(Path(args.target_lists).resolve()),
        "input_sha256": checkpoint_fingerprint(Path(args.target_lists)),
        "output": str(output.resolve()),
        "output_sha256": checkpoint_fingerprint(output),
    }
    metadata = output.with_suffix(output.suffix + ".metadata.json")
    metadata.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-lists", required=True)
    parser.add_argument("--query-tables", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.queries <= 0:
        parser.error("--queries must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
