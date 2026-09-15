#!/usr/bin/env python3
"""Summarize completed R25 Stage-2 pilot rows without changing their evidence."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from mmdd_dataset.wdc_runtime import iter_dataset_artifact


def run(args: argparse.Namespace) -> dict:
    rows = []
    with gzip.open(Path(args.results), "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    positives: dict[str, set[str]] = defaultdict(set)
    for qrel in iter_dataset_artifact(Path(args.dataset_root), "qrels"):
        if qrel.get("split", "train") == "dev":
            positives[str(qrel["query_table_id"])].add(str(qrel["target_table_id"]))
    summary = {}
    for generator in sorted({row["generator_id"] for row in rows}):
        for condition in ("Real", "NoE-fill"):
            subset = [row for row in rows if row["generator_id"] == generator and row["condition"] == condition]
            complete = [row for row in subset if row.get("status") == "complete"]
            failed = [row for row in subset if row.get("status") != "complete"]
            ranks: dict[str, dict[str, int]] = {}
            selected = 0
            nonempty = 0
            joinable = 0
            for row in complete:
                payload = row.get("result", {})
                candidates = payload.get("reranked_candidates", [])
                query_id = str(row["query_id"])
                positive = positives.get(query_id, set())
                hit_ranks = [int(candidate["rerank_rank"]) for candidate in candidates if str(candidate["target_id"]) in positive and candidate.get("rerank_rank") is not None]
                ranks[query_id] = {"positive_count": len(positive), "best_rank": min(hit_ranks) if hit_ranks else 0}
                for candidate in candidates:
                    selected += int(bool(candidate.get("selected_for_recovery")))
                    evidence = candidate.get("branches", {}).get("evidence", {})
                    nonempty += sum(bool(prediction.get("value", "").strip()) for prediction in evidence.get("rows", []))
                    verification = candidate.get("verification") or {}
                    joinable += int(bool(verification.get("joinable")))
            count = len(ranks)
            summary[f"{generator}/{condition}"] = {
                "rows": len(subset),
                "complete_rows": len(complete),
                "failed_rows": len(failed),
                "queries_with_results": count,
                "unique_queries": len({row["query_id"] for row in subset}),
                "input_candidate_counts": sorted({int(row.get("result", {}).get("input_candidate_count", 0)) for row in complete}),
                "selected_for_recovery_total": selected,
                "nonempty_generated_values": nonempty,
                "joinable_candidate_count": joinable,
                "stage2_recall_at": {
                    str(k): sum(1 for item in ranks.values() if item["best_rank"] and item["best_rank"] <= k) / count if count else None
                    for k in (10, 20, 50)
                },
                "grounded_value_metrics": "unknown: pilot qrels expose join metadata but not independent per-row target-value truth",
            }
    output = Path(args.output).resolve()
    output.write_text(json.dumps({"format_version": 1, "module": "S2", "results": str(Path(args.results).resolve()), "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
