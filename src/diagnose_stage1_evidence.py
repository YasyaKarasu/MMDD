#!/usr/bin/env python
"""Audit Stage-1 positive evidence provenance and raw Q->E path reachability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.evidence_diagnostics import (
    analyze_evidence_annotations,
    rank_raw_evidence_paths,
)
from mmdd_stage1.features import FeatureStore


def _named_paths(values: list[str], option: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} values must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"{option} has an empty or duplicate name: {name!r}")
        result[name] = Path(raw_path).resolve()
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    dataset_roots = _named_paths(args.dataset, "--dataset")
    target_lists = _named_paths(args.target_lists, "--target-lists")
    if dataset_roots.keys() != target_lists.keys():
        raise ValueError("--dataset and --target-lists must use the same names")

    annotation_summaries = []
    cases = []
    for name in dataset_roots:
        summary, dataset_cases = analyze_evidence_annotations(
            dataset_name=name,
            dataset_root=dataset_roots[name],
            target_lists_path=target_lists[name],
            split=args.split,
        )
        annotation_summaries.append(summary)
        cases.extend(dataset_cases)

    retrieval_summary = None
    ranked_details = []
    if not args.skip_ranking:
        if args.features is None or args.raw_index is None:
            raise ValueError("--features and --raw-index are required unless --skip-ranking is set")
        store = FeatureStore.from_path(
            Path(args.features).resolve(),
            cache_size=args.feature_cache_size,
        )
        retrieval_summary, ranked_details = rank_raw_evidence_paths(
            cases=cases,
            feature_store=store,
            raw_index_dir=Path(args.raw_index).resolve(),
            evidence_rank_limit=args.evidence_rank_limit,
            target_rank_limit=args.target_rank_limit,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )
    output = {
        "format_version": 1,
        "split": args.split,
        "annotation_diagnostics": annotation_summaries,
    }
    if retrieval_summary is not None:
        output["raw_retrieval_diagnostics"] = retrieval_summary
    ranked_by_key = {
        (row["dataset"], row["query_id"], row["target_id"]): row
        for row in ranked_details
    }
    details = [
        ranked_by_key.get(
            (case["dataset"], case["query_id"], case["target_id"]), case
        )
        for case in cases
    ]
    output_path = Path(args.output)
    details_path = Path(args.details_output)
    _write_json(output_path, output)
    _write_jsonl(details_path, details)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "details_output": str(details_path),
                "queries": len(cases),
                "ranked_queries": len(ranked_details),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Dataset root; repeat for each dataset in the mixed run.",
    )
    parser.add_argument(
        "--target-lists",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Matching Stage-1 target_lists.jsonl; repeat for each dataset.",
    )
    parser.add_argument("--features")
    parser.add_argument("--raw-index")
    parser.add_argument("--skip-ranking", action="store_true")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--evidence-rank-limit", type=int, default=5000)
    parser.add_argument("--target-rank-limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", required=True)
    args = parser.parse_args()
    if min(
        args.evidence_rank_limit,
        args.target_rank_limit,
        args.batch_size,
    ) <= 0:
        parser.error("rank limits and batch size must be positive")
    if args.num_threads < 0:
        parser.error("--num-threads must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
