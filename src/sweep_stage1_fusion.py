#!/usr/bin/env python
"""Sweep Stage-1 raw-channel fusion without repeating ANN retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mmdd_progress import progress
from mmdd_stage1.data import TargetExample, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.protocol import validate_protocol_split
from mmdd_stage1.retrieval import (
    checkpoint_fingerprint,
    fuse_ranked_channels,
    load_corpus_ids,
    load_or_build_raw_embedding_indices,
    retrieve_zero_one_hop_detailed,
)
from mmdd_stage1.selection import write_json

DEFAULT_RECALL_KS = (10, 20, 30, 40, 50)


def _parse_recall_ks(value: str) -> tuple[int, ...]:
    try:
        values = tuple(
            int(part.strip()) for part in value.split(",") if part.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--recall-ks must be comma-separated integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "--recall-ks must contain positive integers"
        )
    return tuple(sorted(dict.fromkeys(values)))


def _metrics(
    indices: Sequence[int],
    rankings: Sequence[Sequence[str]],
    positives: Sequence[set[str]],
    recall_ks: Sequence[int],
    evidence_hits: Sequence[bool] | None = None,
) -> dict[str, float | int]:
    values: dict[str, float | int] = {"queries": len(indices)}
    for k in recall_ks:
        values[f"recall@{k}"] = statistics.fmean(
            len(set(rankings[index][:k]) & positives[index]) / len(positives[index])
            for index in indices
        )
    max_k = max(recall_ks)
    values[f"mrr@{max_k}"] = statistics.fmean(
        next(
            (
                1.0 / rank
                for rank, target_id in enumerate(rankings[index][:max_k], 1)
                if target_id in positives[index]
            ),
            0.0,
        )
        for index in indices
    )
    if evidence_hits is not None:
        count = sum(evidence_hits[index] for index in indices)
        values["positive_evidence_path_queries@10"] = count
        values["positive_evidence_path_coverage@10"] = count / len(indices)
    return values


def _with_datasets(
    examples: Sequence[TargetExample],
    rankings: Sequence[Sequence[str]],
    positives: Sequence[set[str]],
    recall_ks: Sequence[int],
    evidence_hits: Sequence[bool] | None = None,
) -> dict[str, Any]:
    overall = _metrics(
        list(range(len(examples))), rankings, positives, recall_ks, evidence_hits
    )
    overall["by_dataset"] = {
        dataset: _metrics(
            [
                index
                for index, example in enumerate(examples)
                if example.dataset == dataset
            ],
            rankings,
            positives,
            recall_ks,
            evidence_hits,
        )
        for dataset in sorted({example.dataset for example in examples})
    }
    return overall


def _positive_evidence_hit(
    example: TargetExample, fused: Sequence[dict[str, Any]]
) -> bool:
    positives = set(example.positive_target_ids)
    positive_evidence = {
        candidate.target_id: set(candidate.evidence_ids)
        for candidate in example.candidates
        if candidate.target_id in positives and candidate.evidence_ids
    }
    return any(
        str(item["target_id"]) in positive_evidence
        and any(
            path["kind"] == "evidence"
            and str(path["evidence_id"])
            in positive_evidence[str(item["target_id"])]
            for path in item["paths"]
        )
        for item in fused[:10]
    )


def _configs(evidence_weights: Sequence[float]) -> list[dict[str, Any]]:
    configs = [
        {"name": "rrf", "fusion_mode": "rrf", "evidence_weight": 1.0}
    ]
    configs.extend(
        {
            "name": f"weighted_rrf_e{weight:g}",
            "fusion_mode": "weighted_rrf",
            "evidence_weight": weight,
        }
        for weight in evidence_weights
    )
    configs.append(
        {"name": "gated", "fusion_mode": "gated", "evidence_weight": 1.0}
    )
    return configs


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_protocol_split("dev_gate", args.dev_split)
    if any(weight < 0 for weight in args.evidence_weights):
        raise ValueError("--evidence-weights must be non-negative")
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    examples = [
        example
        for path_value in args.dev_data
        for example in load_target_examples(
            Path(path_value), split=args.dev_split, dataset_name=Path(path_value).stem
        )
    ]
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    raw_indices = load_or_build_raw_embedding_indices(
        store,
        load_corpus_ids(corpus_path, store),
        Path(args.raw_index_root),
        corpus_sha256=corpus_sha256,
        batch_size=args.index_batch_size,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    configs = _configs(args.evidence_weights)
    rankings = {config["name"]: [] for config in configs}
    evidence_hits = {config["name"]: [] for config in configs}
    direct_rankings = []
    evidence_rankings = []
    positives = []
    for example in progress(examples, desc="Fusion sweep", unit="query"):
        detailed = retrieve_zero_one_hop_detailed(
            example.query_id,
            raw_indices,
            direct_k=args.direct_k,
            evidence_k=args.evidence_k,
            targets_per_evidence=args.targets_per_evidence,
            evidence_types=tuple(args.evidence_types),
            evidence_aggregation=args.evidence_aggregation,
            evidence_top_k=args.evidence_top_k,
            rrf_k=args.rrf_k,
        )
        direct_rankings.append(
            [str(item["target_id"]) for item in detailed["direct"]]
        )
        evidence_rankings.append(
            [str(item["target_id"]) for item in detailed["evidence"]]
        )
        positives.append(set(example.positive_target_ids))
        for config in configs:
            fused = fuse_ranked_channels(
                detailed["direct"],
                detailed["evidence"],
                rrf_k=args.rrf_k,
                fusion_mode=config["fusion_mode"],
                direct_weight=1.0,
                evidence_weight=config["evidence_weight"],
                gated_evidence_min_paths=args.gated_evidence_min_paths,
                gated_evidence_quantile=args.gated_evidence_quantile,
            )
            rankings[config["name"]].append(
                [str(item["target_id"]) for item in fused]
            )
            evidence_hits[config["name"]].append(
                _positive_evidence_hit(example, fused)
            )

    direct_metrics = _with_datasets(
        examples, direct_rankings, positives, args.recall_ks
    )
    evidence_metrics = _with_datasets(
        examples, evidence_rankings, positives, args.recall_ks
    )
    results = []
    for config in configs:
        name = config["name"]
        results.append(
            {
                **config,
                "metrics": _with_datasets(
                    examples,
                    rankings[name],
                    positives,
                    args.recall_ks,
                    evidence_hits[name],
                ),
            }
        )
    eligible = [
        result
        for result in results
        if result["metrics"]["recall@10"]
        >= direct_metrics["recall@10"] - args.max_direct_drop
    ]
    if eligible:
        recommended = max(
            eligible,
            key=lambda result: (
                result["metrics"]["positive_evidence_path_coverage@10"],
                result["metrics"]["recall@10"],
            ),
        )
    else:
        recommended = max(
            results,
            key=lambda result: (
                result["metrics"]["recall@10"],
                result["metrics"]["positive_evidence_path_coverage@10"],
            ),
        )
    payload = {
        "format_version": 1,
        "corpus_sha256": corpus_sha256,
        "direct": direct_metrics,
        "evidence": evidence_metrics,
        "configs": results,
        "max_direct_drop": args.max_direct_drop,
        "eligible_configs": [result["name"] for result in eligible],
        "recommended": recommended,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", payload)
    max_k = max(args.recall_ks)
    lines = [
        "# Task 4: raw fusion sweep",
        "",
        f"| Config | Fused R@10 | R@{max_k} | MRR@{max_k} | Positive evidence coverage@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['name']} | {metrics['recall@10']:.2%} | "
            f"{metrics[f'recall@{max_k}']:.2%} | "
            f"{metrics[f'mrr@{max_k}']:.4f} | "
            f"{metrics['positive_evidence_path_coverage@10']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Direct raw R@10: {direct_metrics['recall@10']:.2%}.",
            f"Recommended config: `{recommended['name']}`.",
            "",
        ]
    )
    report = "\n".join(lines)
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    with (output_dir.parent / "RESULTS.md").open("a", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--dev-data", required=True, nargs="+")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--raw-index-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-split", default="dev", choices=["dev"])
    parser.add_argument(
        "--recall-ks", type=_parse_recall_ks, default=DEFAULT_RECALL_KS
    )
    parser.add_argument("--evidence-weights", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    parser.add_argument("--max-direct-drop", type=float, default=0.005)
    parser.add_argument("--direct-k", type=int, default=100)
    parser.add_argument("--evidence-k", type=int, default=50)
    parser.add_argument("--targets-per-evidence", type=int, default=50)
    parser.add_argument("--evidence-types", nargs="+", choices=["text", "image"], default=["text", "image"])
    parser.add_argument("--evidence-aggregation", default="logsumexp", choices=["logsumexp", "topk_mean", "topk_sum"])
    parser.add_argument("--evidence-top-k", type=int, default=4)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--gated-evidence-min-paths", type=int, default=2)
    parser.add_argument("--gated-evidence-quantile", type=float, default=0.75)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
