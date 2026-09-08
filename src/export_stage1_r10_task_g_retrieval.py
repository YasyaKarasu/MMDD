#!/usr/bin/env python
"""Export a fixed R10 F4 ranking in the standard Stage-2 retrieval format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from mmdd_progress import progress

from mmdd_stage1.checkpoints import load_path_aggregator, load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import checkpoint_fingerprint, rank_detailed_paths
from mmdd_stage1.selection import load_stage1_selection
from run_stage1_r10_task_f import (
    calibrated_union_fusion,
    score_direct_confidences,
    scoring_object_ids,
)


FIXED_RETRIEVAL_BUDGET = {
    "direct_k": 100,
    "evidence_k_per_modality": 20,
    "targets_per_evidence": 20,
    "evidence_types": ["text", "image"],
}


def compact_stage2_ranking(
    ranking: Sequence[dict[str, Any]],
    *,
    result_k: int,
    path_result_k: int,
    evidence_path_k: int,
) -> list[dict[str, Any]]:
    """Retain F4 order plus the compact paths consumed by Stage 2."""

    results = []
    for result_index, detailed in enumerate(ranking[:result_k]):
        result: dict[str, Any] = {
            "target_id": str(detailed["target_id"]),
            "score": float(detailed["score"]),
            "stage2_table_score": float(detailed["score"]),
        }
        evidence_score = detailed.get("evidence_score")
        if evidence_score is not None:
            result["evidence_score"] = float(evidence_score)
        if result_index < path_result_k:
            paths = detailed.get("paths", [])
            compact = []
            if any(path.get("kind") == "direct" for path in paths):
                compact.append({"kind": "direct"})
            evidence_by_id = {
                str(path["evidence_id"]): path
                for path in paths
                if path.get("kind") == "evidence"
                and path.get("evidence_id") is not None
            }
            selected = detailed.get("selected_evidence_ids")
            if selected is None:
                evidence_paths = [
                    path for path in paths if path.get("kind") == "evidence"
                ][:evidence_path_k]
            else:
                evidence_paths = [
                    evidence_by_id[str(evidence_id)]
                    for evidence_id in selected[:evidence_path_k]
                    if str(evidence_id) in evidence_by_id
                ]
            compact.extend(
                {
                    "kind": "evidence",
                    "evidence_id": str(path["evidence_id"]),
                    "path_score": float(path["path_score"]),
                }
                for path in evidence_paths
            )
            result["paths"] = compact
        results.append(result)
    return results


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(
        args.result_k,
        args.path_result_k,
        args.evidence_path_k,
        args.pair_batch_size,
        args.feature_cache_size,
    ) <= 0:
        raise ValueError("Retrieval, path, batch, and cache limits must be positive")
    if args.path_result_k > args.result_k:
        raise ValueError("--path-result-k cannot exceed --result-k")
    if not 0.0 <= args.evidence_weight <= 1.0:
        raise ValueError("--evidence-weight must be within [0, 1]")

    path_pool = Path(args.path_pool).resolve()
    pool_metadata_path = path_pool.with_suffix(path_pool.suffix + ".metadata.json")
    pool_metadata = json.loads(pool_metadata_path.read_text(encoding="utf-8"))
    if checkpoint_fingerprint(path_pool) != pool_metadata.get("output_sha256"):
        raise ValueError("Path-pool fingerprint differs from its metadata")
    if pool_metadata.get("retrieval_budget") != FIXED_RETRIEVAL_BUDGET:
        raise ValueError("Task G requires the fixed R10 100/20/20 path pool")
    if pool_metadata.get("student_score_space") != "confidence":
        raise ValueError("Task G F4 export requires confidence-space Student scores")

    selection_path = Path(pool_metadata["selection"])
    selection = load_stage1_selection(selection_path)
    checkpoint = Path(selection["best_checkpoint"])
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    if checkpoint_sha256 != pool_metadata.get("student_checkpoint_sha256"):
        raise ValueError("Path pool and selected Student checkpoint differ")
    aggregator = load_path_aggregator(checkpoint)
    if aggregator.config() != pool_metadata.get("path_aggregation"):
        raise ValueError("Path pool and selected aggregation configuration differ")

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    student = load_student(checkpoint, device).eval()
    features = Path(args.features).resolve()
    store = FeatureStore.from_path(features, cache_size=args.feature_cache_size)
    preloaded = 0
    if not args.no_preload:
        preloaded = store.preload_embeddings(scoring_object_ids(path_pool))

    stage2_config = {
        **aggregator.config(),
        "student_score_space": "confidence",
        "target_fusion": "f4_calibrated_union",
        "f4_evidence_weight": args.evidence_weight,
        "path_result_k": args.path_result_k,
        "evidence_path_k": args.evidence_path_k,
        "source_path_pool_sha256": pool_metadata["output_sha256"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    records = 0
    union_targets = 0
    with path_pool.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in progress(source, desc="Export Task-G retrieval", unit="query"):
            record = json.loads(line)
            ranked = rank_detailed_paths(
                record["paths_by_target"],
                aggregator=aggregator,
                rrf_k=60,
                fusion_mode="weighted_rrf",
                direct_weight=1.0,
                evidence_weight=0.05,
                gated_evidence_min_paths=2,
                gated_evidence_quantile=0.75,
            )
            direct = ranked["direct"]
            evidence = ranked["evidence"]
            target_ids = list(
                dict.fromkeys(
                    str(row["target_id"]) for row in [*direct, *evidence]
                )
            )
            direct_confidences = score_direct_confidences(
                student,
                store,
                str(record["query_id"]),
                target_ids,
                device=device,
                batch_size=args.pair_batch_size,
            )
            fused = calibrated_union_fusion(
                direct,
                evidence,
                direct_confidences,
                evidence_weight=args.evidence_weight,
            )
            destination.write(
                json.dumps(
                    {
                        "query_id": str(record["query_id"]),
                        "student_checkpoint_sha256": checkpoint_sha256,
                        "path_aggregation": stage2_config,
                        "results": compact_stage2_ranking(
                            fused,
                            result_k=args.result_k,
                            path_result_k=args.path_result_k,
                            evidence_path_k=args.evidence_path_k,
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            records += 1
            union_targets += len(target_ids)
    temporary.replace(output)

    expected_records = int(pool_metadata["queries"])
    if records != expected_records:
        raise ValueError(f"Read {records} path-pool records, expected {expected_records}")
    metadata = {
        "format_version": 1,
        "purpose": "r10_task_g_stage2_retrieval",
        "split": pool_metadata["split"],
        "records": records,
        "selection": str(selection_path.resolve()),
        "student_checkpoint": str(checkpoint.resolve()),
        "student_checkpoint_sha256": checkpoint_sha256,
        "path_pool": str(path_pool),
        "path_pool_sha256": pool_metadata["output_sha256"],
        "features": str(features),
        "retrieval_budget": pool_metadata["retrieval_budget"],
        "stage2_config": stage2_config,
        "preloaded_objects": preloaded,
        "direct_pair_rescores": union_targets,
        "mean_union_targets_per_query": union_targets / records,
        "output": str(output.resolve()),
        "output_sha256": checkpoint_fingerprint(output),
    }
    _write_json(output.with_suffix(output.suffix + ".metadata.json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-pool", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-weight", type=float, default=0.5)
    parser.add_argument("--result-k", type=int, default=10)
    parser.add_argument("--path-result-k", type=int, default=10)
    parser.add_argument("--evidence-path-k", type=int, default=4)
    parser.add_argument("--pair-batch-size", type=int, default=512)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-preload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
