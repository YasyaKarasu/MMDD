#!/usr/bin/env python
"""Audit whether two r7 epoch-0 runs use identical ANN retrieval pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    checkpoint_fingerprint,
    retrieve_zero_one_hop_detailed_many,
)


def _lake_paths(root: Path, lake: str) -> tuple[Path, Path]:
    r4 = root / "work/stage1_optimization_r4_20260829"
    if lake == "entitables":
        data = root / "work/stage1_stage2_entitables20k_v4_20260827/stage1_data"
    else:
        data = (
            root
            / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data"
        )
    return (
        data / "target_lists.jsonl",
        r4 / f"taskJ_per_lake_baselines/corpora/{lake}_corpus.jsonl",
    )


def _load_indices(
    run_dir: Path,
    store: FeatureStore,
    *,
    device: torch.device,
    corpus_sha256: str,
) -> StudentANNIndices:
    checkpoint = run_dir / "student_path.epochs/epoch_000.pt"
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    return StudentANNIndices(
        load_student(checkpoint, device).eval(),
        store,
        run_dir / "student_path.dev_indices/epoch_000",
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["target_id"]) for row in rows]


def _pool_signature(result: dict[str, list[dict[str, Any]]]) -> str:
    digest = hashlib.sha256()
    by_target = {}
    for row in [*result["direct"], *result["evidence"]]:
        by_target[str(row["target_id"])] = row["paths"]
    for target_id, paths in sorted(by_target.items()):
        digest.update(target_id.encode("utf-8"))
        for path in sorted(
            paths,
            key=lambda value: (
                str(value["kind"]),
                str(value.get("evidence_id", "")),
            ),
        ):
            digest.update(b"\0")
            digest.update(str(path["kind"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(path.get("evidence_id", "")).encode("utf-8"))
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    target_data, corpus = _lake_paths(root, args.lake)
    features = (
        root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    )
    store = FeatureStore.from_path(features, cache_size=args.feature_cache_size)
    device = torch.device(args.device)
    corpus_sha256 = checkpoint_fingerprint(corpus)
    left = _load_indices(
        args.left, store, device=device, corpus_sha256=corpus_sha256
    )
    right = _load_indices(
        args.right, store, device=device, corpus_sha256=corpus_sha256
    )
    examples = load_target_examples(
        target_data, split="dev", dataset_name=target_data.stem
    )
    exact = {"direct": 0, "evidence": 0, "fused": 0, "path_pool": 0}
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    for start in range(0, len(examples), args.query_batch_size):
        query_ids = [
            example.query_id
            for example in examples[start : start + args.query_batch_size]
        ]
        common = {
            "k": 10,
            "gamma": 10,
            "gamma_evidence": 2,
            "evidence_types": ("text", "image"),
            "evidence_aggregation": "logsumexp",
            "evidence_top_k": 4,
            "fusion_mode": "weighted_rrf",
            "direct_weight": 1.0,
            "evidence_weight": 0.05,
            "query_batch_size": args.query_batch_size,
        }
        left_results = retrieve_zero_one_hop_detailed_many(
            query_ids, left, **common
        )
        right_results = retrieve_zero_one_hop_detailed_many(
            query_ids, right, **common
        )
        for left_result, right_result in zip(left_results, right_results):
            for channel in ("direct", "evidence", "fused"):
                exact[channel] += _ids(left_result[channel]) == _ids(
                    right_result[channel]
                )
            left_signature = _pool_signature(left_result)
            right_signature = _pool_signature(right_result)
            exact["path_pool"] += left_signature == right_signature
            left_digest.update(left_signature.encode("ascii"))
            right_digest.update(right_signature.encode("ascii"))
    payload = {
        "format_version": 1,
        "lake": args.lake,
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "queries": len(examples),
        "exact_queries": exact,
        "left_pool_sha256": left_digest.hexdigest(),
        "right_pool_sha256": right_digest.hexdigest(),
        "all_candidate_pools_identical": exact["path_pool"] == len(examples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", required=True, choices=["entitables", "wdc"])
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--feature-cache-size", type=int, default=16_000)
    values = parser.parse_args()
    if min(values.query_batch_size, values.feature_cache_size) <= 0:
        parser.error("batch sizes must be positive")
    return values


if __name__ == "__main__":
    run(parse_args())
