#!/usr/bin/env python
"""Evaluate the selected Stage-1 r6 configuration for one lake."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    checkpoint_fingerprint,
    rank_detailed_paths,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.selection import load_stage1_selection
from mmdd_stage1.significance import paired_bootstrap_delta
from mmdd_stage1.sweep_metrics import (
    append_values as _append_values,
    finalize_records as _finalize_records,
    path_pool as _path_pool,
    query_values as _query_values,
)
from run_stage1_r6_sweeps import (
    RECALL_KS,
    _lake_inputs,
    _write_json,
)


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = json.loads(
        (args.output_root / "selection.json").read_text(encoding="utf-8")
    )
    selected = configuration["lakes"][args.lake]
    inputs = _lake_inputs(root, args.lake)
    selection_path = Path(selected["student_selection"])
    selection = load_stage1_selection(selection_path)
    checkpoint = Path(selection["best_checkpoint"])
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    if checkpoint_sha256 != selection["best_checkpoint_sha256"]:
        raise ValueError("Selected Student checkpoint fingerprint mismatch")
    corpus_sha256 = checkpoint_fingerprint(inputs.corpus)
    if corpus_sha256 != selection["corpus_sha256"]:
        raise ValueError("Selected Student and lake corpus fingerprints differ")

    device = torch.device(args.device)
    features = (
        root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    )
    store = FeatureStore.from_path(features, cache_size=args.feature_cache_size)
    examples = load_target_examples(
        inputs.dev_data, split="dev", dataset_name=inputs.dev_data.stem
    )
    student = load_student(checkpoint, device).eval()
    indices = StudentANNIndices(
        student,
        store,
        Path(selection["best_index"]),
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )
    aggregation = selected["aggregation"]
    fusion = selected["fusion"]
    records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pool_hashes = {k: hashlib.sha256() for k in RECALL_KS}
    for k in RECALL_KS:
        for start in range(0, len(examples), args.query_batch_size):
            batch = examples[start : start + args.query_batch_size]
            detailed = retrieve_zero_one_hop_detailed_many(
                [example.query_id for example in batch],
                indices,
                k=k,
                gamma=10,
                gamma_evidence=2,
                evidence_types=("text", "image"),
                evidence_aggregation="logsumexp",
                evidence_top_k=4,
                fusion_mode="weighted_rrf",
                evidence_weight=0.05,
                query_batch_size=args.query_batch_size,
            )
            for example, baseline in zip(batch, detailed):
                paths = _path_pool(baseline)
                pool_hashes[k].update(example.query_id.encode("utf-8"))
                pool_hashes[k].update(b"\0")
                for target_id in sorted(paths):
                    pool_hashes[k].update(target_id.encode("utf-8"))
                    pool_hashes[k].update(b"\0")
                result = rank_detailed_paths(
                    paths,
                    aggregator=PathAggregator(
                        aggregation["aggregation"],
                        4,
                        temperature=float(aggregation["temperature"]),
                        power=float(aggregation["power"]),
                    ),
                    path_edge_normalization=aggregation["path_edge_normalization"],
                    rrf_k=60,
                    fusion_mode=fusion["fusion_mode"],
                    direct_weight=1.0,
                    evidence_weight=float(fusion["evidence_weight"]),
                    fusion_score_normalization=fusion["score_normalization"],
                    fusion_score_temperature=float(fusion["score_temperature"]),
                    gated_evidence_min_paths=2,
                    gated_evidence_quantile=0.75,
                )
                _append_values(records, "final", k, _query_values(result, example, k))

    finalized = _finalize_records(
        records,
        "final",
        bootstrap_iterations=10_000,
        bootstrap_seed=13,
    )["final"]["metrics"]
    baseline_payload = json.loads(
        (
            args.output_root
            / "taskB_fusion_normalization"
            / f"{args.lake}.json"
        ).read_text(encoding="utf-8")
    )
    baseline = baseline_payload["systems"]["student"]["configs"][
        baseline_payload["baseline"]
    ]["metrics"]["per_query"]
    final_per_query = finalized["per_query"]
    deltas = {
        "recall@10": paired_bootstrap_delta(
            final_per_query["recall@10"],
            baseline["recall@10"],
            iterations=10_000,
            seed=13,
        ),
        "coverage@10": paired_bootstrap_delta(
            final_per_query["coverage@10"],
            baseline["coverage@10"],
            iterations=10_000,
            seed=13,
        ),
        "mrr@50": paired_bootstrap_delta(
            final_per_query["mrr@50"],
            baseline["mrr@50"],
            iterations=10_000,
            seed=13,
        ),
    }
    payload = {
        "format_version": 1,
        "lake": args.lake,
        "queries": len(examples),
        "metrics": finalized,
        "delta_vs_r5": deltas,
        "configuration": selected,
        "candidate_pool_sha256": {
            str(k): digest.hexdigest() for k, digest in pool_hashes.items()
        },
        "candidate_pool_policy": "fixed per query/k across final aggregation and fusion",
        "student_checkpoint": str(checkpoint.resolve()),
        "student_checkpoint_sha256": checkpoint_sha256,
        "teacher_checkpoint": selected["teacher_checkpoint"],
        "evidence_types": ["text", "image"],
        "dual_teacher_provenance": {
            "lake": args.lake,
            "teacher_checkpoint": selected["teacher_checkpoint"],
        },
    }
    _write_json(args.output_root / f"task_final/{args.lake}.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", required=True, choices=["entitables", "wdc"])
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r6_20260830"),
    )
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--feature-cache-size", type=int, default=16_000)
    values = parser.parse_args()
    if values.query_batch_size <= 0:
        parser.error("--query-batch-size must be positive")
    return values


if __name__ == "__main__":
    run(parse_args())
