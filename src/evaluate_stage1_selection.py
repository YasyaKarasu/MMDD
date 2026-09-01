#!/usr/bin/env python
"""Evaluate one dev-selected Stage-1 Student checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_path_aggregator, load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.evaluation import evaluate_student_retrieval
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PATH_AGGREGATIONS, PathAggregator
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    checkpoint_fingerprint,
    load_corpus_ids,
    load_or_build_raw_embedding_indices,
)
from mmdd_stage1.selection import load_stage1_selection, write_json


def _parse_recall_ks(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("--recall-ks must contain positive integers")
    return tuple(sorted(dict.fromkeys(values)))


def _examples(paths: list[str]) -> list[Any]:
    return [
        example
        for value in paths
        for example in load_target_examples(
            Path(value), split="dev", dataset_name=Path(value).stem
        )
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection = load_stage1_selection(Path(args.selection))
    if selection.get("completed_stage") != "student-path":
        raise ValueError("Selection must describe a student-path checkpoint")
    checkpoint_path = Path(selection["best_checkpoint"])
    index_dir = Path(selection["best_index"])
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    if corpus_sha256 != selection.get("corpus_sha256"):
        raise ValueError("Selection and corpus fingerprints differ")
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
    if checkpoint_sha256 != selection.get("best_checkpoint_sha256"):
        raise ValueError("Selection and checkpoint fingerprints differ")

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    student = load_student(checkpoint_path, device)
    student.eval()
    indices = StudentANNIndices(
        student,
        store,
        index_dir,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )
    examples = _examples(args.dev_data)
    saved_aggregator = load_path_aggregator(checkpoint_path)
    aggregator = PathAggregator(
        args.evidence_aggregation or saved_aggregator.evidence_aggregation,
        args.evidence_top_k
        if args.evidence_top_k is not None
        else saved_aggregator.top_k,
        temperature=(
            getattr(args, "evidence_temperature", None)
            if getattr(args, "evidence_temperature", None) is not None
            else saved_aggregator.temperature
        ),
        power=(
            getattr(args, "evidence_power", None)
            if getattr(args, "evidence_power", None) is not None
            else saved_aggregator.power
        ),
    )
    fusion_mode = getattr(args, "fusion_mode", "weighted_rrf")
    direct_weight = getattr(args, "direct_weight", 1.0)
    evidence_weight = getattr(args, "evidence_weight", 0.05)
    score_normalization = getattr(args, "fusion_score_normalization", "none")
    score_temperature = getattr(args, "fusion_score_temperature", 1.0)
    raw_indices = load_or_build_raw_embedding_indices(
        store,
        load_corpus_ids(corpus_path, store),
        Path(selection["raw_embedding_index"]),
        corpus_sha256=corpus_sha256,
        batch_size=args.index_batch_size,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    raw_metrics = evaluate_student_retrieval(
        examples,
        raw_indices,
        recall_ks=args.recall_ks,
        gamma=args.gamma,
        gamma_evidence=args.gamma_evidence,
        direct_k=args.direct_k,
        evidence_k=args.evidence_k,
        targets_per_evidence=args.targets_per_evidence,
        evidence_aggregation=aggregator.evidence_aggregation,
        evidence_top_k=aggregator.top_k,
        evidence_temperature=aggregator.temperature,
        evidence_power=aggregator.power,
        fusion_mode=fusion_mode,
        direct_weight=direct_weight,
        evidence_weight=evidence_weight,
        fusion_score_normalization=score_normalization,
        fusion_score_temperature=score_temperature,
    )
    metrics = evaluate_student_retrieval(
        examples,
        indices,
        recall_ks=args.recall_ks,
        gamma=args.gamma,
        gamma_evidence=args.gamma_evidence,
        direct_k=args.direct_k,
        evidence_k=args.evidence_k,
        targets_per_evidence=args.targets_per_evidence,
        evidence_aggregation=aggregator.evidence_aggregation,
        evidence_top_k=aggregator.top_k,
        evidence_temperature=aggregator.temperature,
        evidence_power=aggregator.power,
        fusion_mode=fusion_mode,
        direct_weight=direct_weight,
        evidence_weight=evidence_weight,
        fusion_score_normalization=score_normalization,
        fusion_score_temperature=score_temperature,
        identity_baseline_metrics=raw_metrics,
    )
    payload = {
        "format_version": 1,
        "selection": str(Path(args.selection).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": selection["best_epoch"],
        "corpus_sha256": corpus_sha256,
        "path_aggregation": {
            "evidence_aggregation": aggregator.evidence_aggregation,
            "evidence_top_k": aggregator.top_k,
            "evidence_temperature": aggregator.temperature,
            "evidence_power": aggregator.power,
        },
        "fusion": {
            "mode": fusion_mode,
            "direct_weight": direct_weight,
            "evidence_weight": evidence_weight,
            "score_normalization": score_normalization,
            "score_temperature": score_temperature,
        },
        "metrics": metrics,
        "raw_embedding": raw_metrics,
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--dev-data", nargs="+", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--evidence-aggregation", choices=sorted(PATH_AGGREGATIONS))
    parser.add_argument("--evidence-top-k", type=int)
    parser.add_argument("--evidence-temperature", type=float)
    parser.add_argument("--evidence-power", type=float)
    parser.add_argument(
        "--fusion-mode",
        choices=["rrf", "weighted_rrf", "gated", "normalized_score", "normalized_rrc"],
        default="weighted_rrf",
    )
    parser.add_argument("--direct-weight", type=float, default=1.0)
    parser.add_argument("--evidence-weight", type=float, default=0.05)
    parser.add_argument(
        "--fusion-score-normalization",
        choices=["none", "zscore", "minmax", "softmax"],
        default="none",
    )
    parser.add_argument("--fusion-score-temperature", type=float, default=1.0)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--recall-ks", type=_parse_recall_ks, default=(10, 20, 30, 40, 50))
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--gamma-evidence", type=int, default=2)
    parser.add_argument("--direct-k", type=int)
    parser.add_argument("--evidence-k", type=int)
    parser.add_argument("--targets-per-evidence", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
