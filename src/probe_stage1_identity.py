#!/usr/bin/env python
"""Evaluate a zero-training identity Student through the Student ANN pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.data import load_target_examples
from mmdd_stage1.evaluation import evaluate_student_retrieval
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import IdentityStudentJoinabilityModel
from mmdd_stage1.retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    build_indices,
    checkpoint_fingerprint,
    load_corpus_ids,
)


def _first_corpus_id(ids_by_type: dict[str, list[str]]) -> str:
    try:
        return next(
            object_id
            for object_ids in ids_by_type.values()
            for object_id in object_ids
        )
    except StopIteration as exc:
        raise ValueError("Cannot run the identity probe on an empty corpus") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cpu":
        raise ValueError("The zero-training identity probe must run on CPU")

    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    ids_by_type = load_corpus_ids(corpus_path, store)
    first_features = store.embedding_features(_first_corpus_id(ids_by_type))
    embedding_dim = int(first_features.embedding.shape[0])
    model = IdentityStudentJoinabilityModel(embedding_dim)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "identity_student.pt"
    torch.save(
        {
            "stage": "identity-probe",
            "model_config": model.config(),
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )
    identity_sha256 = checkpoint_fingerprint(checkpoint_path)
    manifest = build_indices(
        model,
        store,
        ids_by_type,
        output_dir,
        device=device,
        checkpoint_sha256=identity_sha256,
        corpus_sha256=corpus_sha256,
        batch_size=args.batch_size,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    identity_indices = StudentANNIndices(
        model,
        store,
        output_dir,
        device=device,
        checkpoint_sha256=identity_sha256,
        corpus_sha256=corpus_sha256,
    )
    examples = [
        example
        for path in args.dev_data
        for example in load_target_examples(Path(path), split="dev")
    ]
    metrics = evaluate_student_retrieval(
        examples,
        identity_indices,
        direct_k=args.direct_k,
        evidence_k=args.evidence_k,
        targets_per_evidence=args.targets_per_evidence,
        evidence_aggregation=args.evidence_aggregation,
        evidence_top_k=args.evidence_top_k,
        rrf_k=args.rrf_k,
    )
    result: dict[str, Any] = {
        "probe": "identity_student",
        "training_steps": 0,
        "embedding_dim": embedding_dim,
        "corpus_objects": sum(
            record["objects"] for record in manifest["types"].values()
        ),
        "identity": metrics,
    }
    if args.raw_index:
        raw_indices = RawEmbeddingANNIndices(
            store,
            Path(args.raw_index),
            corpus_sha256=corpus_sha256,
        )
        raw_metrics = evaluate_student_retrieval(
            examples,
            raw_indices,
            direct_k=args.direct_k,
            evidence_k=args.evidence_k,
            targets_per_evidence=args.targets_per_evidence,
            evidence_aggregation=args.evidence_aggregation,
            evidence_top_k=args.evidence_top_k,
            rrf_k=args.rrf_k,
        )
        result["raw_embedding"] = raw_metrics
        result["direct_recall@10_delta"] = (
            metrics["direct"]["recall@10"]
            - raw_metrics["direct"]["recall@10"]
        )

    result_path = output_dir / "metrics.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dev-data", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--raw-index",
        help="Optional existing raw embedding index for a same-run parity check.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--direct-k", type=int, default=100)
    parser.add_argument("--evidence-k", type=int, default=50)
    parser.add_argument("--targets-per-evidence", type=int, default=50)
    parser.add_argument(
        "--evidence-aggregation",
        choices=("logsumexp", "topk_mean", "topk_sum"),
        default="logsumexp",
    )
    parser.add_argument("--evidence-top-k", type=int, default=4)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
