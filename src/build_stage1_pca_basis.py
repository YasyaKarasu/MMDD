#!/usr/bin/env python
"""Build a top-k PCA projection from the frozen Stage-1 retrieval corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmdd_progress import progress

from mmdd_stage1.features import OBJECT_TYPES, FeatureStore
from mmdd_stage1.pca import PCA_FORMAT_VERSION, compute_pca_projection
from mmdd_stage1.retrieval import checkpoint_fingerprint, load_corpus_ids


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    corpus_path = Path(args.corpus)
    ids_by_type = load_corpus_ids(corpus_path, store)
    object_ids = [
        object_id
        for object_type in OBJECT_TYPES
        for object_id in ids_by_type[object_type]
    ]
    input_dim = store.embedding_dimension()
    embeddings = torch.empty(len(object_ids), input_dim)
    for row, object_id in enumerate(
        progress(object_ids, desc="Load PCA embeddings", unit="object")
    ):
        embeddings[row].copy_(store.embedding_features(object_id).embedding)

    projection, mean, explained_variance_ratio = compute_pca_projection(
        embeddings,
        args.student_dim,
        device=device,
        oversampling=args.oversampling,
        iterations=args.iterations,
        seed=args.seed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": PCA_FORMAT_VERSION,
        "input_dim": input_dim,
        "student_dim": args.student_dim,
        "objects": len(object_ids),
        "corpus_sha256": checkpoint_fingerprint(corpus_path),
        "seed": args.seed,
        "oversampling": args.oversampling,
        "iterations": args.iterations,
        "explained_variance_ratio": explained_variance_ratio,
        "mean": mean,
        "projection": projection,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    summary = {
        "output": str(output_path),
        "objects": len(object_ids),
        "input_dim": input_dim,
        "student_dim": args.student_dim,
        "explained_variance_ratio": explained_variance_ratio,
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--student-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--oversampling", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
