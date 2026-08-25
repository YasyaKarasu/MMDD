#!/usr/bin/env python
"""Build one Student HNSW index for each destination object type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import build_indices, checkpoint_fingerprint, load_corpus_ids


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.student_checkpoint)
    model = load_student(checkpoint_path, device)
    store = FeatureStore.from_path(Path(args.features), cache_size=args.feature_cache_size)
    ids_by_type = load_corpus_ids(Path(args.corpus), store)
    manifest = build_indices(
        model,
        store,
        ids_by_type,
        Path(args.output_dir),
        device=device,
        checkpoint_sha256=checkpoint_fingerprint(checkpoint_path),
        batch_size=args.batch_size,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "objects": sum(record["objects"] for record in manifest["types"].values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--corpus", required=True, help="JSONL containing object_id and optional object_type.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
