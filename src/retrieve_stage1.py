#!/usr/bin/env python
"""Run directed zero/one-hop Student retrieval for one query object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import StudentANNIndices, checkpoint_fingerprint, retrieve_zero_one_hop


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.student_checkpoint)
    model = load_student(checkpoint_path, device)
    model.eval()
    store = FeatureStore.from_path(Path(args.features), cache_size=args.feature_cache_size)
    indices = StudentANNIndices(
        model,
        store,
        Path(args.index_dir),
        device=device,
        checkpoint_sha256=checkpoint_fingerprint(checkpoint_path),
    )
    results = retrieve_zero_one_hop(
        args.query_id,
        indices,
        direct_k=args.direct_k,
        evidence_k=args.evidence_k,
        targets_per_evidence=args.targets_per_evidence,
        result_k=args.result_k,
        evidence_types=tuple(args.evidence_types),
    )
    payload = json.dumps({"query_id": args.query_id, "results": results}, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--output")
    parser.add_argument("--features", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--direct-k", type=int, default=100)
    parser.add_argument("--evidence-k", type=int, default=50)
    parser.add_argument("--targets-per-evidence", type=int, default=50)
    parser.add_argument("--result-k", type=int, default=100)
    parser.add_argument("--evidence-types", nargs="+", choices=["text", "image"], default=["text", "image"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
