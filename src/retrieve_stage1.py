#!/usr/bin/env python
"""Run directed zero/one-hop Student retrieval for one query object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmdd_stage1.checkpoints import load_path_aggregation, load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    checkpoint_fingerprint,
    retrieve_zero_one_hop,
)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.student_checkpoint)
    model = load_student(checkpoint_path, device)
    saved_aggregation, saved_top_k = load_path_aggregation(checkpoint_path)
    evidence_aggregation = args.evidence_aggregation or saved_aggregation
    evidence_top_k = args.evidence_top_k if args.evidence_top_k is not None else saved_top_k
    model.eval()
    store = FeatureStore.from_path(Path(args.features), cache_size=args.feature_cache_size)
    corpus_sha256 = (
        checkpoint_fingerprint(Path(args.corpus)) if args.corpus else None
    )
    indices = StudentANNIndices(
        model,
        store,
        Path(args.index_dir),
        device=device,
        checkpoint_sha256=checkpoint_fingerprint(checkpoint_path),
        corpus_sha256=corpus_sha256,
    )
    results = retrieve_zero_one_hop(
        args.query_id,
        indices,
        direct_k=args.direct_k,
        evidence_k=args.evidence_k,
        targets_per_evidence=args.targets_per_evidence,
        result_k=args.result_k,
        evidence_types=tuple(args.evidence_types),
        evidence_aggregation=evidence_aggregation,
        evidence_top_k=evidence_top_k,
        rrf_k=args.rrf_k,
        path_result_k=args.path_result_k,
        evidence_path_k=args.evidence_path_k,
    )
    payload = json.dumps(
        {
            "query_id": args.query_id,
            "student_checkpoint_sha256": checkpoint_fingerprint(checkpoint_path),
            "path_aggregation": {
                "evidence_aggregation": evidence_aggregation,
                "evidence_top_k": evidence_top_k,
                "target_fusion": "rrf",
                "rrf_k": args.rrf_k,
                "path_result_k": args.path_result_k,
                "evidence_path_k": (
                    evidence_top_k
                    if args.evidence_path_k is None
                    else args.evidence_path_k
                ),
            },
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )
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
    parser.add_argument(
        "--corpus",
        help="Full shared corpus used to build the index; validates its fingerprint.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--direct-k", type=int, default=100)
    parser.add_argument("--evidence-k", type=int, default=50)
    parser.add_argument("--targets-per-evidence", type=int, default=50)
    parser.add_argument("--result-k", type=int, default=100)
    parser.add_argument(
        "--path-result-k",
        type=int,
        default=10,
        help="Keep Stage-2 path detail only for this many globally ranked targets.",
    )
    parser.add_argument(
        "--evidence-path-k",
        type=int,
        help="Evidence paths retained per target; defaults to the saved evidence top-k.",
    )
    parser.add_argument("--evidence-types", nargs="+", choices=["text", "image"], default=["text", "image"])
    parser.add_argument("--evidence-aggregation", choices=["logsumexp", "topk_mean", "topk_sum"])
    parser.add_argument("--evidence-top-k", type=int)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
