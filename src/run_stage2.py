#!/usr/bin/env python
"""Run RATA/FOCUS verification for one Stage-1 retrieval result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmdd_stage1.feature_cache import FeatureStore
from mmdd_stage1.export import validate_stage2_gate
from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.data import (
    direct_target_ids,
    iter_retrieval_results,
    load_stage2_objects,
    validate_retrieval_path_budget,
)
from mmdd_stage2.pipeline import Stage2Verifier
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.routing import SimilarityEvidenceRouter
from mmdd_stage2.verifier import build_evidence_bundles


def run(args: argparse.Namespace) -> dict:
    if args.input_candidate_budget <= 0:
        raise ValueError("--input-candidate-budget must be positive")
    if args.recovery_budget < 0:
        raise ValueError("--recovery-budget must be non-negative")
    if args.top_k_evidence <= 0:
        raise ValueError("--top-k-evidence must be positive")
    validate_stage2_gate(
        Path(args.stage1_gate), [Path(args.retrieval_results)]
    )
    records = list(iter_retrieval_results(Path(args.retrieval_results)))
    if args.query_id:
        records = [record for record in records if str(record["query_id"]) == args.query_id]
    if len(records) != 1:
        raise ValueError("Select exactly one retrieval record with --query-id")
    record = records[0]
    validate_retrieval_path_budget(
        record,
        max_targets=args.input_candidate_budget,
        top_k_evidence=args.top_k_evidence,
    )
    results = record["results"][: args.input_candidate_budget]
    bundles = build_evidence_bundles(results, top_k_evidence=args.top_k_evidence)
    direct_ids = direct_target_ids(results)
    objects = load_stage2_objects(
        Path(args.dataset_root),
        str(record["query_id"]),
        bundles,
        extra_target_ids=direct_ids,
    )
    scorer = load_candidate_scorer(
        Path(args.scorer_checkpoint),
        torch.device("cpu"),
        expected_model_dir=Path(args.model_dir),
    )
    backend = QwenStage2Backend(
        Path(args.model_dir),
        device=args.device,
        dtype=args.dtype,
        focus_start_layer=args.focus_start_layer,
        max_text_evidence_tokens=args.max_text_evidence_tokens,
        text_overlap_tokens=args.text_overlap_tokens,
        max_span_tokens=args.max_span_tokens,
        roi_candidates=args.roi_candidates,
        embedding_batch_size=args.embedding_batch_size,
        max_embedding_tokens=args.max_embedding_tokens,
    )
    scorer.to(backend.device)
    evidence_router = SimilarityEvidenceRouter(FeatureStore.from_path(Path(args.stage1_features)))
    verifier = Stage2Verifier(
        backend,
        scorer,
        evidence_router=evidence_router,
        similarity_threshold=args.similarity_threshold,
        min_row_coverage=args.min_row_coverage,
        similarity_batch_size=args.similarity_batch_size,
    )
    payload = verifier.verify(
        objects.query,
        results,
        objects.targets,
        objects.evidence,
        recovery_budget=args.recovery_budget,
        top_k_evidence=args.top_k_evidence,
    ).to_dict()
    payload["input_candidate_budget"] = args.input_candidate_budget
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--retrieval-results", required=True)
    parser.add_argument(
        "--stage1-gate",
        required=True,
        help="Final dev-gated Stage-1 selection manifest.",
    )
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument(
        "--stage1-features",
        required=True,
        help="Feature cache containing query row_embeddings and retrieved evidence embeddings.",
    )
    parser.add_argument("--query-id")
    parser.add_argument("--output")
    parser.add_argument("--model-dir", default="hf_models/Qwen3.5-9B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--focus-start-layer", type=int, default=14)
    parser.add_argument("--top-k-evidence", type=int, default=4)
    parser.add_argument(
        "--input-candidate-budget", "--max-targets",
        dest="input_candidate_budget", type=int, default=50,
        help="N: unique Stage-1 input targets, all requiring path detail (default: 50). "
        "--max-targets is a legacy alias for this input budget only.",
    )
    parser.add_argument(
        "--recovery-budget", type=int, default=20,
        help="M: unique evidence targets to recover after full-pool column scoring (default: 20). "
        "Direct paths are all verified and do not consume this budget.",
    )
    parser.add_argument("--max-text-evidence-tokens", type=int, default=1024)
    parser.add_argument("--text-overlap-tokens", type=int, default=128)
    parser.add_argument("--max-span-tokens", type=int, default=192)
    parser.add_argument("--roi-candidates", type=int, default=4)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--max-embedding-tokens", type=int, default=128)
    parser.add_argument("--similarity-batch-size", type=int, default=1024)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--min-row-coverage", type=float, default=0.6)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
