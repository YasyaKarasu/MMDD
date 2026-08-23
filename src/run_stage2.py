#!/usr/bin/env python
"""Run RATA/FOCUS verification for one Stage-1 retrieval result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.data import direct_target_ids, iter_retrieval_results, load_stage2_objects
from mmdd_stage2.pipeline import Stage2Verifier
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.verifier import build_evidence_bundles


def run(args: argparse.Namespace) -> dict:
    records = list(iter_retrieval_results(Path(args.retrieval_results)))
    if args.query_id:
        records = [record for record in records if str(record["query_id"]) == args.query_id]
    if len(records) != 1:
        raise ValueError("Select exactly one retrieval record with --query-id")
    record = records[0]
    results = record["results"][: args.max_targets]
    bundles = build_evidence_bundles(results, top_k_evidence=args.top_k_evidence)
    direct_ids = direct_target_ids(results)[: args.max_direct_targets]
    objects = load_stage2_objects(
        Path(args.dataset_root),
        str(record["query_id"]),
        bundles,
        extra_target_ids=direct_ids,
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
    )
    scorer = load_candidate_scorer(Path(args.scorer_checkpoint), backend.device)
    verifier = Stage2Verifier(
        backend,
        scorer,
        similarity_threshold=args.similarity_threshold,
        min_row_coverage=args.min_row_coverage,
    )
    payload = verifier.verify(
        objects.query,
        bundles,
        objects.targets,
        objects.evidence,
        direct_target_ids=direct_ids,
    ).to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--retrieval-results", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--query-id")
    parser.add_argument("--output")
    parser.add_argument("--model-dir", default="hf_models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--focus-start-layer", type=int, default=14)
    parser.add_argument("--top-k-evidence", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=10)
    parser.add_argument("--max-direct-targets", type=int, default=5)
    parser.add_argument("--max-text-evidence-tokens", type=int, default=1024)
    parser.add_argument("--text-overlap-tokens", type=int, default=128)
    parser.add_argument("--max-span-tokens", type=int, default=192)
    parser.add_argument("--roi-candidates", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--min-row-coverage", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
