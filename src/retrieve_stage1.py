#!/usr/bin/env python
"""Run directed zero/one-hop Student retrieval for one query object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmdd_stage1.checkpoints import load_path_aggregator, load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import STUDENT_SCORE_SPACES
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    checkpoint_fingerprint,
    retrieve_zero_one_hop,
)
from mmdd_stage1.objectives import PATH_AGGREGATIONS


def _modality_weight(value: str) -> tuple[str, float]:
    modality, separator, raw_weight = value.partition("=")
    if separator != "=" or modality not in {"text", "image"}:
        raise argparse.ArgumentTypeError("expected text=WEIGHT or image=WEIGHT")
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("modality weight must be numeric") from exc
    if weight < 0:
        raise argparse.ArgumentTypeError("modality weight must be non-negative")
    return modality, weight


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.student_checkpoint)
    model = load_student(checkpoint_path, device)
    saved_aggregator = load_path_aggregator(checkpoint_path)
    evidence_aggregation = (
        args.evidence_aggregation or saved_aggregator.evidence_aggregation
    )
    evidence_top_k = (
        args.evidence_top_k
        if args.evidence_top_k is not None
        else saved_aggregator.top_k
    )
    evidence_temperature = (
        args.evidence_temperature
        if args.evidence_temperature is not None
        else saved_aggregator.temperature
    )
    evidence_power = (
        args.evidence_power
        if args.evidence_power is not None
        else saved_aggregator.power
    )
    path_combination = (
        args.path_combination
        if args.path_combination is not None
        else saved_aggregator.path_combination
    )
    evidence_threshold = (
        args.evidence_threshold
        if args.evidence_threshold is not None
        else saved_aggregator.threshold
    )
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
        score_space=args.student_score_space,
    )
    results = retrieve_zero_one_hop(
        args.query_id,
        indices,
        k=args.k,
        gamma=args.gamma,
        gamma_evidence=args.gamma_evidence,
        direct_k=args.direct_k,
        evidence_k=args.evidence_k,
        targets_per_evidence=args.targets_per_evidence,
        result_k=args.result_k,
        evidence_types=tuple(args.evidence_types),
        evidence_aggregation=evidence_aggregation,
        evidence_top_k=evidence_top_k,
        evidence_temperature=evidence_temperature,
        evidence_power=evidence_power,
        path_combination=path_combination,
        evidence_threshold=evidence_threshold,
        evidence_target_temperature=saved_aggregator.target_temperature,
        row_support_model=saved_aggregator.row_support_model,
        row_support_model_sha256=saved_aggregator.row_support_model_sha256,
        row_support_top_l=saved_aggregator.row_support_top_l,
        evidence_content_keys=saved_aggregator.evidence_content_keys,
        evidence_content_keys_sha256=(
            saved_aggregator.evidence_content_keys_sha256
        ),
        rrf_k=args.rrf_k,
        fusion_mode=args.fusion_mode,
        direct_weight=args.direct_weight,
        evidence_weight=args.evidence_weight,
        fusion_score_normalization=args.fusion_score_normalization,
        fusion_score_temperature=args.fusion_score_temperature,
        gated_evidence_min_paths=args.gated_evidence_min_paths,
        gated_evidence_quantile=args.gated_evidence_quantile,
        evidence_modality_weights=dict(args.evidence_modality_weights),
        path_result_k=args.path_result_k,
        evidence_path_k=args.evidence_path_k,
    )
    payload = json.dumps(
        {
            "query_id": args.query_id,
            "student_checkpoint_sha256": checkpoint_fingerprint(checkpoint_path),
            "path_aggregation": {
                "k": args.k,
                "gamma": args.gamma,
                "gamma_evidence": args.gamma_evidence,
                "evidence_aggregation": evidence_aggregation,
                "evidence_top_k": evidence_top_k,
                "evidence_temperature": evidence_temperature,
                "evidence_power": evidence_power,
                "path_combination": path_combination,
                "evidence_threshold": evidence_threshold,
                "evidence_target_temperature": (
                    saved_aggregator.target_temperature
                ),
                "row_support_model": saved_aggregator.row_support_model,
                "row_support_model_sha256": (
                    saved_aggregator.row_support_model_sha256
                ),
                "row_support_top_l": saved_aggregator.row_support_top_l,
                "evidence_content_keys": saved_aggregator.evidence_content_keys,
                "evidence_content_keys_sha256": (
                    saved_aggregator.evidence_content_keys_sha256
                ),
                "student_score_space": args.student_score_space,
                "target_fusion": args.fusion_mode,
                "fusion_score_normalization": args.fusion_score_normalization,
                "fusion_score_temperature": args.fusion_score_temperature,
                "rrf_k": args.rrf_k,
                "direct_weight": args.direct_weight,
                "evidence_weight": args.evidence_weight,
                "gated_evidence_min_paths": args.gated_evidence_min_paths,
                "gated_evidence_quantile": args.gated_evidence_quantile,
                "evidence_modality_weights": dict(
                    args.evidence_modality_weights
                ),
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
    parser.add_argument(
        "--student-score-space",
        choices=STUDENT_SCORE_SPACES,
        default="raw_logit",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--gamma-evidence", type=int, default=2)
    parser.add_argument("--direct-k", type=int)
    parser.add_argument("--evidence-k", type=int)
    parser.add_argument("--targets-per-evidence", type=int)
    parser.add_argument(
        "--result-k",
        type=int,
        help="Advanced serialized-result limit; defaults to --k.",
    )
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
    parser.add_argument("--evidence-aggregation", choices=sorted(PATH_AGGREGATIONS))
    parser.add_argument("--evidence-top-k", type=int)
    parser.add_argument("--evidence-temperature", type=float)
    parser.add_argument("--evidence-power", type=float)
    parser.add_argument("--path-combination", choices=["sum", "min", "product"])
    parser.add_argument("--evidence-threshold", type=float)
    parser.add_argument("--rrf-k", type=int, default=60)
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
    parser.add_argument("--gated-evidence-min-paths", type=int, default=2)
    parser.add_argument("--gated-evidence-quantile", type=float, default=0.75)
    parser.add_argument("--evidence-modality-weights", nargs="*", type=_modality_weight, default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
