#!/usr/bin/env python
"""Export a fixed Stage-1 ANN path pool for aggregation-only experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mmdd_progress import progress

from mmdd_stage1.checkpoints import load_path_aggregator, load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    checkpoint_fingerprint,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.selection import load_stage1_selection


def _paths_by_target(
    detailed: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for channel in ("direct", "evidence"):
        for row in detailed[channel]:
            result[str(row["target_id"])] = row["paths"]
    return result


def _student_score_space(selection: dict[str, Any]) -> str:
    return str(selection.get("student_score_space", "raw_logit"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    selection = load_stage1_selection(selection_path)
    if selection.get("completed_stage") != "student-path":
        raise ValueError("Selection must describe a student-path checkpoint")
    checkpoint_path = Path(selection["best_checkpoint"])
    corpus_path = Path(args.corpus).resolve()
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    if corpus_sha256 != selection.get("corpus_sha256"):
        raise ValueError("Selection and corpus fingerprints differ")

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    if args.system == "student":
        checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
        if checkpoint_sha256 != selection.get("best_checkpoint_sha256"):
            raise ValueError("Selection and checkpoint fingerprints differ")
        student_score_space = _student_score_space(selection)
        student = load_student(checkpoint_path, device).eval()
        indices = StudentANNIndices(
            student,
            store,
            Path(selection["best_index"]),
            device=device,
            checkpoint_sha256=checkpoint_sha256,
            corpus_sha256=corpus_sha256,
            score_space=student_score_space,
        )
    else:
        checkpoint_sha256 = None
        student_score_space = None
        raw_index = Path(selection["raw_embedding_index"])
        if not (raw_index / "manifest.json").is_file():
            raise FileNotFoundError(f"Raw ANN index is incomplete: {raw_index}")
        indices = RawEmbeddingANNIndices(
            store,
            raw_index,
            corpus_sha256=corpus_sha256,
        )

    examples = [
        example
        for value in args.data
        for example in load_target_examples(
            Path(value), split=args.split, dataset_name=Path(value).stem
        )
    ]
    aggregator = load_path_aggregator(checkpoint_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    records = 0
    with temporary.open("w", encoding="utf-8") as handle:
        starts = range(0, len(examples), args.query_batch_size)
        for start in progress(
            starts,
            total=len(starts),
            desc=f"Export {args.system} path pools",
            unit="batch",
        ):
            batch = examples[start : start + args.query_batch_size]
            detailed = retrieve_zero_one_hop_detailed_many(
                [example.query_id for example in batch],
                indices,
                k=max(args.recall_ks),
                direct_k=args.direct_k,
                evidence_k=args.evidence_k,
                targets_per_evidence=args.targets_per_evidence,
                evidence_types=tuple(args.evidence_types),
                evidence_aggregation=aggregator.evidence_aggregation,
                evidence_top_k=aggregator.top_k,
                evidence_temperature=aggregator.temperature,
                evidence_power=aggregator.power,
                path_combination=aggregator.path_combination,
                evidence_threshold=aggregator.threshold,
                evidence_target_temperature=aggregator.target_temperature,
                row_support_model=aggregator.row_support_model,
                row_support_model_sha256=aggregator.row_support_model_sha256,
                row_support_top_l=aggregator.row_support_top_l,
                evidence_content_keys=aggregator.evidence_content_keys,
                evidence_content_keys_sha256=(
                    aggregator.evidence_content_keys_sha256
                ),
                fusion_mode="weighted_rrf",
                direct_weight=1.0,
                evidence_weight=0.05,
                query_batch_size=args.query_batch_size,
            )
            for example, result in zip(batch, detailed):
                positives = set(example.positive_target_ids)
                handle.write(
                    json.dumps(
                        {
                            "query_id": example.query_id,
                            "dataset": example.dataset,
                            "split": example.split,
                            "positive_target_ids": list(example.positive_target_ids),
                            "positive_evidence_by_target": {
                                candidate.target_id: list(candidate.evidence_ids)
                                for candidate in example.candidates
                                if candidate.target_id in positives
                                and candidate.evidence_ids
                            },
                            "paths_by_target": _paths_by_target(result),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                records += 1
    temporary.replace(output_path)

    metadata = {
        "format_version": 1,
        "system": args.system,
        "selection": str(selection_path),
        "student_checkpoint_sha256": checkpoint_sha256,
        "student_score_space": student_score_space,
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_sha256,
        "features": str(Path(args.features).resolve()),
        "data": [str(Path(value).resolve()) for value in args.data],
        "split": args.split,
        "queries": records,
        "recall_ks": list(args.recall_ks),
        "retrieval_budget": {
            "direct_k": args.direct_k,
            "evidence_k_per_modality": args.evidence_k,
            "targets_per_evidence": args.targets_per_evidence,
            "evidence_types": args.evidence_types,
        },
        "path_score_space": (
            f"student_{student_score_space}_{aggregator.path_combination}"
            if args.system == "student"
            else f"raw_cosine_{aggregator.path_combination}"
        ),
        "path_aggregation": aggregator.config(),
        "path_pool_policy": "all unique targets reached by the fixed ANN budget",
        "output": str(output_path.resolve()),
        "output_sha256": checkpoint_fingerprint(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    _write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def _positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part) for part in value.split(",")}))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--system", choices=["raw", "student"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--recall-ks", type=_positive_ints, default=(10, 20, 50))
    parser.add_argument("--direct-k", type=int, default=100)
    parser.add_argument("--evidence-k", type=int, default=20)
    parser.add_argument("--targets-per-evidence", type=int, default=20)
    parser.add_argument(
        "--evidence-types",
        nargs="+",
        choices=["text", "image"],
        default=["text", "image"],
    )
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(
        args.direct_k,
        args.evidence_k,
        args.targets_per_evidence,
        args.query_batch_size,
        args.feature_cache_size,
    ) <= 0:
        parser.error("Retrieval, batch, and cache sizes must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
