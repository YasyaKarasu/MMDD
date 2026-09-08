#!/usr/bin/env python
"""Evaluate directed Stage-1 edge ranking and confidence quality."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from mmdd_progress import progress

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_checkpoint, load_student, load_teacher
from mmdd_stage1.data import EdgeExample, load_edge_examples
from mmdd_stage1.edge_metrics import summarize_edge_quality
from mmdd_stage1.features import FeatureStore, normalize_object_type
from mmdd_stage1.models import (
    STUDENT_SCORE_SPACES,
    StudentJoinabilityModel,
    TeacherJoinabilityModel,
)
from mmdd_stage1.scoring import score_edge_batch


def _parse_recall_ks(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "--recall-ks must contain positive integers"
        )
    return tuple(dict.fromkeys(values))


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _required_object_ids(examples: Sequence[EdgeExample]) -> list[str]:
    return list(
        dict.fromkeys(
            object_id
            for example in examples
            for object_id in (example.query_id, *example.candidate_ids)
        )
    )


def _batches(
    examples: Sequence[EdgeExample], batch_size: int
) -> list[Sequence[EdgeExample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        examples[start : start + batch_size]
        for start in range(0, len(examples), batch_size)
    ]


def _trim_rows(
    values: torch.Tensor, examples: Sequence[EdgeExample]
) -> list[list[float]]:
    return [
        [float(value) for value in values[row, : len(example.candidate_ids)].tolist()]
        for row, example in enumerate(examples)
    ]


def _validate_declared_types(
    example: EdgeExample, store: FeatureStore
) -> None:
    if example.source_type is not None:
        actual = store.object_type(example.query_id)
        declared = normalize_object_type(example.source_type)
        if actual != declared:
            raise ValueError(
                f"{example.query_id}: declared source_type {declared!r} "
                f"does not match cached type {actual!r}"
            )
    if example.destination_type is not None:
        declared = normalize_object_type(example.destination_type)
        for candidate_id in example.candidate_ids:
            actual = store.object_type(candidate_id)
            if actual != declared:
                raise ValueError(
                    f"{candidate_id}: declared destination_type {declared!r} "
                    f"does not match cached type {actual!r}"
                )


@torch.no_grad()
def _score_raw_batch(
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
) -> tuple[list[list[float]], list[list[float]]]:
    source_embeddings = []
    destination_embeddings = []
    lengths = []
    for example in examples:
        _validate_declared_types(example, store)
        source = store.embedding_features(example.query_id).embedding
        lengths.append(len(example.candidate_ids))
        source_embeddings.extend([source] * len(example.candidate_ids))
        destination_embeddings.extend(
            store.embedding_features(candidate_id).embedding
            for candidate_id in example.candidate_ids
        )
    flat_scores = torch.nn.functional.cosine_similarity(
        torch.stack(source_embeddings).to(device=device, dtype=torch.float32),
        torch.stack(destination_embeddings).to(device=device, dtype=torch.float32),
        dim=-1,
    )
    rows = list(flat_scores.split(lengths))
    ranking = [[float(value) for value in row.tolist()] for row in rows]
    confidence = [
        [float(value) for value in torch.sigmoid(row).tolist()] for row in rows
    ]
    return ranking, confidence


@torch.no_grad()
def _score_model_batch(
    model: StudentJoinabilityModel | TeacherJoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    *,
    student_score_space: str,
) -> tuple[list[list[float]], list[list[float]]]:
    ranking = score_edge_batch(
        model,
        examples,
        store,
        device,
        student_score_space=student_score_space,
    ).logits
    if isinstance(model, StudentJoinabilityModel):
        confidence = score_edge_batch(
            model,
            examples,
            store,
            device,
            student_score_space="confidence",
        ).logits
    else:
        confidence = torch.sigmoid(ranking)
    return _trim_rows(ranking, examples), _trim_rows(confidence, examples)


def _load_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[StudentJoinabilityModel | TeacherJoinabilityModel | None, dict[str, Any]]:
    if args.model_kind == "raw":
        if args.checkpoint is not None:
            raise ValueError("--checkpoint is not used with --model-kind raw")
        return None, {
            "kind": "raw",
            "ranking_score_space": "raw_embedding_cosine_similarity",
            "confidence_mapping": "uncalibrated_sigmoid_of_cosine_similarity",
        }
    if args.checkpoint is None:
        raise ValueError(f"--model-kind {args.model_kind} requires --checkpoint")

    checkpoint = Path(args.checkpoint).resolve()
    payload = load_checkpoint(checkpoint)
    common = {
        "kind": args.model_kind,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "config": payload["config"],
    }
    if args.model_kind == "student":
        model = load_student(checkpoint, device).eval()
        common.update(
            {
                "ranking_score_space": args.student_score_space,
                "confidence_mapping": (
                    "learned_relation_affine_sigmoid"
                    if model.confidence_transform
                    else "uncalibrated_sigmoid_of_raw_logit"
                ),
            }
        )
        return model, common

    model = load_teacher(checkpoint, device).eval()
    if args.teacher_amp == "bf16":
        if device.type != "cuda":
            raise ValueError("--teacher-amp bf16 requires a CUDA device")
        model.set_compute_dtype(torch.bfloat16)
    common.update(
        {
            "ranking_score_space": "teacher_logit",
            "confidence_mapping": "uncalibrated_sigmoid_of_teacher_logit",
            "teacher_amp": args.teacher_amp,
        }
    )
    return model, common


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    model, model_metadata = _load_model(args, device)
    edge_path = Path(args.edge_data).resolve()
    split = None if args.split == "all" else args.split
    examples = load_edge_examples(edge_path, split=split)
    store = FeatureStore.from_path(
        Path(args.features),
        cache_size=args.feature_cache_size,
        teacher_paths=[Path(value) for value in args.teacher_features],
    )
    if model is None or isinstance(model, StudentJoinabilityModel):
        store.preload_embeddings(_required_object_ids(examples))

    ranking_scores: list[list[float]] = []
    confidence_scores: list[list[float]] = []
    batches = _batches(examples, args.batch_size)
    for batch in progress(batches, desc="Evaluate edges", unit="batch"):
        if model is None:
            ranking, confidence = _score_raw_batch(batch, store, device)
        else:
            ranking, confidence = _score_model_batch(
                model,
                batch,
                store,
                device,
                student_score_space=args.student_score_space,
            )
        ranking_scores.extend(ranking)
        confidence_scores.extend(confidence)

    metrics = summarize_edge_quality(
        examples,
        ranking_scores,
        confidence_scores,
        recall_ks=args.recall_ks,
        reliability_bins=args.reliability_bins,
    )
    payload = {
        "format_version": 1,
        "edge_data": str(edge_path),
        "edge_data_sha256": checkpoint_fingerprint(edge_path),
        "features": str(Path(args.features).resolve()),
        "split": args.split,
        "examples": len(examples),
        "candidate_occurrences": sum(
            len(example.candidate_ids) for example in examples
        ),
        "device": str(device),
        "batch_size": args.batch_size,
        "model": model_metadata,
        "feature_cache": store.cache_info(),
        "metrics": metrics,
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-kind", choices=("raw", "student", "teacher"), required=True
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--features", required=True)
    parser.add_argument("--teacher-features", nargs="*", default=[])
    parser.add_argument("--edge-data", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument(
        "--student-score-space",
        choices=STUDENT_SCORE_SPACES,
        default="raw_logit",
    )
    parser.add_argument("--teacher-amp", choices=("off", "bf16"), default="off")
    parser.add_argument(
        "--recall-ks", type=_parse_recall_ks, default=(1, 5, 10, 20)
    )
    parser.add_argument("--reliability-bins", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
