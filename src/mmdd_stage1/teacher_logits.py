"""Persistent frozen-Teacher logits for Student training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
from mmdd_progress import progress
from torch.nn.utils.rnn import pad_sequence

from .data import (
    EdgeExample,
    TargetExample,
    TeacherScoreConfig,
)
from .features import FeatureStore
from .models import TeacherJoinabilityModel
from .objectives import PathAggregator
from .scoring import score_edge_batch, score_target_batch

TrainingExample = EdgeExample | TargetExample


def _example_record(example: TrainingExample) -> dict[str, Any]:
    if isinstance(example, EdgeExample):
        return {
            "query_id": example.query_id,
            "candidate_ids": example.candidate_ids,
            "positive_index": example.positive_index,
            "source_type": example.source_type,
            "destination_type": example.destination_type,
            "dataset": example.dataset,
            "split": example.split,
        }
    return {
        "query_id": example.query_id,
        "candidates": [
            (candidate.target_id, candidate.evidence_ids)
            for candidate in example.candidates
        ],
        "direct_positive_index": example.direct_positive_index,
        "evidence_positive_index": example.evidence_positive_index,
        "dataset": example.dataset,
        "split": example.split,
    }


def examples_fingerprint(examples: Sequence[TrainingExample]) -> str:
    payload = json.dumps(
        [_example_record(example) for example in examples],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(
    cache_dir: Path,
    examples: Sequence[TrainingExample],
    teacher_sha256: str,
    aggregator: PathAggregator | None,
) -> Path:
    kind = "edge" if isinstance(examples[0], EdgeExample) else "target"
    aggregation = (
        ""
        if aggregator is None
        else f"-{aggregator.evidence_aggregation}-k{aggregator.top_k}"
    )
    name = f"{kind}{aggregation}-{examples_fingerprint(examples)}.pt"
    return cache_dir / teacher_sha256 / name


def has_teacher_logits(
    examples: Sequence[TrainingExample],
    teacher_sha256: str | None = None,
    aggregator: PathAggregator | None = None,
) -> bool:
    score_config = (
        TeacherScoreConfig(aggregator.evidence_aggregation, aggregator.top_k)
        if aggregator is not None
        else None
    )
    return all(
        (
            (
                example.teacher_logits is not None
                if isinstance(example, EdgeExample)
                else example.teacher_direct_logits is not None
                and example.teacher_evidence_logits is not None
                and (
                    score_config is None
                    or example.teacher_score_config == score_config
                )
            )
            and (
                teacher_sha256 is None
                or example.teacher_checkpoint_sha256 == teacher_sha256
            )
        )
        for example in examples
    )


def load_teacher_logits(
    examples: Sequence[TrainingExample],
    cache_dir: Path,
    teacher_sha256: str,
    aggregator: PathAggregator | None = None,
) -> tuple[list[TrainingExample], Path, bool]:
    path = _cache_path(cache_dir, examples, teacher_sha256, aggregator)
    if not path.is_file():
        return list(examples), path, False
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "format_version": 1,
        "teacher_checkpoint_sha256": teacher_sha256,
        "examples_sha256": examples_fingerprint(examples),
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"{path}: Teacher logit cache metadata does not match")

    if isinstance(examples[0], EdgeExample):
        logits = payload.get("teacher_logits")
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 2
            or logits.shape[0] != len(examples)
            or any(
                logits.shape[1] < len(example.candidate_ids)
                for example in examples
            )
        ):
            raise ValueError(f"{path}: Teacher edge logits do not align with examples")
        loaded = [
            replace(
                example,
                teacher_logits=tuple(
                    float(value)
                    for value in logits[row, : len(example.candidate_ids)].tolist()
                ),
                teacher_checkpoint_sha256=teacher_sha256,
            )
            for row, example in enumerate(examples)
        ]
    else:
        assert aggregator is not None
        direct = payload.get("teacher_direct_logits")
        evidence = payload.get("teacher_evidence_logits")
        if (
            not isinstance(direct, torch.Tensor)
            or not isinstance(evidence, torch.Tensor)
            or direct.ndim != 2
            or evidence.ndim != 2
            or direct.shape[0] != len(examples)
            or evidence.shape[0] != len(examples)
            or payload.get("evidence_aggregation")
            != aggregator.evidence_aggregation
            or payload.get("evidence_top_k") != aggregator.top_k
            or any(
                direct.shape[1] < len(example.candidates)
                or evidence.shape[1] < len(example.candidates)
                for example in examples
            )
        ):
            raise ValueError(f"{path}: Teacher target logits do not align with examples")
        config = TeacherScoreConfig(
            aggregator.evidence_aggregation, aggregator.top_k
        )
        loaded = [
            replace(
                example,
                teacher_direct_logits=tuple(
                    float(value)
                    for value in direct[row, : len(example.candidates)].tolist()
                ),
                teacher_evidence_logits=tuple(
                    float(value)
                    for value in evidence[
                        row, : len(example.candidates)
                    ].tolist()
                ),
                teacher_score_config=config,
                teacher_checkpoint_sha256=teacher_sha256,
            )
            for row, example in enumerate(examples)
        ]
    return loaded, path, True


@torch.no_grad()
def score_and_cache_teacher_logits(
    examples: Sequence[TrainingExample],
    teacher: TeacherJoinabilityModel,
    store: FeatureStore,
    cache_dir: Path,
    teacher_sha256: str,
    *,
    device: torch.device,
    batch_size: int,
    aggregator: PathAggregator | None = None,
) -> tuple[list[TrainingExample], Path]:
    result = list(examples)
    missing = [
        index
        for index, example in enumerate(result)
        if not has_teacher_logits([example], teacher_sha256, aggregator)
    ]
    teacher.eval()
    starts = range(0, len(missing), batch_size)
    for start in progress(
        starts,
        total=len(starts),
        desc="Teacher logits",
        unit="batch",
        leave=False,
    ):
        indices = missing[start : start + batch_size]
        batch = [result[index] for index in indices]
        if isinstance(batch[0], EdgeExample):
            scores = score_edge_batch(teacher, batch, store, device)
            for row, index in enumerate(indices):
                example = result[index]
                assert isinstance(example, EdgeExample)
                count = len(example.candidate_ids)
                result[index] = replace(
                    example,
                    teacher_logits=tuple(
                        float(value)
                        for value in scores.logits[row, :count].cpu().tolist()
                    ),
                    teacher_checkpoint_sha256=teacher_sha256,
                )
        else:
            assert aggregator is not None
            scores = score_target_batch(
                teacher, batch, store, device, aggregator
            )
            for row, index in enumerate(indices):
                example = result[index]
                assert isinstance(example, TargetExample)
                count = len(example.candidates)
                result[index] = replace(
                    example,
                    teacher_direct_logits=tuple(
                        float(value)
                        for value in scores.direct.logits[row, :count].cpu().tolist()
                    ),
                    teacher_evidence_logits=tuple(
                        float(value)
                        for value in scores.evidence.logits[row, :count].cpu().tolist()
                    ),
                    teacher_score_config=TeacherScoreConfig(
                        aggregator.evidence_aggregation, aggregator.top_k
                    ),
                    teacher_checkpoint_sha256=teacher_sha256,
                )

    path = _cache_path(cache_dir, result, teacher_sha256, aggregator)
    payload: dict[str, Any] = {
        "format_version": 1,
        "teacher_checkpoint_sha256": teacher_sha256,
        "examples_sha256": examples_fingerprint(result),
    }
    if isinstance(result[0], EdgeExample):
        payload["teacher_logits"] = pad_sequence(
            [torch.tensor(example.teacher_logits) for example in result],
            batch_first=True,
        ).float()
    else:
        assert aggregator is not None
        payload["evidence_aggregation"] = aggregator.evidence_aggregation
        payload["evidence_top_k"] = aggregator.top_k
        payload["teacher_direct_logits"] = pad_sequence(
            [torch.tensor(example.teacher_direct_logits) for example in result],
            batch_first=True,
        ).float()
        payload["teacher_evidence_logits"] = pad_sequence(
            [torch.tensor(example.teacher_evidence_logits) for example in result],
            batch_first=True,
        ).float()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return result, path
