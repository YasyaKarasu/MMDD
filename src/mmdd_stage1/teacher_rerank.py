"""Reusable raw-top-k Teacher reranking evaluation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

import torch

from mmdd_progress import progress

from .data import TargetExample
from .features import FeatureStore, ObjectFeatures
from .models import TeacherJoinabilityModel

RECALL_KS = (10, 100)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_scale == 0 or right_scale == 0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _retrieval_metrics(
    rankings: Sequence[Sequence[str]], positives: Sequence[set[str]]
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"queries": len(rankings)}
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = statistics.fmean(
            len(set(ranking[:k]) & relevant) / len(relevant)
            for ranking, relevant in zip(rankings, positives)
        )
    metrics["mrr@100"] = statistics.fmean(
        next(
            (
                1.0 / rank
                for rank, target_id in enumerate(ranking[:100], 1)
                if target_id in relevant
            ),
            0.0,
        )
        for ranking, relevant in zip(rankings, positives)
    )
    return metrics


def _metrics_by_dataset(
    examples: Sequence[TargetExample],
    rankings: Sequence[Sequence[str]],
    positives: Sequence[set[str]],
) -> dict[str, dict[str, float | int]]:
    return {
        dataset: _retrieval_metrics(
            [
                ranking
                for ranking, example in zip(rankings, examples)
                if example.dataset == dataset
            ],
            [
                relevant
                for relevant, example in zip(positives, examples)
                if example.dataset == dataset
            ],
        )
        for dataset in sorted({example.dataset for example in examples})
    }


def _device_features(
    teacher: TeacherJoinabilityModel,
    store: FeatureStore,
    object_id: str,
    device: torch.device,
) -> ObjectFeatures:
    return store.get(object_id, include_hidden=True).for_scoring(
        device,
        include_hidden=True,
        hidden_dtype=teacher.compute_dtype or next(teacher.parameters()).dtype,
    )


@torch.inference_mode()
def _teacher_scores(
    teacher: TeacherJoinabilityModel,
    query_id: str,
    candidate_ids: Sequence[str],
    store: FeatureStore,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    query = _device_features(teacher, store, query_id, device)
    compression_cache: dict[str, torch.Tensor] = {}
    values = []
    for start in range(0, len(candidate_ids), batch_size):
        batch_ids = candidate_ids[start : start + batch_size]
        candidates = [
            _device_features(teacher, store, candidate_id, device)
            for candidate_id in batch_ids
        ]
        scores = teacher.score_pairs(
            [query] * len(candidates),
            candidates,
            compression_cache=compression_cache,
        )
        values.extend(float(value) for value in scores.cpu())
    return values


def evaluate_teacher_reranking(
    teacher: TeacherJoinabilityModel,
    examples: Sequence[TargetExample],
    raw_hits_by_query: Sequence[Sequence[tuple[str, float]]],
    store: FeatureStore,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Rerank fixed raw candidates and return overall and per-dataset metrics."""

    if len(examples) != len(raw_hits_by_query) or batch_size <= 0:
        raise ValueError("Teacher rerank inputs must align and batch_size be positive")
    teacher.eval()
    raw_rankings = []
    teacher_rankings = []
    positive_sets = []
    correlations = []
    for example, raw_hits in progress(
        zip(examples, raw_hits_by_query),
        total=len(examples),
        desc="Teacher rerank",
        unit="query",
    ):
        candidate_ids = [target_id for target_id, _score in raw_hits]
        raw_scores = [float(score) for _target_id, score in raw_hits]
        teacher_scores = _teacher_scores(
            teacher,
            example.query_id,
            candidate_ids,
            store,
            device,
            batch_size,
        )
        raw_rankings.append(candidate_ids)
        teacher_rankings.append(
            [
                candidate_ids[index]
                for index in sorted(
                    range(len(candidate_ids)),
                    key=lambda index: (-teacher_scores[index], candidate_ids[index]),
                )
            ]
        )
        positive_sets.append(set(example.positive_target_ids))
        correlations.append(spearman_correlation(raw_scores, teacher_scores))

    raw_metrics = _retrieval_metrics(raw_rankings, positive_sets)
    raw_metrics["by_dataset"] = _metrics_by_dataset(
        examples, raw_rankings, positive_sets
    )
    teacher_metrics = _retrieval_metrics(teacher_rankings, positive_sets)
    teacher_metrics["by_dataset"] = _metrics_by_dataset(
        examples, teacher_rankings, positive_sets
    )
    return {
        "raw_direct": raw_metrics,
        "teacher_reranked": teacher_metrics,
        "recall@10_delta": (
            float(teacher_metrics["recall@10"])
            - float(raw_metrics["recall@10"])
        ),
        "spearman": {
            "mean": statistics.fmean(correlations),
            "median": statistics.median(correlations),
        },
    }
