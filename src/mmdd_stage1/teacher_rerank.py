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

RECALL_KS = (10, 20, 30, 40, 50)


def z_scores(values: Sequence[float]) -> list[float]:
    """Normalize one query's candidate scores without cross-query leakage."""

    if not values:
        return []
    mean = statistics.fmean(values)
    scale = math.sqrt(statistics.fmean((value - mean) ** 2 for value in values))
    if scale == 0:
        return [0.0] * len(values)
    return [(value - mean) / scale for value in values]


def ensemble_scores(
    raw_scores: Sequence[float],
    teacher_scores: Sequence[float],
    alpha: float,
) -> list[float]:
    """Combine per-query normalized raw and Teacher scores."""

    if len(raw_scores) != len(teacher_scores):
        raise ValueError("Raw and Teacher candidate scores must align")
    if not 0 <= alpha <= 1:
        raise ValueError("Teacher ensemble alpha must be in [0, 1]")
    normalized_raw = z_scores(raw_scores)
    normalized_teacher = z_scores(teacher_scores)
    return [
        alpha * teacher + (1.0 - alpha) * raw
        for raw, teacher in zip(normalized_raw, normalized_teacher)
    ]


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
    rankings: Sequence[Sequence[str]],
    positives: Sequence[set[str]],
    recall_ks: Sequence[int] = RECALL_KS,
    *,
    return_per_query: bool = False,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"queries": len(rankings)}
    recall_ks = tuple(sorted(dict.fromkeys(int(k) for k in recall_ks)))
    if not recall_ks:
        raise ValueError("recall_ks must not be empty")
    for k in recall_ks:
        metrics[f"recall@{k}"] = statistics.fmean(
            len(set(ranking[:k]) & relevant) / len(relevant)
            for ranking, relevant in zip(rankings, positives)
        )
    max_k = max(recall_ks)
    metrics[f"mrr@{max_k}"] = statistics.fmean(
        next(
            (
                1.0 / rank
                for rank, target_id in enumerate(ranking[:max_k], 1)
                if target_id in relevant
            ),
            0.0,
        )
        for ranking, relevant in zip(rankings, positives)
    )
    if return_per_query:
        metrics["per_query"] = {
            f"recall@{k}": [
                len(set(ranking[:k]) & relevant) / len(relevant)
                for ranking, relevant in zip(rankings, positives)
            ]
            for k in recall_ks
        }
    return metrics


def _metrics_by_dataset(
    examples: Sequence[TargetExample],
    rankings: Sequence[Sequence[str]],
    positives: Sequence[set[str]],
    recall_ks: Sequence[int] = RECALL_KS,
    *,
    return_per_query: bool = False,
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
            recall_ks,
            return_per_query=return_per_query,
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
    score_cache: dict[tuple[str, str], float] | None = None,
) -> list[float]:
    query = _device_features(teacher, store, query_id, device)
    compression_cache: dict[str, torch.Tensor] = {}
    computed_scores: dict[str, float] = {}
    missing_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if score_cache is None or (query_id, candidate_id) not in score_cache
    ]
    for start in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[start : start + batch_size]
        candidates = [
            _device_features(teacher, store, candidate_id, device)
            for candidate_id in batch_ids
        ]
        scores = teacher.score_pairs(
            [query] * len(candidates),
            candidates,
            compression_cache=compression_cache,
        )
        destination = score_cache if score_cache is not None else computed_scores
        destination.update(
            {
                (query_id, candidate_id)
                if score_cache is not None
                else candidate_id: float(score)
                for candidate_id, score in zip(batch_ids, scores.cpu())
            }
        )
    if score_cache is None:
        return [computed_scores[candidate_id] for candidate_id in candidate_ids]
    return [score_cache[(query_id, candidate_id)] for candidate_id in candidate_ids]


class TeacherRerankedANNIndices:
    """Rerank each fixed raw ANN edge pool with a frozen Teacher."""

    def __init__(
        self,
        raw_indices: Any,
        teacher: TeacherJoinabilityModel,
        store: FeatureStore,
        *,
        device: torch.device,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.raw_indices = raw_indices
        self.teacher = teacher.eval()
        self.store = store
        self.device = device
        self.batch_size = batch_size
        self.score_cache: dict[tuple[str, str], float] = {}

    def search(
        self, source_id: str, destination_type: str, k: int
    ) -> list[tuple[str, float]]:
        raw_hits = self.raw_indices.search(source_id, destination_type, k)
        candidate_ids = [object_id for object_id, _score in raw_hits]
        scores = _teacher_scores(
            self.teacher,
            source_id,
            candidate_ids,
            self.store,
            self.device,
            self.batch_size,
            self.score_cache,
        )
        order = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-scores[index], candidate_ids[index]),
        )
        return [(candidate_ids[index], scores[index]) for index in order]

    def search_many(
        self,
        source_ids: Sequence[str],
        destination_type: str,
        k: int,
    ) -> list[list[tuple[str, float]]]:
        return [
            self.search(source_id, destination_type, k)
            for source_id in source_ids
        ]


def evaluate_teacher_reranking(
    teacher: TeacherJoinabilityModel,
    examples: Sequence[TargetExample],
    raw_hits_by_query: Sequence[Sequence[tuple[str, float]]],
    store: FeatureStore,
    *,
    device: torch.device,
    batch_size: int,
    ensemble_alphas: Sequence[float] = (),
    recall_ks: Sequence[int] = RECALL_KS,
    return_per_query: bool = False,
    score_cache: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    """Rerank fixed raw candidates and return overall and per-dataset metrics."""

    if len(examples) != len(raw_hits_by_query) or batch_size <= 0:
        raise ValueError("Teacher rerank inputs must align and batch_size be positive")
    if any(not 0 <= alpha <= 1 for alpha in ensemble_alphas):
        raise ValueError("Teacher ensemble alphas must be in [0, 1]")
    alphas = list(dict.fromkeys(float(alpha) for alpha in ensemble_alphas))
    teacher.eval()
    raw_rankings = []
    teacher_rankings = []
    ensemble_rankings: dict[float, list[list[str]]] = {
        alpha: [] for alpha in alphas
    }
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
            score_cache,
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
        for alpha in alphas:
            combined = ensemble_scores(raw_scores, teacher_scores, alpha)
            ensemble_rankings[alpha].append(
                [
                    candidate_ids[index]
                    for index in sorted(
                        range(len(candidate_ids)),
                        key=lambda index: (-combined[index], candidate_ids[index]),
                    )
                ]
            )
        positive_sets.append(set(example.positive_target_ids))
        correlations.append(spearman_correlation(raw_scores, teacher_scores))

    raw_metrics = _retrieval_metrics(
        raw_rankings, positive_sets, recall_ks, return_per_query=return_per_query
    )
    raw_metrics["by_dataset"] = _metrics_by_dataset(
        examples, raw_rankings, positive_sets, recall_ks,
        return_per_query=return_per_query,
    )
    teacher_metrics = _retrieval_metrics(
        teacher_rankings, positive_sets, recall_ks, return_per_query=return_per_query
    )
    teacher_metrics["by_dataset"] = _metrics_by_dataset(
        examples, teacher_rankings, positive_sets, recall_ks,
        return_per_query=return_per_query,
    )
    ensembles = []
    for alpha in alphas:
        metrics = _retrieval_metrics(
            ensemble_rankings[alpha], positive_sets, recall_ks,
            return_per_query=return_per_query,
        )
        metrics["by_dataset"] = _metrics_by_dataset(
            examples, ensemble_rankings[alpha], positive_sets, recall_ks,
            return_per_query=return_per_query,
        )
        ensembles.append({"alpha": alpha, **metrics})
    delta_k = 10 if 10 in recall_ks else min(recall_ks)
    result = {
        "raw_direct": raw_metrics,
        "teacher_reranked": teacher_metrics,
        f"recall@{delta_k}_delta": (
            float(teacher_metrics[f"recall@{delta_k}"])
            - float(raw_metrics[f"recall@{delta_k}"])
        ),
        "spearman": {
            "mean": statistics.fmean(correlations),
            "median": statistics.median(correlations),
        },
        "ensembles": ensembles,
    }
    return result
