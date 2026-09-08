"""Auditable ranking and confidence metrics for directed Stage-1 edges."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .data import EdgeExample
from .features import normalize_object_type


def _relation(example: EdgeExample) -> str:
    if example.source_type is None or example.destination_type is None:
        return "unspecified"
    return (
        f"{normalize_object_type(example.source_type)}_to_"
        f"{normalize_object_type(example.destination_type)}"
    )


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _score, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _average_precision(
    scores: Sequence[float], labels: Sequence[int]
) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    seen = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = ordered[index:end]
        true_positives += sum(label for _score, label in group)
        seen += len(group)
        recall = true_positives / positives
        precision = true_positives / seen
        result += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return result


def _reliability_bins(
    confidence: Sequence[float], labels: Sequence[int], bins: int
) -> list[dict[str, float | int]]:
    counts = [0] * bins
    confidence_sums = [0.0] * bins
    positive_sums = [0] * bins
    for value, label in zip(confidence, labels):
        index = min(int(value * bins), bins - 1)
        counts[index] += 1
        confidence_sums[index] += value
        positive_sums[index] += label
    return [
        {
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": count,
            "mean_confidence": confidence_sums[index] / count,
            "positive_rate": positive_sums[index] / count,
        }
        for index, count in enumerate(counts)
        if count
    ]


def _binary_metrics(
    scores: Sequence[float],
    confidence: Sequence[float],
    labels: Sequence[int],
    *,
    reliability_bins: int,
) -> dict[str, Any]:
    if not (len(scores) == len(confidence) == len(labels)):
        raise ValueError("Binary edge scores, confidence, and labels must align")
    if reliability_bins <= 0:
        raise ValueError("reliability_bins must be positive")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("Confirmed edge labels must be binary")
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("Edge ranking scores must be finite")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidence):
        raise ValueError("Edge confidence must be finite and within [0, 1]")

    positives = sum(labels)
    negatives = len(labels) - positives
    if not labels:
        return {
            "confirmed": 0,
            "positive": 0,
            "negative": 0,
            "positive_rate": None,
            "auroc": None,
            "auprc": None,
            "brier": None,
            "nll": None,
            "confidence_quantiles": {},
            "positive_confidence_quantiles": {},
            "negative_confidence_quantiles": {},
            "reliability_bins": [],
        }

    clipped = [min(max(value, 1e-7), 1.0 - 1e-7) for value in confidence]
    positive_confidence = [
        value for value, label in zip(confidence, labels) if label == 1
    ]
    negative_confidence = [
        value for value, label in zip(confidence, labels) if label == 0
    ]
    probabilities = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

    def quantiles(values: Sequence[float]) -> dict[str, float | None]:
        return {
            f"p{round(probability * 100):02d}": _quantile(values, probability)
            for probability in probabilities
        }

    return {
        "confirmed": len(labels),
        "positive": positives,
        "negative": negatives,
        "positive_rate": positives / len(labels),
        "auroc": _auroc(scores, labels),
        "auprc": _average_precision(scores, labels),
        "brier": sum(
            (value - label) ** 2 for value, label in zip(confidence, labels)
        )
        / len(labels),
        "nll": -sum(
            label * math.log(value) + (1 - label) * math.log(1.0 - value)
            for value, label in zip(clipped, labels)
        )
        / len(labels),
        "confidence_quantiles": quantiles(confidence),
        "positive_confidence_quantiles": quantiles(positive_confidence),
        "negative_confidence_quantiles": quantiles(negative_confidence),
        "reliability_bins": _reliability_bins(
            confidence, labels, reliability_bins
        ),
    }


def summarize_edge_quality(
    examples: Sequence[EdgeExample],
    ranking_scores: Sequence[Sequence[float]],
    confidence_scores: Sequence[Sequence[float]],
    *,
    recall_ks: Sequence[int] = (1, 5, 10, 20),
    reliability_bins: int = 10,
) -> dict[str, Any]:
    """Summarize multi-positive list recall and confirmed-label quality."""

    if not (
        len(examples) == len(ranking_scores) == len(confidence_scores)
    ):
        raise ValueError("Edge examples and score rows must align")
    cutoffs = tuple(dict.fromkeys(int(value) for value in recall_ks))
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError("recall_ks must contain positive integers")

    groups: dict[str, list[int]] = defaultdict(list)
    groups["overall"] = list(range(len(examples)))
    for index, example in enumerate(examples):
        groups[_relation(example)].append(index)

    result: dict[str, Any] = {}
    for relation, indices in sorted(groups.items()):
        recall_values = {cutoff: [] for cutoff in cutoffs}
        binary_scores: list[float] = []
        binary_confidence: list[float] = []
        binary_labels: list[int] = []
        unknown = 0
        candidates = 0
        for index in indices:
            example = examples[index]
            scores = [float(value) for value in ranking_scores[index]]
            confidence = [float(value) for value in confidence_scores[index]]
            if len(scores) != len(example.candidate_ids) or len(confidence) != len(
                example.candidate_ids
            ):
                raise ValueError("Edge score row does not match its candidate list")
            if any(not math.isfinite(value) for value in scores):
                raise ValueError("Edge ranking scores must be finite")
            positive_ids = set(example.positive_ids) or {
                example.candidate_ids[example.positive_index]
            }
            ranking = sorted(
                range(len(scores)), key=lambda candidate: (-scores[candidate], candidate)
            )
            for cutoff in cutoffs:
                retrieved = {
                    example.candidate_ids[candidate]
                    for candidate in ranking[:cutoff]
                }
                recall_values[cutoff].append(
                    len(retrieved & positive_ids) / len(positive_ids)
                )

            labels = example.confirmed_labels or (None,) * len(scores)
            if len(labels) != len(scores):
                raise ValueError("Confirmed labels do not match edge candidates")
            candidates += len(scores)
            for score, probability, label in zip(scores, confidence, labels):
                if label is None:
                    unknown += 1
                    continue
                binary_scores.append(score)
                binary_confidence.append(probability)
                binary_labels.append(label)

        result[relation] = {
            "lists": len(indices),
            "candidates": candidates,
            "unknown": unknown,
            "ranking": {
                f"recall@{cutoff}": (
                    sum(recall_values[cutoff]) / len(recall_values[cutoff])
                    if recall_values[cutoff]
                    else None
                )
                for cutoff in cutoffs
            },
            "confirmed_quality": _binary_metrics(
                binary_scores,
                binary_confidence,
                binary_labels,
                reliability_bins=reliability_bins,
            ),
        }
    return {
        "recall_ks": list(cutoffs),
        "reliability_bin_count": reliability_bins,
        "overall": result.pop("overall"),
        "by_relation": result,
    }
