"""Full-corpus retrieval metrics used by Stage-1 checkpoint selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mmdd_progress import progress

from .data import TargetExample
from .retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    fuse_ranked_channels,
    retrieve_zero_one_hop_detailed,
)

RECALL_KS = (1, 5, 10, 50, 100)


def _recall(ranked_ids: Sequence[str], positives: set[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & positives) / len(positives)


def _reciprocal_rank(ranked_ids: Sequence[str], positives: set[str], k: int) -> float:
    return next(
        (
            1.0 / rank
            for rank, target_id in enumerate(ranked_ids[:k], 1)
            if target_id in positives
        ),
        0.0,
    )


def _channel_metrics(
    rankings: Sequence[Sequence[str]], positives: Sequence[set[str]]
) -> dict[str, float]:
    query_count = len(rankings)
    values = {
        f"recall@{k}": sum(
            _recall(ranking, relevant, k)
            for ranking, relevant in zip(rankings, positives)
        )
        / query_count
        for k in RECALL_KS
    }
    values["mrr@100"] = (
        sum(
            _reciprocal_rank(ranking, relevant, 100)
            for ranking, relevant in zip(rankings, positives)
        )
        / query_count
    )
    return values


def _retrieval_metrics(
    query_indices: Sequence[int],
    rankings: dict[str, list[list[str]]],
    positive_sets: Sequence[set[str]],
    positive_evidence_hits: dict[str, Sequence[bool]],
) -> dict[str, Any]:
    selected_positives = [positive_sets[index] for index in query_indices]
    selected_rankings = {
        channel: [values[index] for index in query_indices]
        for channel, values in rankings.items()
    }
    def fused_metrics(channel: str) -> dict[str, Any]:
        count = sum(positive_evidence_hits[channel][index] for index in query_indices)
        return {
            **_channel_metrics(selected_rankings[channel], selected_positives),
            "positive_evidence_path_queries@10": count,
            "positive_evidence_path_coverage@10": count / len(query_indices),
        }

    primary_fused = fused_metrics("fused")
    fused = _channel_metrics(selected_rankings["fused"], selected_positives)
    return {
        "queries": len(query_indices),
        **fused,
        "direct": _channel_metrics(selected_rankings["direct"], selected_positives),
        "evidence": _channel_metrics(
            selected_rankings["evidence"], selected_positives
        ),
        "fused_e0": fused_metrics("fused_e0"),
        "fused_e005": fused_metrics("fused_e005"),
        "positive_evidence_path_queries@10": primary_fused[
            "positive_evidence_path_queries@10"
        ],
        "positive_evidence_path_coverage@10": primary_fused[
            "positive_evidence_path_coverage@10"
        ],
    }


def evaluate_student_retrieval(
    examples: Sequence[TargetExample],
    indices: StudentANNIndices | RawEmbeddingANNIndices,
    *,
    direct_k: int = 100,
    evidence_k: int = 50,
    targets_per_evidence: int = 50,
    evidence_types: tuple[str, ...] = ("text", "image"),
    evidence_aggregation: str = "logsumexp",
    evidence_top_k: int = 4,
    rrf_k: int = 60,
    fusion_mode: str = "weighted_rrf",
    direct_weight: float = 1.0,
    evidence_weight: float = 0.05,
    gated_evidence_min_paths: int = 2,
    gated_evidence_quantile: float = 0.75,
    evidence_modality_weights: dict[str, float] | None = None,
    identity_baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate fixed queries against the shared full-corpus ANN indexes."""

    if not examples:
        raise ValueError("Retrieval evaluation requires at least one query")
    positive_sets: list[set[str]] = []
    rankings: dict[str, list[list[str]]] = {
        "fused": [],
        "direct": [],
        "evidence": [],
        "fused_e0": [],
        "fused_e005": [],
    }
    positive_evidence_hits: dict[str, list[bool]] = {
        "fused": [],
        "fused_e0": [],
        "fused_e005": [],
    }
    for example in progress(
        examples, desc="Retrieval evaluation", unit="query", leave=False
    ):
        positives = set(example.positive_target_ids)
        positive_sets.append(positives)
        result = retrieve_zero_one_hop_detailed(
            example.query_id,
            indices,
            direct_k=max(direct_k, max(RECALL_KS)),
            evidence_k=evidence_k,
            targets_per_evidence=targets_per_evidence,
            evidence_types=evidence_types,
            evidence_aggregation=evidence_aggregation,
            evidence_top_k=evidence_top_k,
            rrf_k=rrf_k,
            fusion_mode=fusion_mode,
            direct_weight=direct_weight,
            evidence_weight=evidence_weight,
            gated_evidence_min_paths=gated_evidence_min_paths,
            gated_evidence_quantile=gated_evidence_quantile,
            evidence_modality_weights=evidence_modality_weights,
        )
        result["fused_e0"] = fuse_ranked_channels(
            result["direct"],
            result["evidence"],
            rrf_k=rrf_k,
            fusion_mode="weighted_rrf",
            direct_weight=direct_weight,
            evidence_weight=0.0,
        )
        result["fused_e005"] = fuse_ranked_channels(
            result["direct"],
            result["evidence"],
            rrf_k=rrf_k,
            fusion_mode="weighted_rrf",
            direct_weight=direct_weight,
            evidence_weight=0.05,
        )
        for channel in rankings:
            rankings[channel].append(
                [str(item["target_id"]) for item in result[channel]]
            )

        positive_evidence = {
            candidate.target_id: set(candidate.evidence_ids)
            for candidate in example.candidates
            if candidate.target_id in positives and candidate.evidence_ids
        }
        for channel in positive_evidence_hits:
            positive_evidence_hits[channel].append(any(
                str(item["target_id"]) in positive_evidence
                and any(
                    path["kind"] == "evidence"
                    and str(path["evidence_id"])
                    in positive_evidence[str(item["target_id"])]
                    for path in item["paths"]
                )
                for item in result[channel][:10]
            ))

    metrics = _retrieval_metrics(
        list(range(len(examples))), rankings, positive_sets, positive_evidence_hits
    )
    datasets = sorted({example.dataset for example in examples})
    metrics["by_dataset"] = {
        dataset: _retrieval_metrics(
            [
                index
                for index, example in enumerate(examples)
                if example.dataset == dataset
            ],
            rankings,
            positive_sets,
            positive_evidence_hits,
        )
        for dataset in datasets
    }
    if identity_baseline_metrics is not None:
        metrics["evidence_identity_baseline"] = {
            **identity_baseline_metrics["evidence"],
            "by_dataset": {
                dataset: values["evidence"]
                for dataset, values in identity_baseline_metrics[
                    "by_dataset"
                ].items()
            },
        }
    return metrics


def evaluate_direct_retrieval(
    examples: Sequence[TargetExample],
    indices: StudentANNIndices | RawEmbeddingANNIndices,
    *,
    direct_k: int = 100,
) -> dict[str, float | int]:
    """Evaluate only direct Q-to-table retrieval with the Stage-1 metric contract."""

    if not examples:
        raise ValueError("Retrieval evaluation requires at least one query")
    rankings = []
    positives = []
    for example in progress(
        examples, desc="Direct retrieval evaluation", unit="query", leave=False
    ):
        ranked = sorted(
            indices.search(example.query_id, "table", max(direct_k, max(RECALL_KS))),
            key=lambda item: (-item[1], item[0]),
        )
        rankings.append([target_id for target_id, _score in ranked])
        positives.append(set(example.positive_target_ids))
    return {"queries": len(examples), **_channel_metrics(rankings, positives)}
