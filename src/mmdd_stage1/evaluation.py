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
    retrieve_zero_one_hop_detailed_many,
)

DEFAULT_RECALL_KS = (10, 20, 30, 40, 50)


def _validate_recall_ks(recall_ks: Sequence[int]) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value) for value in recall_ks))
    if not values or any(value <= 0 for value in values):
        raise ValueError("recall_ks must contain positive integers")
    return tuple(sorted(values))


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
    rankings_by_k: dict[int, Sequence[Sequence[str]]],
    positives: Sequence[set[str]],
    recall_ks: tuple[int, ...],
) -> dict[str, float]:
    query_count = len(positives)
    values = {
        f"recall@{k}": sum(
            _recall(ranking, relevant, k)
            for ranking, relevant in zip(rankings_by_k[k], positives)
        )
        / query_count
        for k in recall_ks
    }
    max_k = max(recall_ks)
    values[f"mrr@{max_k}"] = sum(
        _reciprocal_rank(ranking, relevant, max_k)
        for ranking, relevant in zip(rankings_by_k[max_k], positives)
    ) / query_count
    return values


def _retrieval_metrics(
    query_indices: Sequence[int],
    rankings: dict[str, dict[int, list[list[str]]]],
    positive_sets: Sequence[set[str]],
    positive_evidence_hits: dict[str, dict[int, list[bool]]],
    recall_ks: tuple[int, ...],
    *,
    return_per_query: bool = False,
) -> dict[str, Any]:
    selected_positives = [positive_sets[index] for index in query_indices]
    selected_rankings = {
        channel: {
            k: [values[index] for index in query_indices]
            for k, values in channel_values.items()
        }
        for channel, channel_values in rankings.items()
    }
    coverage_k = 10

    def fused_metrics(channel: str) -> dict[str, Any]:
        count = sum(
            positive_evidence_hits[channel][coverage_k][index]
            for index in query_indices
        )
        values = _channel_metrics(
            selected_rankings[channel], selected_positives, recall_ks
        )
        values.update(
            {
                "positive_evidence_path_queries@10": count,
                "positive_evidence_path_coverage@10": count / len(query_indices),
            }
        )
        return values

    primary_fused = fused_metrics("fused")
    result: dict[str, Any] = {
        "queries": len(query_indices),
        **_channel_metrics(selected_rankings["fused"], selected_positives, recall_ks),
        "direct": _channel_metrics(selected_rankings["direct"], selected_positives, recall_ks),
        "evidence": _channel_metrics(selected_rankings["evidence"], selected_positives, recall_ks),
        "fused_e0": fused_metrics("fused_e0"),
        "fused_e005": fused_metrics("fused_e005"),
        "positive_evidence_path_queries@10": primary_fused["positive_evidence_path_queries@10"],
        "positive_evidence_path_coverage@10": primary_fused["positive_evidence_path_coverage@10"],
    }
    if return_per_query:
        result["per_query"] = {
            channel: {
                f"recall@{k}": [
                    _recall(ranking, relevant, k)
                    for ranking, relevant in zip(selected_rankings[channel][k], selected_positives)
                ]
                for k in recall_ks
            }
            for channel in rankings
        }
    return result


def evaluate_student_retrieval(
    examples: Sequence[TargetExample],
    indices: StudentANNIndices | RawEmbeddingANNIndices,
    *,
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS,
    k: int | None = None,
    gamma: int = 4,
    gamma_evidence: int = 2,
    direct_k: int | None = None,
    evidence_k: int | None = None,
    targets_per_evidence: int | None = None,
    evidence_types: tuple[str, ...] = ("text", "image"),
    evidence_aggregation: str = "logsumexp",
    evidence_top_k: int = 4,
    evidence_temperature: float = 1.0,
    evidence_power: float = 2.0,
    path_edge_normalization: str = "none",
    rrf_k: int = 60,
    fusion_mode: str = "weighted_rrf",
    direct_weight: float = 1.0,
    evidence_weight: float = 0.05,
    fusion_score_normalization: str = "none",
    fusion_score_temperature: float = 1.0,
    gated_evidence_min_paths: int = 2,
    gated_evidence_quantile: float = 0.75,
    evidence_modality_weights: dict[str, float] | None = None,
    identity_baseline_metrics: dict[str, Any] | None = None,
    return_per_query: bool = False,
) -> dict[str, Any]:
    """Evaluate each requested k with an independent gamma-derived retrieval pool."""

    if not examples:
        raise ValueError("Retrieval evaluation requires at least one query")
    recall_ks = _validate_recall_ks(recall_ks)
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    if gamma <= 0 or gamma_evidence <= 0:
        raise ValueError("gamma and gamma_evidence must be positive")
    if k is not None:
        recall_ks = (int(k),)
    retrieval_ks = tuple(sorted({*recall_ks, 10}))
    positive_sets = [set(example.positive_target_ids) for example in examples]
    rankings: dict[str, dict[int, list[list[str]]]] = {
        channel: {requested_k: [] for requested_k in retrieval_ks}
        for channel in ("fused", "direct", "evidence", "fused_e0", "fused_e005")
    }
    positive_evidence_hits: dict[str, dict[int, list[bool]]] = {
        channel: {requested_k: [] for requested_k in retrieval_ks}
        for channel in ("fused", "fused_e0", "fused_e005")
    }

    for requested_k in retrieval_ks:
        query_ids = [
            example.query_id
            for example in progress(
                examples,
                desc=f"Prepare retrieval evaluation k={requested_k}",
                unit="query",
                leave=False,
            )
        ]
        results = retrieve_zero_one_hop_detailed_many(
            query_ids,
            indices,
            k=requested_k,
            gamma=gamma,
            gamma_evidence=gamma_evidence,
            direct_k=direct_k,
            evidence_k=evidence_k,
            targets_per_evidence=targets_per_evidence,
            evidence_types=evidence_types,
            evidence_aggregation=evidence_aggregation,
            evidence_top_k=evidence_top_k,
            evidence_temperature=evidence_temperature,
            evidence_power=evidence_power,
            path_edge_normalization=path_edge_normalization,
            rrf_k=rrf_k,
            fusion_mode=fusion_mode,
            direct_weight=direct_weight,
            evidence_weight=evidence_weight,
            fusion_score_normalization=fusion_score_normalization,
            fusion_score_temperature=fusion_score_temperature,
            gated_evidence_min_paths=gated_evidence_min_paths,
            gated_evidence_quantile=gated_evidence_quantile,
            evidence_modality_weights=evidence_modality_weights,
        )
        for example, result in zip(
            examples,
            results,
        ):
            result["fused_e0"] = fuse_ranked_channels(
                result["direct"], result["evidence"], rrf_k=rrf_k,
                fusion_mode="weighted_rrf", direct_weight=direct_weight,
                evidence_weight=0.0,
            )
            result["fused_e005"] = fuse_ranked_channels(
                result["direct"], result["evidence"], rrf_k=rrf_k,
                fusion_mode="weighted_rrf", direct_weight=direct_weight,
                evidence_weight=0.05,
            )
            for channel in rankings:
                rankings[channel][requested_k].append(
                    [str(item["target_id"]) for item in result[channel]]
                )
            positives = set(example.positive_target_ids)
            positive_evidence = {
                candidate.target_id: set(candidate.evidence_ids)
                for candidate in example.candidates
                if candidate.target_id in positives and candidate.evidence_ids
            }
            for channel in positive_evidence_hits:
                positive_evidence_hits[channel][requested_k].append(
                    any(
                        str(item["target_id"]) in positive_evidence
                        and any(
                            path["kind"] == "evidence"
                            and str(path["evidence_id"]) in positive_evidence[str(item["target_id"])]
                            for path in item["paths"]
                        )
                        for item in result[channel][:10]
                    )
                )

    metrics = _retrieval_metrics(
        list(range(len(examples))), rankings, positive_sets,
        positive_evidence_hits, recall_ks, return_per_query=return_per_query,
    )
    datasets = sorted({example.dataset for example in examples})
    metrics["by_dataset"] = {
        dataset: _retrieval_metrics(
            [index for index, example in enumerate(examples) if example.dataset == dataset],
            rankings, positive_sets, positive_evidence_hits, recall_ks,
            return_per_query=return_per_query,
        )
        for dataset in datasets
    }
    metrics["retrieval_budget"] = {
        "recall_ks": list(recall_ks),
        "coverage_k": 10,
        "gamma": gamma,
        "gamma_evidence": gamma_evidence,
        "direct_k_override": direct_k,
        "evidence_k_override": evidence_k,
        "targets_per_evidence_override": targets_per_evidence,
        "per_k": {
            str(requested_k): {
                "direct_k": direct_k if direct_k is not None else gamma * requested_k,
                "evidence_k": evidence_k if evidence_k is not None else gamma_evidence * requested_k,
                "targets_per_evidence": targets_per_evidence if targets_per_evidence is not None else gamma_evidence * requested_k,
            }
            for requested_k in retrieval_ks
        },
    }
    if return_per_query:
        metrics["per_query_by_dataset"] = {
            dataset: metrics["by_dataset"][dataset].get("per_query", {})
            for dataset in datasets
        }
    if identity_baseline_metrics is not None:
        metrics["evidence_identity_baseline"] = {
            **identity_baseline_metrics["evidence"],
            "by_dataset": {
                dataset: values["evidence"]
                for dataset, values in identity_baseline_metrics["by_dataset"].items()
            },
        }
    return metrics


def evaluate_direct_retrieval(
    examples: Sequence[TargetExample],
    indices: StudentANNIndices | RawEmbeddingANNIndices,
    *,
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS,
    k: int | None = None,
    direct_k: int | None = None,
) -> dict[str, float | int]:
    """Evaluate only direct Q-to-table retrieval with independent requested k pools."""

    if not examples:
        raise ValueError("Retrieval evaluation requires at least one query")
    recall_ks = _validate_recall_ks(recall_ks)
    if k is not None:
        recall_ks = (int(k),)
    rankings = {requested_k: [] for requested_k in recall_ks}
    positives = [set(example.positive_target_ids) for example in examples]
    for requested_k in recall_ks:
        budget = direct_k if direct_k is not None else requested_k
        for example in progress(
            examples, desc=f"Direct retrieval evaluation k={requested_k}", unit="query", leave=False
        ):
            ranked = sorted(
                indices.search(example.query_id, "table", max(budget, requested_k)),
                key=lambda item: (-item[1], item[0]),
            )
            rankings[requested_k].append([target_id for target_id, _score in ranked])
    return {"queries": len(examples), **_channel_metrics(rankings, positives, recall_ks)}
