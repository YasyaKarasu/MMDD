"""Full-corpus retrieval metrics used by Stage-1 checkpoint selection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
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


def retrieval_metric_values(
    rankings: Sequence[Sequence[str]],
    positives: Sequence[set[str]],
    recall_ks: Sequence[int],
) -> dict[str, list[float]]:
    """Return per-query recall@k and mrr@max(k) values."""

    values = {
        f"recall@{k}": [
            _recall(ranking, relevant, k)
            for ranking, relevant in zip(rankings, positives)
        ]
        for k in recall_ks
    }
    max_k = max(recall_ks)
    values[f"mrr@{max_k}"] = [
        _reciprocal_rank(ranking, relevant, max_k)
        for ranking, relevant in zip(rankings, positives)
    ]
    return values


def _channel_metrics(
    rankings_by_k: dict[int, Sequence[Sequence[str]]],
    positives: Sequence[set[str]],
    recall_ks: tuple[int, ...],
) -> dict[str, float]:
    query_count = len(positives)
    per_query = {
        f"recall@{k}": retrieval_metric_values(
            rankings_by_k[k], positives, (k,)
        )[f"recall@{k}"]
        for k in recall_ks
    }
    max_k = max(recall_ks)
    per_query[f"mrr@{max_k}"] = retrieval_metric_values(
        rankings_by_k[max_k], positives, (max_k,)
    )[f"mrr@{max_k}"]
    return {
        metric: sum(metric_values) / query_count
        for metric, metric_values in per_query.items()
    }


def _retrieval_metrics(
    query_indices: Sequence[int],
    rankings: dict[str, dict[int, list[list[str]]]],
    positive_sets: Sequence[set[str]],
    positive_evidence_hits: dict[str, dict[int, list[bool]]],
    valid_path_counts: dict[int, list[tuple[int, int]]],
    recall_ks: tuple[int, ...],
    evidence_bundle_budget: int,
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
    valid_path = {}
    for k in recall_ks:
        numerator = sum(valid_path_counts[k][index][0] for index in query_indices)
        denominator = sum(valid_path_counts[k][index][1] for index in query_indices)
        name = f"valid_path_recall@{k},{evidence_bundle_budget}"
        valid_path[name] = numerator / denominator if denominator else 0.0
        valid_path[f"supported_positive_pairs@{k},{evidence_bundle_budget}"] = denominator
    result: dict[str, Any] = {
        "queries": len(query_indices),
        **_channel_metrics(selected_rankings["fused"], selected_positives, recall_ks),
        "direct": _channel_metrics(selected_rankings["direct"], selected_positives, recall_ks),
        "evidence": _channel_metrics(selected_rankings["evidence"], selected_positives, recall_ks),
        "fused_e0": fused_metrics("fused_e0"),
        "fused_e005": fused_metrics("fused_e005"),
        "positive_evidence_path_queries@10": primary_fused["positive_evidence_path_queries@10"],
        "positive_evidence_path_coverage@10": primary_fused["positive_evidence_path_coverage@10"],
        **valid_path,
    }
    if return_per_query:
        result["per_query"] = {
            channel: {
                f"recall@{k}": retrieval_metric_values(
                    selected_rankings[channel][k], selected_positives, (k,)
                )[f"recall@{k}"]
                for k in recall_ks
            }
            for channel in rankings
        }
    return result


def evidence_funnel_metrics(
    examples: Sequence[TargetExample],
    detailed_results: Sequence[dict[str, list[dict[str, Any]]]],
    *,
    evidence_bundle_budget: int,
) -> dict[str, Any]:
    """Measure evidence arrival and retention before final target admission."""

    pair_rows = []
    for example, result in zip(examples, detailed_results):
        if example.query_kind not in {None, "implicit"}:
            continue
        positive_evidence = example.positive_evidence_by_target or {
            candidate.target_id: candidate.evidence_ids
            for candidate in example.candidates
            if candidate.target_id in set(example.positive_target_ids)
            and candidate.evidence_ids
        }
        rows_by_target = example.positive_evidence_rows_by_target or {}
        evidence_by_target = {
            str(row["target_id"]): row for row in result["evidence"]
        }
        all_query_evidence = {
            str(path["evidence_id"])
            for row in result["evidence"]
            for path in row["paths"]
            if path["kind"] == "evidence"
        }
        for target_id, valid_values in positive_evidence.items():
            valid_evidence = set(valid_values)
            target = evidence_by_target.get(str(target_id))
            target_paths = (
                [path for path in target["paths"] if path["kind"] == "evidence"]
                if target is not None
                else []
            )
            pool_evidence = {
                str(path["evidence_id"]) for path in target_paths
            }
            if target is not None and "selected_evidence_ids" in target:
                selected = [
                    str(value)
                    for value in target["selected_evidence_ids"][
                        :evidence_bundle_budget
                    ]
                ]
            else:
                selected = [
                    str(path["evidence_id"])
                    for path in sorted(
                        target_paths,
                        key=lambda path: (
                            -float(path["path_score"]),
                            str(path["evidence_id"]),
                        ),
                    )[:evidence_bundle_budget]
                ]
            valid_selected = valid_evidence & set(selected)
            supported_rows = set()
            for evidence_id in valid_selected:
                supported_rows.update(
                    rows_by_target.get(str(target_id), {}).get(evidence_id, ())
                )
            row_denominator = example.query_row_count or 0
            pair_rows.append(
                {
                    "query_id": example.query_id,
                    "target_id": str(target_id),
                    "valid_pool": int(bool(valid_evidence & pool_evidence)),
                    "valid_b": int(bool(valid_selected)),
                    "row_b_numerator": len(supported_rows),
                    "row_b_denominator": row_denominator,
                    "q_to_e": int(bool(valid_evidence & all_query_evidence)),
                    "e_to_t_given_q_to_e": int(
                        bool(valid_evidence & pool_evidence)
                    ),
                }
            )
    pair_count = len(pair_rows)
    q_to_e_count = sum(row["q_to_e"] for row in pair_rows)
    return {
        "implicit_positive_pairs": pair_count,
        "valid_pool_count": sum(row["valid_pool"] for row in pair_rows),
        "valid_pool": (
            sum(row["valid_pool"] for row in pair_rows) / pair_count
            if pair_count
            else 0.0
        ),
        "valid_b_count": sum(row["valid_b"] for row in pair_rows),
        "valid_b": (
            sum(row["valid_b"] for row in pair_rows) / pair_count
            if pair_count
            else 0.0
        ),
        "row_b": (
            sum(
                row["row_b_numerator"] / row["row_b_denominator"]
                if row["row_b_denominator"]
                else 0.0
                for row in pair_rows
            )
            / pair_count
            if pair_count
            else 0.0
        ),
        "q_to_e_pair_recall": q_to_e_count / pair_count if pair_count else 0.0,
        "e_to_t_pair_recall_given_q_to_e": (
            sum(row["e_to_t_given_q_to_e"] for row in pair_rows)
            / q_to_e_count
            if q_to_e_count
            else 0.0
        ),
        "per_pair": pair_rows,
    }


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
    path_combination: str = "sum",
    evidence_threshold: float = 0.0,
    evidence_target_temperature: float = 1.0,
    row_support_model: str | Path | None = None,
    row_support_model_sha256: str | None = None,
    row_support_top_l: int = 20,
    evidence_content_keys: str | Path | None = None,
    evidence_content_keys_sha256: str | None = None,
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
    valid_path_counts: dict[int, list[tuple[int, int]]] = {
        requested_k: [] for requested_k in retrieval_ks
    }
    detailed_results_by_k: dict[
        int, list[dict[str, list[dict[str, Any]]]]
    ] = {}

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
            path_combination=path_combination,
            evidence_threshold=evidence_threshold,
            evidence_target_temperature=evidence_target_temperature,
            row_support_model=row_support_model,
            row_support_model_sha256=row_support_model_sha256,
            row_support_top_l=row_support_top_l,
            evidence_content_keys=evidence_content_keys,
            evidence_content_keys_sha256=evidence_content_keys_sha256,
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
        detailed_results_by_k[requested_k] = results
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
            valid_numerator = 0
            valid_denominator = len(positive_evidence)
            fused_by_target = {
                str(item["target_id"]): item for item in result["fused"][:requested_k]
            }
            for target_id, valid_evidence_ids in positive_evidence.items():
                item = fused_by_target.get(target_id)
                if item is None:
                    continue
                if "selected_evidence_ids" in item:
                    selected_evidence_ids = set(
                        str(value)
                        for value in item["selected_evidence_ids"][:evidence_top_k]
                    )
                else:
                    selected_paths = sorted(
                        (
                            path
                            for path in item["paths"]
                            if path["kind"] == "evidence"
                        ),
                        key=lambda path: (
                            -float(path["path_score"]),
                            str(path["evidence_id"]),
                        ),
                    )[:evidence_top_k]
                    selected_evidence_ids = {
                        str(path["evidence_id"]) for path in selected_paths
                    }
                valid_numerator += bool(
                    selected_evidence_ids & valid_evidence_ids
                )
            valid_path_counts[requested_k].append(
                (valid_numerator, valid_denominator)
            )
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
        positive_evidence_hits, valid_path_counts, recall_ks,
        evidence_top_k, return_per_query=return_per_query,
    )
    metrics["evidence_funnel"] = evidence_funnel_metrics(
        examples,
        detailed_results_by_k[10],
        evidence_bundle_budget=evidence_top_k,
    )
    datasets = sorted({example.dataset for example in examples})
    metrics["by_dataset"] = {
        dataset: _retrieval_metrics(
            [index for index, example in enumerate(examples) if example.dataset == dataset],
            rankings, positive_sets, positive_evidence_hits, valid_path_counts,
            recall_ks, evidence_top_k,
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
