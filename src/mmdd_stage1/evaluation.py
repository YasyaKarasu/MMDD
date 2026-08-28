"""Full-corpus retrieval metrics used by Stage-1 checkpoint selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mmdd_progress import progress

from .data import TargetExample
from .retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    retrieve_zero_one_hop_detailed,
)

RECALL_KS = (1, 5, 10, 50, 100)


def _recall(ranked_ids: Sequence[str], positives: set[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & positives) / len(positives)


def _reciprocal_rank(ranked_ids: Sequence[str], positives: set[str], k: int) -> float:
    return next(
        (1.0 / rank for rank, target_id in enumerate(ranked_ids[:k], 1) if target_id in positives),
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
    values["mrr@100"] = sum(
        _reciprocal_rank(ranking, relevant, 100)
        for ranking, relevant in zip(rankings, positives)
    ) / query_count
    return values


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
) -> dict[str, Any]:
    """Evaluate fixed queries against the shared full-corpus ANN indexes."""

    if not examples:
        raise ValueError("Retrieval evaluation requires at least one query")
    positive_sets: list[set[str]] = []
    rankings: dict[str, list[list[str]]] = {
        "fused": [],
        "direct": [],
        "evidence": [],
    }
    positive_evidence_path_queries = 0
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
        if any(
            str(item["target_id"]) in positive_evidence
            and any(
                path["kind"] == "evidence"
                and str(path["evidence_id"])
                in positive_evidence[str(item["target_id"])]
                for path in item["paths"]
            )
            for item in result["fused"][:10]
        ):
            positive_evidence_path_queries += 1

    fused = _channel_metrics(rankings["fused"], positive_sets)
    return {
        "queries": len(examples),
        **fused,
        "direct": _channel_metrics(rankings["direct"], positive_sets),
        "evidence": _channel_metrics(rankings["evidence"], positive_sets),
        "positive_evidence_path_queries@10": positive_evidence_path_queries,
        "positive_evidence_path_coverage@10": (
            positive_evidence_path_queries / len(examples)
        ),
    }
