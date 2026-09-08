"""Per-query metric bookkeeping shared by the Stage-1 configuration sweeps.

These helpers implement the per-query fused/direct/evidence recall, MRR and
positive-evidence coverage protocol used by the r6/r7/r8 sweep entrypoints,
together with the record aggregation and paired-bootstrap deltas reported
against a named baseline configuration.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .data import TargetExample
from .objectives import PathAggregator
from .retrieval import rank_detailed_paths, retrieve_zero_one_hop_detailed_many
from .significance import paired_bootstrap_delta


def path_pool(result: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_target = {}
    for row in [*result["direct"], *result["evidence"]]:
        by_target[str(row["target_id"])] = row["paths"]
    return by_target


def _positive_evidence(example: TargetExample) -> dict[str, set[str]]:
    positives = set(example.positive_target_ids)
    return {
        candidate.target_id: set(candidate.evidence_ids)
        for candidate in example.candidates
        if candidate.target_id in positives and candidate.evidence_ids
    }


def query_values(
    result: dict[str, list[dict[str, Any]]], example: TargetExample, k: int
) -> dict[str, float]:
    positives = set(example.positive_target_ids)

    def recall(channel: str) -> float:
        ids = {str(row["target_id"]) for row in result[channel][:k]}
        return len(ids & positives) / len(positives)

    fused = result["fused"]
    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, row in enumerate(fused[:k], 1)
            if str(row["target_id"]) in positives
        ),
        0.0,
    )
    gold_evidence = _positive_evidence(example)
    coverage = any(
        str(row["target_id"]) in gold_evidence
        and any(
            path["kind"] == "evidence"
            and str(path["evidence_id"]) in gold_evidence[str(row["target_id"])]
            for path in row["paths"]
        )
        for row in fused[:10]
    )
    return {
        "fused_recall": recall("fused"),
        "direct_recall": recall("direct"),
        "evidence_recall": recall("evidence"),
        "reciprocal_rank": reciprocal_rank,
        "coverage": float(coverage),
    }


def append_values(
    records: dict[str, dict[int, dict[str, list[float]]]],
    config: str,
    k: int,
    values: dict[str, float],
) -> None:
    for name, value in values.items():
        records[config][k][name].append(value)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def evaluate_retrieval_configurations(
    examples: Sequence[TargetExample],
    indices: Any,
    *,
    fusion_configs: Sequence[dict[str, Any]],
    aggregation_configs: Sequence[dict[str, Any]],
    recall_ks: Sequence[int],
    query_batch_size: int,
) -> tuple[dict[str, dict[int, dict[str, list[float]]]], dict[int, Any]]:
    """Evaluate every fusion×aggregation combination on one fixed path pool
    per query and k, returning the per-query records and the pool hashes."""

    records: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    pool_hashes = {k: hashlib.sha256() for k in recall_ks}
    for k in recall_ks:
        for start in range(0, len(examples), query_batch_size):
            batch = examples[start : start + query_batch_size]
            detailed = retrieve_zero_one_hop_detailed_many(
                [example.query_id for example in batch],
                indices,
                k=k,
                gamma=10,
                gamma_evidence=2,
                evidence_types=("text", "image"),
                evidence_aggregation="logsumexp",
                evidence_top_k=4,
                fusion_mode="weighted_rrf",
                direct_weight=1.0,
                evidence_weight=0.05,
                query_batch_size=query_batch_size,
            )
            for example, baseline in zip(batch, detailed):
                paths = path_pool(baseline)
                pool_hashes[k].update(example.query_id.encode("utf-8"))
                for target_id in sorted(paths):
                    pool_hashes[k].update(b"\0")
                    pool_hashes[k].update(target_id.encode("utf-8"))
                for aggregation in aggregation_configs:
                    for fusion in fusion_configs:
                        name = f"{fusion['name']}__{aggregation['name']}"
                        variant = rank_detailed_paths(
                            paths,
                            aggregator=PathAggregator(
                                aggregation["aggregation"], 4
                            ),
                            path_edge_normalization=aggregation["normalization"],
                            rrf_k=60,
                            fusion_mode=fusion["mode"],
                            direct_weight=1.0,
                            evidence_weight=fusion["evidence_weight"],
                            fusion_score_normalization=fusion["normalization"],
                            fusion_score_temperature=1.0,
                            gated_evidence_min_paths=2,
                            gated_evidence_quantile=0.75,
                        )
                        append_values(
                            records,
                            name,
                            k,
                            query_values(variant, example, k),
                        )
    return records, pool_hashes


def finalize_records(
    records: dict[str, dict[int, dict[str, list[float]]]],
    baseline: str,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    output = {}
    for config, by_k in records.items():
        metrics: dict[str, Any] = {"queries": len(by_k[min(by_k)]["fused_recall"])}
        for k, values in sorted(by_k.items()):
            metrics[f"recall@{k}"] = mean(values["fused_recall"])
            metrics.setdefault("direct", {})[f"recall@{k}"] = mean(
                values["direct_recall"]
            )
            metrics.setdefault("evidence", {})[f"recall@{k}"] = mean(
                values["evidence_recall"]
            )
        max_k = max(by_k)
        metrics[f"mrr@{max_k}"] = mean(by_k[max_k]["reciprocal_rank"])
        if 10 in by_k:
            metrics["positive_evidence_path_coverage@10"] = mean(
                by_k[10]["coverage"]
            )
        metrics["per_query"] = {
            "recall@10": by_k[10]["fused_recall"] if 10 in by_k else [],
            f"mrr@{max_k}": by_k[max_k]["reciprocal_rank"],
            "coverage@10": by_k[10]["coverage"] if 10 in by_k else [],
        }
        output[config] = {"metrics": metrics}

    baseline_values = records[baseline]
    for config, payload in output.items():
        values = records[config]
        deltas = {}
        for name, k, field in (
            ("recall@10", 10, "fused_recall"),
            ("coverage@10", 10, "coverage"),
            (f"mrr@{max(baseline_values)}", max(baseline_values), "reciprocal_rank"),
        ):
            if k not in values or k not in baseline_values:
                continue
            deltas[name] = paired_bootstrap_delta(
                values[k][field],
                baseline_values[k][field],
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
        payload["delta_vs_baseline"] = deltas
    return output
