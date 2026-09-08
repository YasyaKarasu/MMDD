#!/usr/bin/env python
"""Evaluate R10 Task-F fusion rules on one fixed Student path pool."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch
from mmdd_progress import progress

from mmdd_stage1.checkpoints import load_path_aggregator, load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    checkpoint_fingerprint,
    fuse_ranked_channels,
    rank_detailed_paths,
)
from mmdd_stage1.selection import load_stage1_selection, write_json


DEFAULT_RECALL_KS = (10, 20, 50)
SCORE_EPSILON = 1e-6


def _copy_result(result: dict[str, Any], *, score: float) -> dict[str, Any]:
    return {**result, "score": float(score)}


def calibrated_union_fusion(
    direct: Sequence[dict[str, Any]],
    evidence: Sequence[dict[str, Any]],
    direct_confidences: dict[str, float],
    *,
    evidence_weight: float,
) -> list[dict[str, Any]]:
    """F4: blend direct confidence with evidence support over the D/E union."""

    if not 0.0 <= evidence_weight <= 1.0:
        raise ValueError("evidence_weight must be within [0, 1]")
    by_target = {
        str(row["target_id"]): row for row in [*direct, *evidence]
    }
    missing = set(by_target) - set(direct_confidences)
    if missing:
        raise ValueError(
            "Missing direct confidence for union targets: "
            + ", ".join(sorted(missing)[:10])
        )
    evidence_by_target = {
        str(row["target_id"]): float(row["evidence_score"])
        for row in evidence
    }
    fused = []
    for target_id, row in by_target.items():
        direct_score = float(direct_confidences[target_id])
        evidence_score = evidence_by_target.get(target_id, 0.0)
        if not math.isfinite(direct_score) or not (
            -SCORE_EPSILON <= direct_score <= 1.0 + SCORE_EPSILON
        ):
            raise ValueError("F4 direct confidence must be within [0, 1]")
        if not math.isfinite(evidence_score) or not (
            -SCORE_EPSILON <= evidence_score <= 1.0 + SCORE_EPSILON
        ):
            raise ValueError("F4 evidence support must be within [0, 1]")
        direct_score = min(max(direct_score, 0.0), 1.0)
        evidence_score = min(max(evidence_score, 0.0), 1.0)
        score = (
            (1.0 - evidence_weight) * direct_score
            + evidence_weight * evidence_score
        )
        fused.append(
            {
                **row,
                "score": score,
                "f4_direct_confidence": direct_score,
                "f4_evidence_support": evidence_score,
            }
        )
    return sorted(
        fused,
        key=lambda row: (-float(row["score"]), str(row["target_id"])),
    )


def _take_unique(
    ranking: Sequence[dict[str, Any]],
    selected: set[str],
    count: int,
) -> tuple[list[dict[str, Any]], int]:
    if count == 0:
        return [], 0
    rows = []
    next_index = 0
    for next_index, row in enumerate(ranking, 1):
        target_id = str(row["target_id"])
        if target_id in selected:
            continue
        selected.add(target_id)
        rows.append(row)
        if len(rows) == count:
            break
    return rows, next_index


def reserved_channel_fusion(
    direct: Sequence[dict[str, Any]],
    evidence: Sequence[dict[str, Any]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    """F5: reserve half of top-K for each channel, dedupe, then alternate."""

    if k <= 0:
        raise ValueError("k must be positive")
    selected: set[str] = set()
    direct_reserved, direct_index = _take_unique(
        direct, selected, (k + 1) // 2
    )
    evidence_reserved, evidence_index = _take_unique(
        evidence, selected, k // 2
    )
    result = []
    for index in range(max(len(direct_reserved), len(evidence_reserved))):
        if index < len(direct_reserved):
            result.append({**direct_reserved[index], "f5_channel": "direct"})
        if index < len(evidence_reserved):
            result.append({**evidence_reserved[index], "f5_channel": "evidence"})

    direct_tail = iter(direct[direct_index:])
    evidence_tail = iter(evidence[evidence_index:])
    streams = (direct_tail, evidence_tail)
    exhausted = [False, False]
    stream_index = 0
    while len(result) < k and not all(exhausted):
        current = stream_index % 2
        stream_index += 1
        if exhausted[current]:
            continue
        for row in streams[current]:
            target_id = str(row["target_id"])
            if target_id in selected:
                continue
            selected.add(target_id)
            result.append(
                {
                    **row,
                    "f5_channel": "direct" if current == 0 else "evidence",
                }
            )
            break
        else:
            exhausted[current] = True
    return [
        _copy_result(row, score=1.0 / rank)
        for rank, row in enumerate(result[:k], 1)
    ]


def _selected_evidence(
    target: dict[str, Any] | None, evidence_budget: int
) -> list[str]:
    if target is None:
        return []
    if "selected_evidence_ids" in target:
        return [
            str(value)
            for value in target["selected_evidence_ids"][:evidence_budget]
        ]
    evidence = sorted(
        (path for path in target["paths"] if path["kind"] == "evidence"),
        key=lambda path: (-float(path["path_score"]), str(path["evidence_id"])),
    )
    return [str(path["evidence_id"]) for path in evidence[:evidence_budget]]


def _recoverable_rows(
    paths: Sequence[Path], split: str
) -> dict[tuple[str, str, str], set[int]]:
    rows: dict[tuple[str, str, str], set[int]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if str(record.get("split")) != split:
                    continue
                evidence_id = str(record.get("evidence", {}).get("asset_id", ""))
                key = (
                    str(record["query_table_id"]),
                    str(record["target_table_id"]),
                    evidence_id,
                )
                rows.setdefault(key, set()).add(int(record["query_row_id"]))
    return rows


def _recall(ranking: Sequence[str], positives: set[str], k: int) -> float:
    return len(set(ranking[:k]) & positives) / len(positives)


def _mrr(ranking: Sequence[str], positives: set[str], k: int) -> float:
    return next(
        (
            1.0 / rank
            for rank, target_id in enumerate(ranking[:k], 1)
            if target_id in positives
        ),
        0.0,
    )


def _empty_metrics(recall_ks: Sequence[int]) -> dict[str, Any]:
    return {
        "queries": 0,
        "recall": {f"recall@{k}": [] for k in recall_ks},
        "mrr": [],
        "unique_targets": {f"unique_targets@{k}": [] for k in recall_ks},
        "valid_path": {f"valid_path_recall@{k},4": [0, 0] for k in recall_ks},
        "row_support": {f"row_support_coverage@{k},4": [] for k in recall_ks},
        "recoverable_row_support": {
            f"recoverable_row_coverage@{k},4": [] for k in recall_ks
        },
        "attribution": {
            f"targets_added_vs_direct@{k}": 0 for k in recall_ks
        }
        | {f"targets_removed_vs_direct@{k}": 0 for k in recall_ks}
        | {f"positive_rescued@{k}": 0 for k in recall_ks}
        | {f"positive_displaced@{k}": 0 for k in recall_ks}
        | {f"evidence_independent_positive@{k}": 0 for k in recall_ks}
        | {f"direct_positive_with_valid_evidence@{k}": 0 for k in recall_ks},
        "per_query": [],
    }


def _accumulate(
    metrics: dict[str, Any],
    record: dict[str, Any],
    ranking: Sequence[dict[str, Any]],
    direct: Sequence[dict[str, Any]],
    evidence: Sequence[dict[str, Any]],
    recoveries: dict[tuple[str, str, str], set[int]],
    *,
    recall_ks: Sequence[int],
    query_rows: int,
    evidence_budget: int,
    rankings_by_k: dict[int, Sequence[dict[str, Any]]] | None = None,
) -> None:
    metrics["queries"] += 1
    query_id = str(record["query_id"])
    positives = {str(value) for value in record["positive_target_ids"]}
    ranked_ids = [str(row["target_id"]) for row in ranking]
    direct_ids = [str(row["target_id"]) for row in direct]
    evidence_ids = [str(row["target_id"]) for row in evidence]
    targets = {str(row["target_id"]): row for row in ranking}
    valid_evidence = {
        str(target_id): {str(value) for value in values}
        for target_id, values in record.get(
            "positive_evidence_by_target", {}
        ).items()
    }
    per_query: dict[str, Any] = {"query_id": query_id}
    for k in recall_ks:
        current_ranking = (rankings_by_k or {}).get(k, ranking)
        ranked_ids = [str(row["target_id"]) for row in current_ranking]
        targets = {str(row["target_id"]): row for row in current_ranking}
        final_top = set(ranked_ids[:k])
        direct_top = set(direct_ids[:k])
        evidence_top = set(evidence_ids[:k])
        recall = _recall(ranked_ids, positives, k)
        metrics["recall"][f"recall@{k}"].append(recall)
        metrics["unique_targets"][f"unique_targets@{k}"].append(
            len(final_top)
        )
        metrics["attribution"][f"targets_added_vs_direct@{k}"] += len(
            final_top - direct_top
        )
        metrics["attribution"][f"targets_removed_vs_direct@{k}"] += len(
            direct_top - final_top
        )
        metrics["attribution"][f"positive_rescued@{k}"] += len(
            (final_top - direct_top) & positives
        )
        metrics["attribution"][f"positive_displaced@{k}"] += len(
            (direct_top - final_top) & positives
        )
        metrics["attribution"][f"evidence_independent_positive@{k}"] += len(
            (final_top & evidence_top & positives) - direct_top
        )

        valid_key = f"valid_path_recall@{k},4"
        valid_numerator = 0
        valid_denominator = 0
        supported_row_values = []
        recoverable_row_values = []
        direct_with_valid = 0
        for target_id, expected_evidence in valid_evidence.items():
            valid_denominator += 1
            selected = (
                _selected_evidence(targets.get(target_id), evidence_budget)
                if target_id in final_top
                else []
            )
            valid_selected = set(selected) & expected_evidence
            valid_numerator += bool(valid_selected)
            if target_id in direct_top and valid_selected:
                direct_with_valid += 1
            recoverable = set().union(
                *(
                    recoveries.get((query_id, target_id, evidence_id), set())
                    for evidence_id in expected_evidence
                )
            )
            supported = set().union(
                *(
                    recoveries.get((query_id, target_id, evidence_id), set())
                    for evidence_id in valid_selected
                )
            )
            row_value = len(supported) / query_rows
            recoverable_value = (
                len(supported) / len(recoverable) if recoverable else 0.0
            )
            metrics["row_support"][f"row_support_coverage@{k},4"].append(
                row_value
            )
            metrics["recoverable_row_support"][
                f"recoverable_row_coverage@{k},4"
            ].append(recoverable_value)
            supported_row_values.append(row_value)
            recoverable_row_values.append(recoverable_value)
        metrics["valid_path"][valid_key][0] += valid_numerator
        metrics["valid_path"][valid_key][1] += valid_denominator
        metrics["attribution"][f"direct_positive_with_valid_evidence@{k}"] += (
            direct_with_valid
        )
        per_query[f"recall@{k}"] = recall
        per_query[valid_key] = (
            valid_numerator / valid_denominator if valid_denominator else 0.0
        )
        per_query[f"row_support_coverage@{k},4"] = (
            statistics.fmean(supported_row_values)
            if supported_row_values
            else 0.0
        )
        per_query[f"recoverable_row_coverage@{k},4"] = (
            statistics.fmean(recoverable_row_values)
            if recoverable_row_values
            else 0.0
        )
    max_k = max(recall_ks)
    ranked_ids = [str(row["target_id"]) for row in ranking]
    mrr = _mrr(ranked_ids, positives, max_k)
    metrics["mrr"].append(mrr)
    per_query[f"mrr@{max_k}"] = mrr
    metrics["per_query"].append(per_query)


def _mean(values: Sequence[float | int]) -> float:
    return statistics.fmean(values) if values else 0.0


def _finalize(metrics: dict[str, Any], recall_ks: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "queries": metrics["queries"],
        "attribution": metrics["attribution"],
        "per_query": metrics["per_query"],
        **{name: _mean(values) for name, values in metrics["recall"].items()},
        f"mrr@{max(recall_ks)}": _mean(metrics["mrr"]),
    }
    result["unique_targets"] = {
        name: {
            "mean": _mean(values),
            "minimum": min(values) if values else 0,
            "maximum": max(values) if values else 0,
        }
        for name, values in metrics["unique_targets"].items()
    }
    result["valid_path"] = {
        name: {
            "value": numerator / denominator if denominator else 0.0,
            "supported_positive_pairs": denominator,
        }
        for name, (numerator, denominator) in metrics["valid_path"].items()
    }
    result["row_support"] = {
        name: _mean(values) for name, values in metrics["row_support"].items()
    }
    result["recoverable_row_support"] = {
        name: _mean(values)
        for name, values in metrics["recoverable_row_support"].items()
    }
    return result


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "min": min(values) if values else None,
        "p10": _percentile(values, 0.10),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def score_direct_confidences(
    student: torch.nn.Module,
    store: FeatureStore,
    query_id: str,
    target_ids: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    query = store.embedding_features(query_id).for_scoring(
        device, include_hidden=False
    )
    result = {}
    with torch.inference_mode():
        for start in range(0, len(target_ids), batch_size):
            batch_ids = target_ids[start : start + batch_size]
            targets = [
                store.embedding_features(target_id).for_scoring(
                    device, include_hidden=False
                )
                for target_id in batch_ids
            ]
            scores = student.score_pairs_in_space(
                [query] * len(targets), targets, "confidence"
            )
            result.update(
                (target_id, float(score))
                for target_id, score in zip(batch_ids, scores.detach().cpu())
            )
    return result


def scoring_object_ids(path_pool: Path) -> list[str]:
    object_ids = []
    with path_pool.open(encoding="utf-8") as handle:
        for line in progress(
            handle, desc="Collect F4 scoring objects", unit="query", leave=False
        ):
            record = json.loads(line)
            object_ids.append(str(record["query_id"]))
            object_ids.extend(str(value) for value in record["paths_by_target"])
    return list(dict.fromkeys(object_ids))


def _parse_recall_ks(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(part) for part in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--recall-ks must be comma-separated integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("recall ks must be positive")
    return values


def _candidate_metrics(
    results: dict[str, dict[str, Any]], name: str
) -> tuple[float, float, float]:
    metrics = results[name]
    return (
        float(metrics["valid_path"]["valid_path_recall@10,4"]["value"]),
        float(metrics["row_support"]["row_support_coverage@10,4"]),
        float(metrics["recall@10"]),
    )


def _non_dominated(results: dict[str, dict[str, Any]]) -> list[str]:
    values = {name: _candidate_metrics(results, name) for name in results}
    retained = []
    for name, metrics in values.items():
        dominated = any(
            other != name
            and all(left >= right for left, right in zip(other_metrics, metrics))
            and any(left > right for left, right in zip(other_metrics, metrics))
            for other, other_metrics in values.items()
        )
        if not dominated:
            retained.append(name)
    return sorted(retained, key=lambda name: values[name], reverse=True)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# R10 Task F: {payload['candidate']}",
        "",
        "All configurations use the same fixed 100/20/20 ANN path pool and "
        "return unique targets. F4 direct confidences are computed for every "
        "target in the direct/evidence union.",
        "",
        "| Configuration | R@10 | MRR@50 | ValidPath@10,4 | RowSupport@10,4 | Unique@10 | Rescued | Displaced |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in payload["results"].items():
        lines.append(
            f"| `{name}` | {metrics['recall@10']:.2%} | "
            f"{metrics['mrr@50']:.4f} | "
            f"{metrics['valid_path']['valid_path_recall@10,4']['value']:.2%} | "
            f"{metrics['row_support']['row_support_coverage@10,4']:.2%} | "
            f"{metrics['unique_targets']['unique_targets@10']['mean']:.2f} | "
            f"{metrics['attribution']['positive_rescued@10']} | "
            f"{metrics['attribution']['positive_displaced@10']} |"
        )
    lines.extend(
        [
            "",
            f"Selected F4 lambda: `{payload['selected_f4']}`.",
            "Dev non-dominated configurations: "
            + ", ".join(f"`{name}`" for name in payload["non_dominated"])
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    path_pool = Path(args.path_pool).resolve()
    metadata_path = path_pool.with_suffix(path_pool.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["output_sha256"] != checkpoint_fingerprint(path_pool):
        raise ValueError("Path-pool fingerprint differs from its metadata")
    expected_budget = {
        "direct_k": 100,
        "evidence_k_per_modality": 20,
        "targets_per_evidence": 20,
        "evidence_types": ["text", "image"],
    }
    if metadata.get("retrieval_budget") != expected_budget:
        raise ValueError("Task F requires the fixed R10 100/20/20 path pool")
    if metadata.get("split") != "dev":
        raise ValueError("Task F fusion selection must use the dev split")

    selection_path = Path(metadata["selection"])
    selection = load_stage1_selection(selection_path)
    checkpoint = Path(selection["best_checkpoint"])
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    if checkpoint_sha256 != metadata.get("student_checkpoint_sha256"):
        raise ValueError("Path pool and selected Student checkpoint differ")
    aggregator = load_path_aggregator(checkpoint)
    if aggregator.config() != metadata.get("path_aggregation"):
        raise ValueError("Path pool and selected aggregation configuration differ")

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    student = load_student(checkpoint, device).eval()
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    preloaded = 0
    if not args.no_preload:
        preloaded = store.preload_embeddings(scoring_object_ids(path_pool))

    recoveries = _recoverable_rows(
        [Path(value) for value in args.recoveries], str(metadata["split"])
    )
    config_names = ["f0_direct", "f1_evidence", "f2_rrf_e005", "f3_rrf_equal"]
    config_names.extend(f"f4_lambda_{value:g}" for value in args.lambdas)
    config_names.append("f5_reserved_half")
    accumulators = {
        name: _empty_metrics(args.recall_ks) for name in config_names
    }
    distributions = {
        "direct_confidence": {"positive": [], "negative": []},
        "evidence_support": {"positive": [], "negative": []},
    }
    total_union_targets = 0
    max_k = max(args.recall_ks)
    records = 0
    with path_pool.open(encoding="utf-8") as handle:
        for line in progress(handle, desc="Evaluate Task F", unit="query"):
            record = json.loads(line)
            ranked = rank_detailed_paths(
                record["paths_by_target"],
                aggregator=aggregator,
                rrf_k=60,
                fusion_mode="weighted_rrf",
                direct_weight=1.0,
                evidence_weight=0.05,
                gated_evidence_min_paths=2,
                gated_evidence_quantile=0.75,
            )
            direct = ranked["direct"]
            evidence = ranked["evidence"]
            union_ids = list(
                dict.fromkeys(
                    str(row["target_id"]) for row in [*direct, *evidence]
                )
            )
            direct_confidences = score_direct_confidences(
                student,
                store,
                str(record["query_id"]),
                union_ids,
                device=device,
                batch_size=args.pair_batch_size,
            )
            total_union_targets += len(union_ids)
            positives = {str(value) for value in record["positive_target_ids"]}
            evidence_by_target = {
                str(row["target_id"]): float(row["evidence_score"])
                for row in evidence
            }
            for target_id, score in direct_confidences.items():
                label = "positive" if target_id in positives else "negative"
                distributions["direct_confidence"][label].append(score)
                distributions["evidence_support"][label].append(
                    evidence_by_target.get(target_id, 0.0)
                )

            configurations = {
                "f0_direct": [
                    _copy_result(row, score=float(row["direct_score"]))
                    for row in direct
                ],
                "f1_evidence": [
                    _copy_result(row, score=float(row["evidence_score"]))
                    for row in evidence
                ],
                "f2_rrf_e005": fuse_ranked_channels(
                    direct,
                    evidence,
                    rrf_k=60,
                    fusion_mode="weighted_rrf",
                    direct_weight=1.0,
                    evidence_weight=0.05,
                ),
                "f3_rrf_equal": fuse_ranked_channels(
                    direct,
                    evidence,
                    rrf_k=60,
                    fusion_mode="rrf",
                ),
                "f5_reserved_half": reserved_channel_fusion(
                    direct, evidence, k=max_k
                ),
            }
            configurations.update(
                {
                    f"f4_lambda_{value:g}": calibrated_union_fusion(
                        direct,
                        evidence,
                        direct_confidences,
                        evidence_weight=value,
                    )
                    for value in args.lambdas
                }
            )
            for name, ranking in configurations.items():
                _accumulate(
                    accumulators[name],
                    record,
                    ranking,
                    direct,
                    evidence,
                    recoveries,
                    recall_ks=args.recall_ks,
                    query_rows=args.query_rows,
                    evidence_budget=args.evidence_budget,
                    rankings_by_k=(
                        {k: reserved_channel_fusion(direct, evidence, k=k)
                         for k in args.recall_ks}
                        if name == "f5_reserved_half" else None
                    ),
                )
            records += 1

    if records != int(metadata["queries"]):
        raise ValueError(
            f"Task F read {records} queries, expected {metadata['queries']}"
        )
    results = {
        name: _finalize(metrics, args.recall_ks)
        for name, metrics in accumulators.items()
    }
    f4_names = [f"f4_lambda_{value:g}" for value in args.lambdas]
    selected_f4 = max(
        f4_names,
        key=lambda name: _candidate_metrics(results, name),
    )
    payload = {
        "format_version": 1,
        "candidate": args.candidate,
        "path_pool": str(path_pool),
        "path_pool_sha256": metadata["output_sha256"],
        "selection": str(selection_path),
        "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": checkpoint_sha256,
        "features": str(Path(args.features).resolve()),
        "retrieval_budget": metadata["retrieval_budget"],
        "evidence_bundle_budget": args.evidence_budget,
        "recall_ks": list(args.recall_ks),
        "lambdas": list(args.lambdas),
        "results": results,
        "selected_f4": selected_f4,
        "non_dominated": _non_dominated(results),
        "score_distributions": {
            score_name: {
                label: _distribution(values)
                for label, values in by_label.items()
            }
            for score_name, by_label in distributions.items()
        },
        "cost": {
            "queries": records,
            "preloaded_objects": preloaded,
            "shared_f4_direct_pair_rescores": total_union_targets,
            "mean_union_targets_per_query": total_union_targets / records,
            "fusion_ann_calls": 0,
            "fixed_upstream_ann_budget": metadata["retrieval_budget"],
            "reader_calls": 0,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", payload)
    (output_dir / "RESULTS.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "candidate": args.candidate,
                "selected_f4": selected_f4,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--path-pool", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--recoveries", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-rows", type=int, default=5)
    parser.add_argument("--evidence-budget", type=int, default=4)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--recall-ks", type=_parse_recall_ks, default=DEFAULT_RECALL_KS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pair-batch-size", type=int, default=512)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()
    if min(
        args.query_rows,
        args.evidence_budget,
        args.pair_batch_size,
        args.feature_cache_size,
    ) <= 0:
        parser.error("Row, evidence, batch, and cache budgets must be positive")
    if not args.lambdas or any(not 0.0 <= value <= 1.0 for value in args.lambdas):
        parser.error("--lambdas must contain values within [0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
