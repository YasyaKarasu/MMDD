#!/usr/bin/env python
"""Evaluate R11 E0/E1/E2 evidence retention on one fixed path pool."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from mmdd_progress import progress

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.row_support import (
    greedy_row_bundle,
    load_evidence_content_keys,
)


STRATEGIES = ("e0_top_quality", "e1_content_dedup", "e2_row_coverage")
INTERVENTIONS = ("original_mixed", "remove_image", "duplicate_same_row")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _evidence_paths(
    paths: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (dict(path) for path in paths if path.get("kind") == "evidence"),
        key=lambda path: (
            -float(path["path_score"]),
            str(path["evidence_id"]),
        ),
    )


def _deduplicate(
    paths: Sequence[dict[str, Any]], content_keys: dict[str, str]
) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for path in paths:
        evidence_id = str(path["evidence_id"])
        try:
            key = content_keys[evidence_id]
        except KeyError as exc:
            raise KeyError(f"Missing exact-content key for {evidence_id!r}") from exc
        previous = best.get(key)
        if previous is None or (
            -float(path["path_score"]), evidence_id
        ) < (
            -float(previous["path_score"]), str(previous["evidence_id"])
        ):
            best[key] = path
    return sorted(
        best.values(),
        key=lambda path: (-float(path["path_score"]), str(path["evidence_id"])),
    )


def _row_strengths(
    query_id: str,
    evidence_ids: Sequence[str],
    store: FeatureStore,
) -> dict[str, list[float]]:
    query = store.embedding_features(query_id)
    if query.row_embeddings is None:
        raise ValueError(f"{query_id}: missing cached row embeddings")
    result = {}
    for evidence_id in evidence_ids:
        evidence = store.embedding_features(evidence_id)
        cosine = torch.mv(
            query.row_embeddings.detach().cpu().float(),
            evidence.embedding.detach().cpu().float(),
        )
        result[evidence_id] = [
            min(1.0, max(0.0, (float(value) + 1.0) / 2.0))
            for value in cosine
        ]
    return result


def empty_intervention_stats() -> dict[str, float | int]:
    return {
        "targets_seen": 0,
        "image_paths_seen": 0,
        "image_paths_removed": 0,
        "image_paths_replaced": 0,
        "same_predicted_row_matches": 0,
        "fallback_matches": 0,
        "no_non_image_donor": 0,
        "path_score_gap_sum": 0.0,
    }


def finalize_intervention_stats(
    values: dict[str, float | int],
) -> dict[str, float | int | None]:
    replaced = int(values["image_paths_replaced"])
    return {
        **values,
        "mean_absolute_path_score_gap": (
            float(values["path_score_gap_sum"]) / replaced if replaced else None
        ),
    }


def intervene_paths(
    paths: Sequence[dict[str, Any]],
    *,
    intervention: str,
    query_id: str,
    store: FeatureStore,
    support_cache: dict[str, list[float]],
    stats: dict[str, float | int],
) -> list[dict[str, Any]]:
    """Apply a label-blind fixed-target evidence intervention."""

    if intervention not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {intervention}")
    copied = [dict(path) for path in paths]
    evidence = [path for path in copied if path.get("kind") == "evidence"]
    images = [path for path in evidence if path.get("evidence_type") == "image"]
    stats["targets_seen"] += 1
    stats["image_paths_seen"] += len(images)
    if intervention == "original_mixed" or not images:
        return copied

    non_images = [path for path in evidence if path.get("evidence_type") != "image"]
    retained = [path for path in copied if path.get("evidence_type") != "image"]
    stats["image_paths_removed"] += len(images)
    if intervention == "remove_image":
        return retained
    if not non_images:
        stats["no_non_image_donor"] += len(images)
        return retained

    evidence_ids = sorted(
        {
            str(path["evidence_id"])
            for path in [*images, *non_images]
            if str(path["evidence_id"]) not in support_cache
        }
    )
    support_cache.update(_row_strengths(query_id, evidence_ids, store))

    def strongest_row(path: dict[str, Any]) -> int:
        strengths = support_cache[str(path["evidence_id"])]
        return max(range(len(strengths)), key=lambda index: (strengths[index], -index))

    donors_by_row: dict[int, list[dict[str, Any]]] = {}
    for donor in non_images:
        donors_by_row.setdefault(strongest_row(donor), []).append(donor)
    replacements = []
    for image in images:
        same_row = donors_by_row.get(strongest_row(image), [])
        candidates = same_row or non_images
        if same_row:
            stats["same_predicted_row_matches"] += 1
        else:
            stats["fallback_matches"] += 1
        image_score = float(image["path_score"])
        donor = min(
            candidates,
            key=lambda path: (
                abs(float(path["path_score"]) - image_score),
                -float(path["path_score"]),
                str(path["evidence_id"]),
            ),
        )
        replacement = dict(donor)
        replacement["intervention_replaces_image_id"] = str(image["evidence_id"])
        replacements.append(replacement)
        stats["image_paths_replaced"] += 1
        stats["path_score_gap_sum"] += abs(float(donor["path_score"]) - image_score)
    return [*retained, *replacements]


def select_evidence(
    strategy: str,
    paths: Sequence[dict[str, Any]],
    *,
    query_id: str,
    store: FeatureStore,
    content_keys: dict[str, str],
    top_l: int,
    budget: int,
    support_cache: dict[str, list[float]],
) -> tuple[list[str], float | None]:
    candidates = _evidence_paths(paths)
    if strategy != "e0_top_quality":
        candidates = _deduplicate(candidates, content_keys)
    candidates = candidates[:top_l]
    if strategy != "e2_row_coverage":
        selected = [str(path["evidence_id"]) for path in candidates[:budget]]
        scores = [float(path["path_score"]) for path in candidates[:budget]]
        evidence_score = (
            math.log(sum(math.exp(value) for value in scores))
            if scores
            else None
        )
        return selected, evidence_score

    missing = [
        str(path["evidence_id"])
        for path in candidates
        if str(path["evidence_id"]) not in support_cache
    ]
    support_cache.update(_row_strengths(query_id, missing, store))
    greedy_candidates = [
        {
            "evidence_id": str(path["evidence_id"]),
            "quality": _sigmoid(float(path["path_score"])),
        }
        for path in candidates
    ]
    selected, coverage_strength = greedy_row_bundle(
        greedy_candidates,
        row_support={
            str(path["evidence_id"]): support_cache[str(path["evidence_id"])]
            for path in candidates
        },
        budget=budget,
        threshold=0.0,
    )
    return selected, coverage_strength


def _empty() -> dict[str, Any]:
    return {
        "implicit_positive_pairs": 0,
        "valid_b_count": 0,
        "row_b_values": [],
        "recoverable_row_values": [],
        "multi_row_2_count": 0,
        "multi_row_3_count": 0,
        "supported_row_count_distribution": Counter(),
        "selected_by_modality": Counter(),
        "valid_selected_by_modality": Counter(),
        "selected_evidence_count": 0,
        "per_pair": [],
    }


def _accumulate(
    values: dict[str, Any],
    record: dict[str, Any],
    target_id: str,
    selected: Sequence[str],
    evidence_type: dict[str, str],
) -> None:
    expected = {
        str(value)
        for value in record.get("positive_evidence_by_target", {}).get(
            target_id, []
        )
    }
    rows_by_evidence = {
        str(evidence_id): {int(row) for row in rows}
        for evidence_id, rows in record.get(
            "positive_evidence_rows_by_target", {}
        ).get(target_id, {}).items()
    }
    valid_selected = set(selected) & expected
    supported_rows = set().union(
        *(rows_by_evidence.get(evidence_id, set()) for evidence_id in valid_selected)
    )
    recoverable_rows = set().union(*rows_by_evidence.values()) if rows_by_evidence else set()
    row_count = int(record.get("query_row_count") or 0)
    if row_count <= 0:
        raise ValueError(f"{record['query_id']}: invalid query_row_count")
    row_b = len(supported_rows) / row_count
    recoverable = (
        len(supported_rows) / len(recoverable_rows) if recoverable_rows else 0.0
    )
    values["implicit_positive_pairs"] += 1
    values["valid_b_count"] += int(bool(valid_selected))
    values["row_b_values"].append(row_b)
    values["recoverable_row_values"].append(recoverable)
    values["multi_row_2_count"] += int(len(supported_rows) >= 2)
    values["multi_row_3_count"] += int(len(supported_rows) >= 3)
    values["supported_row_count_distribution"][len(supported_rows)] += 1
    values["selected_evidence_count"] += len(selected)
    values["selected_by_modality"].update(
        evidence_type.get(evidence_id, "unknown") for evidence_id in selected
    )
    values["valid_selected_by_modality"].update(
        evidence_type.get(evidence_id, "unknown") for evidence_id in valid_selected
    )
    values["per_pair"].append(
        {
            "query_id": str(record["query_id"]),
            "target_id": target_id,
            "query_kind": str(record.get("query_kind", "unknown")),
            "query_row_count": row_count,
            "selected_evidence_ids": list(selected),
            "valid_selected_evidence_ids": sorted(valid_selected),
            "supported_rows": sorted(supported_rows),
            "recoverable_rows": sorted(recoverable_rows),
            "valid_b": int(bool(valid_selected)),
            "row_b": row_b,
            "recoverable_row": recoverable,
            "multi_row_2": int(len(supported_rows) >= 2),
            "multi_row_3": int(len(supported_rows) >= 3),
        }
    )


def _finalize(values: dict[str, Any]) -> dict[str, Any]:
    denominator = values["implicit_positive_pairs"]
    return {
        "implicit_positive_pairs": denominator,
        "valid_b_count": values["valid_b_count"],
        "valid_b": values["valid_b_count"] / denominator if denominator else 0.0,
        "row_b": statistics.fmean(values["row_b_values"]) if denominator else 0.0,
        "recoverable_row": (
            statistics.fmean(values["recoverable_row_values"])
            if denominator
            else 0.0
        ),
        "multi_row_2_count": values["multi_row_2_count"],
        "multi_row_2": values["multi_row_2_count"] / denominator if denominator else 0.0,
        "multi_row_3_count": values["multi_row_3_count"],
        "multi_row_3": values["multi_row_3_count"] / denominator if denominator else 0.0,
        "supported_row_count_distribution": {
            str(key): count
            for key, count in sorted(values["supported_row_count_distribution"].items())
        },
        "mean_selected_evidence": (
            values["selected_evidence_count"] / denominator if denominator else 0.0
        ),
        "selected_by_modality": dict(sorted(values["selected_by_modality"].items())),
        "valid_selected_by_modality": dict(
            sorted(values["valid_selected_by_modality"].items())
        ),
        "per_pair": values["per_pair"],
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# R11 Task E: {payload['system']}",
        "",
        "All strategies use the same fixed L=20 path pool and B=4. E2 uses an ",
        "unlabeled monotonic transform of row-evidence cosine as localization ",
        "strength; it is not a calibrated support probability.",
        "",
        "| Strategy | ValidB | RowB | RecoverableRow | MultiRow2 | MultiRow3 | Mean selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in STRATEGIES:
        row = payload["results"][name]
        lines.append(
            f"| `{name}` | {row['valid_b']:.2%} "
            f"({row['valid_b_count']}/{row['implicit_positive_pairs']}) | "
            f"{row['row_b']:.2%} | {row['recoverable_row']:.2%} | "
            f"{row['multi_row_2']:.2%} | {row['multi_row_3']:.2%} | "
            f"{row['mean_selected_evidence']:.2f} |"
        )
    lines.extend(
        [
            "",
            "E3 was not run unless the separate attribute audit establishes its "
            "predeclared label and source-group trigger.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    pool = Path(args.path_pool).resolve()
    metadata_path = pool.with_suffix(pool.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("output_sha256") != checkpoint_fingerprint(pool):
        raise ValueError("Path-pool fingerprint differs from metadata")
    retrieval_budget = metadata.get("retrieval_budget", {})
    evidence_types = retrieval_budget.get("evidence_types", [])
    if (
        retrieval_budget.get("direct_k") != 100
        or retrieval_budget.get("targets_per_evidence") != 20
        or int(retrieval_budget.get("evidence_k_per_modality", 0))
        * len(evidence_types)
        != 40
        or not set(evidence_types) <= {"text", "image"}
    ):
        raise ValueError("Task E requires fixed direct=100, total Q->E=40, E->T=20")

    content_keys, content_sha256 = load_evidence_content_keys(
        Path(args.content_keys)
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    metrics = {name: _empty() for name in STRATEGIES}
    intervention_stats = empty_intervention_stats()
    duplicate_paths = 0
    evidence_paths = 0
    records = 0
    with pool.open(encoding="utf-8") as handle:
        for line in progress(handle, desc="Evaluate R11 Task E", unit="query"):
            record = json.loads(line)
            support_cache: dict[str, list[float]] = {}
            evidence_type = {}
            intervened_paths = {}
            for target_id, paths in record["paths_by_target"].items():
                candidates = _evidence_paths(paths)
                evidence_paths += len(candidates)
                duplicate_paths += len(candidates) - len(
                    _deduplicate(candidates, content_keys)
                )
                transformed = intervene_paths(
                    paths,
                    intervention=args.intervention,
                    query_id=str(record["query_id"]),
                    store=store,
                    support_cache=support_cache,
                    stats=intervention_stats,
                )
                intervened_paths[str(target_id)] = transformed
                evidence_type.update(
                    (str(path["evidence_id"]), str(path["evidence_type"]))
                    for path in _evidence_paths(transformed)
                )
            if str(record.get("query_kind")) == "implicit":
                for target_id in record.get("positive_evidence_by_target", {}):
                    paths = intervened_paths.get(str(target_id), [])
                    for strategy in STRATEGIES:
                        selected, _score = select_evidence(
                            strategy,
                            paths,
                            query_id=str(record["query_id"]),
                            store=store,
                            content_keys=content_keys,
                            top_l=args.top_l,
                            budget=args.evidence_budget,
                            support_cache=support_cache,
                        )
                        _accumulate(
                            metrics[strategy],
                            record,
                            str(target_id),
                            selected,
                            evidence_type,
                        )
            records += 1

    results = {name: _finalize(values) for name, values in metrics.items()}
    payload = {
        "format_version": 1,
        "experiment": "R11 Task E fixed-pool evidence retention",
        "system": metadata["system"],
        "split": metadata["split"],
        "path_pool": str(pool),
        "path_pool_sha256": metadata["output_sha256"],
        "queries": records,
        "top_l": args.top_l,
        "evidence_budget": args.evidence_budget,
        "intervention": args.intervention,
        "intervention_definition": (
            "replace each image path with the closest-scoring existing non-image "
            "path predicted to support the same strongest query row; fall back to "
            "the closest-scoring non-image path without using labels"
            if args.intervention == "duplicate_same_row"
            else args.intervention
        ),
        "intervention_stats": finalize_intervention_stats(intervention_stats),
        "content_keys": str(Path(args.content_keys).resolve()),
        "content_keys_sha256": content_sha256,
        "row_support_policy": (
            "unlabeled strength clamp((raw row-evidence cosine + 1) / 2, 0, 1); "
            "not a calibrated probability"
        ),
        "quality_policy": "sigmoid(raw fixed two-hop path score); monotonic strength",
        "content_duplicate_paths": duplicate_paths,
        "evidence_paths": evidence_paths,
        "results": results,
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "metrics.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_per_query.jsonl").open("w", encoding="utf-8") as handle:
        by_key = {
            name: {
                (row["query_id"], row["target_id"]): row
                for row in results[name]["per_pair"]
            }
            for name in STRATEGIES
        }
        for key in sorted(by_key[STRATEGIES[0]]):
            handle.write(
                json.dumps(
                    {
                        "query_id": key[0],
                        "target_id": key[1],
                        "strategies": {
                            name: by_key[name][key] for name in STRATEGIES
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (output_dir / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "pass", "output_dir": str(output_dir)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-pool", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--content-keys", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-l", type=int, default=20)
    parser.add_argument("--evidence-budget", type=int, default=4)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument(
        "--intervention", choices=INTERVENTIONS, default="original_mixed"
    )
    args = parser.parse_args()
    if min(args.top_l, args.evidence_budget, args.feature_cache_size) <= 0:
        parser.error("Budgets and cache size must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
