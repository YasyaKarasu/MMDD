#!/usr/bin/env python
"""Fit modality-specific monotonic row-evidence support maps for R10 G5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mmdd_progress import progress

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.row_support import fit_isotonic


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _candidate_evidence(
    pool_paths: list[Path],
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    by_query: dict[str, dict[str, str]] = {}
    metadata_rows = []
    for pool_path in pool_paths:
        metadata_path = pool_path.with_suffix(pool_path.suffix + ".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["output_sha256"] != checkpoint_fingerprint(pool_path):
            raise ValueError(f"{pool_path}: path-pool fingerprint mismatch")
        if metadata["split"] != "train":
            raise ValueError(f"{pool_path}: row-support calibration requires train split")
        metadata_rows.append(metadata)
        with pool_path.open(encoding="utf-8") as handle:
            for line in progress(
                handle,
                desc=f"Read {pool_path.name}",
                unit="query",
                leave=False,
            ):
                record = json.loads(line)
                query_id = str(record["query_id"])
                evidence = by_query.setdefault(query_id, {})
                for paths in record["paths_by_target"].values():
                    for path in paths:
                        if path["kind"] != "evidence":
                            continue
                        evidence_id = str(path["evidence_id"])
                        evidence_type = str(path["evidence_type"])
                        previous = evidence.setdefault(evidence_id, evidence_type)
                        if previous != evidence_type:
                            raise ValueError(
                                f"{evidence_id}: inconsistent evidence type"
                            )
    return by_query, metadata_rows


def _recovery_rows(
    paths: list[Path], query_ids: set[str]
) -> dict[tuple[str, str], set[int]]:
    result: dict[tuple[str, str], set[int]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in progress(
                handle,
                desc=f"Read {path.name}",
                unit="recovery",
                leave=False,
            ):
                row = json.loads(line)
                query_id = str(row["query_table_id"])
                if str(row.get("split")) != "train" or query_id not in query_ids:
                    continue
                evidence_id = str(row.get("evidence", {}).get("asset_id", ""))
                result.setdefault((query_id, evidence_id), set()).add(
                    int(row["query_row_id"])
                )
    return result


def _auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _score, label in ranked[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    pool_paths = [Path(value).resolve() for value in args.calibration_pools]
    candidates, metadata_rows = _candidate_evidence(pool_paths)
    recoveries = _recovery_rows(
        [Path(value) for value in args.recoveries], set(candidates)
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    rows: dict[str, dict[str, list[Any]]] = {
        "text": {"scores": [], "labels": []},
        "image": {"scores": [], "labels": []},
    }
    used_pairs = 0
    skipped_unverified_pairs = 0
    for query_id, evidence in progress(
        candidates.items(), desc="Build row-support calibration", unit="query"
    ):
        query = store.embedding_features(query_id)
        if query.row_embeddings is None:
            raise ValueError(f"{query_id}: missing cached row embeddings")
        for evidence_id, evidence_type in evidence.items():
            positive_rows = recoveries.get((query_id, evidence_id))
            if not positive_rows:
                skipped_unverified_pairs += 1
                continue
            evidence_vector = store.embedding_features(evidence_id).embedding
            similarities = torch.mv(query.row_embeddings, evidence_vector).tolist()
            if max(positive_rows) >= len(similarities):
                raise ValueError(f"{query_id}/{evidence_id}: recovery row out of range")
            rows[evidence_type]["scores"].extend(float(value) for value in similarities)
            rows[evidence_type]["labels"].extend(
                int(index in positive_rows) for index in range(len(similarities))
            )
            used_pairs += 1

    models = {}
    diagnostics = {}
    for evidence_type, values in rows.items():
        scores = values["scores"]
        labels = values["labels"]
        model = fit_isotonic(scores, labels)
        predicted = [model.predict(score) for score in scores]
        models[evidence_type] = model.to_json()
        diagnostics[evidence_type] = {
            "rows": len(labels),
            "positive_rows": sum(labels),
            "positive_rate": sum(labels) / len(labels),
            "retrieved_recovery_supported_evidence_pairs": len(labels) // 5,
            "raw_cosine_auroc": _auc(scores, labels),
            "isotonic_brier_in_sample": sum(
                (prediction - label) ** 2
                for prediction, label in zip(predicted, labels)
            )
            / len(labels),
            "isotonic_blocks": len(model.values),
        }
    payload = {
        "format_version": 1,
        "kind": "r10_row_support_isotonic",
        "fit_split": "train_calibration materialized with split=train",
        "calibration_pools": [str(path) for path in pool_paths],
        "calibration_pool_sha256": [
            metadata["output_sha256"] for metadata in metadata_rows
        ],
        "features": str(Path(args.features).resolve()),
        "label_policy": (
            "Use only retrieved evidence with at least one confirmed recovery for "
            "the query. Recovered rows are positive; other rows for that same "
            "query-evidence pair are localization negatives. Unverified evidence "
            "pairs are excluded rather than labeled negative."
        ),
        "used_query_evidence_pairs": used_pairs,
        "skipped_unverified_query_evidence_pairs": skipped_unverified_pairs,
        "models": models,
        "diagnostics": diagnostics,
    }
    _write_json(Path(args.output), payload)
    print(json.dumps({"status": "pass", "output": args.output}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-pools", required=True, nargs="+")
    parser.add_argument("--recoveries", required=True, nargs="+")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    args = parser.parse_args()
    if args.feature_cache_size <= 0:
        parser.error("Feature cache size must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
