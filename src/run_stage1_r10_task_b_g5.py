#!/usr/bin/env python
"""Evaluate R10 G5 greedy row-support aggregation on fixed path pools."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from mmdd_progress import progress

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import checkpoint_fingerprint, fuse_ranked_channels
from mmdd_stage1.row_support import (
    IsotonicModel,
    greedy_row_bundle,
    load_evidence_content_keys,
)
from run_stage1_r10_task_b import (
    _accumulate_query,
    _empty_metrics,
    _finalize,
    _recovery_rows,
    _write_json,
)
from run_stage1_r10_task_b_advanced import _sigmoid


THRESHOLDS = (0.0, 0.25, 0.5)


def _row_support_predictions(
    query_id: str,
    paths_by_target: dict[str, list[dict[str, Any]]],
    *,
    store: FeatureStore,
    models: dict[str, IsotonicModel],
) -> dict[str, list[float]]:
    evidence_types = {}
    for paths in paths_by_target.values():
        for path in paths:
            if path["kind"] == "evidence":
                evidence_types[str(path["evidence_id"])] = str(
                    path["evidence_type"]
                )
    query = store.embedding_features(query_id)
    if query.row_embeddings is None:
        raise ValueError(f"{query_id}: missing cached row embeddings")
    result = {}
    for evidence_id, evidence_type in evidence_types.items():
        evidence = store.embedding_features(evidence_id)
        similarities = torch.mv(query.row_embeddings, evidence.embedding).tolist()
        model = models[evidence_type]
        result[evidence_id] = [model.predict(value) for value in similarities]
    return result


def _evidence_candidates(
    paths: list[dict[str, Any]],
    *,
    content_keys: dict[str, str],
    top_l: int,
) -> list[dict[str, Any]]:
    by_content = {}
    for source in paths:
        if source["kind"] != "evidence":
            continue
        path = dict(source)
        evidence_id = str(path["evidence_id"])
        quality = min(
            _sigmoid(float(path["query_evidence_score"])),
            _sigmoid(float(path["evidence_target_score"])),
        )
        path["path_score"] = quality
        candidate = {
            "evidence_id": evidence_id,
            "quality": quality,
            "path": path,
        }
        try:
            content_key = content_keys[evidence_id]
        except KeyError as exc:
            raise KeyError(
                f"Evidence content-key manifest has no object {evidence_id!r}"
            ) from exc
        previous = by_content.get(content_key)
        if previous is None or (-quality, evidence_id) < (
            -float(previous["quality"]),
            str(previous["evidence_id"]),
        ):
            by_content[content_key] = candidate
    return sorted(
        by_content.values(),
        key=lambda value: (-float(value["quality"]), str(value["evidence_id"])),
    )[:top_l]


def _rank_g5(
    record: dict[str, Any],
    *,
    predicted_support: dict[str, list[float]],
    oracle_recoveries: dict[tuple[str, str, str], set[int]],
    content_keys: dict[str, str],
    evidence_budget: int,
    top_l: int,
    threshold: float,
    oracle: bool,
) -> dict[str, list[dict[str, Any]]]:
    query_id = str(record["query_id"])
    results = []
    for target_id, source_paths in record["paths_by_target"].items():
        target_id = str(target_id)
        direct_score = next(
            (
                float(path["path_score"])
                for path in source_paths
                if path["kind"] == "direct"
            ),
            None,
        )
        candidates = _evidence_candidates(
            source_paths, content_keys=content_keys, top_l=top_l
        )
        selected = []
        evidence_score = None
        if candidates:
            if oracle:
                row_count = len(next(iter(predicted_support.values())))
                support = {
                    str(candidate["evidence_id"]): [
                        float(
                            row_index
                            in oracle_recoveries.get(
                                (
                                    query_id,
                                    target_id,
                                    str(candidate["evidence_id"]),
                                ),
                                set(),
                            )
                        )
                        for row_index in range(row_count)
                    ]
                    for candidate in candidates
                }
            else:
                support = {
                    str(candidate["evidence_id"]): predicted_support[
                        str(candidate["evidence_id"])
                    ]
                    for candidate in candidates
                }
            selected, evidence_score = greedy_row_bundle(
                candidates,
                row_support=support,
                budget=evidence_budget,
                threshold=threshold,
            )
        paths = [dict(path) for path in source_paths if path["kind"] == "direct"]
        paths.extend(candidate["path"] for candidate in candidates)
        results.append(
            {
                "target_id": target_id,
                "direct_score": direct_score,
                "evidence_score": evidence_score,
                "selected_evidence_ids": selected,
                "paths": paths,
            }
        )
    direct = sorted(
        (row for row in results if row["direct_score"] is not None),
        key=lambda row: (-float(row["direct_score"]), str(row["target_id"])),
    )
    evidence = sorted(
        (row for row in results if row["evidence_score"] is not None),
        key=lambda row: (-float(row["evidence_score"]), str(row["target_id"])),
    )
    fused = fuse_ranked_channels(
        direct,
        evidence,
        rrf_k=60,
        fusion_mode="weighted_rrf",
        direct_weight=1.0,
        evidence_weight=0.05,
        gated_evidence_min_paths=2,
        gated_evidence_quantile=0.75,
    )
    return {"fused": fused, "direct": direct, "evidence": evidence}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# R10 Task B G5: {payload['system']}",
        "",
        "Predicted G5 uses train-calibration isotonic row-support maps. Oracle rows "
        "use dev recovery labels only as an upper bound and are not a selectable "
        "system. All configurations use top-L=20 and B=4.",
        "",
        "| Configuration | Fused R@10 | Evidence R@10 | ValidPath@10,4 | RowSupport@10,4 | RecoverableRow@10,4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["results"].items():
        lines.append(
            f"| `{name}` | {row['fused']['recall@10']:.2%} | "
            f"{row['evidence']['recall@10']:.2%} | "
            f"{row['valid_path']['valid_path_recall@10,4']['value']:.2%} | "
            f"{row['row_support']['row_support_coverage@10,4']:.2%} | "
            f"{row['recoverable_row_support']['recoverable_row_coverage@10,4']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    pool_path = Path(args.path_pool).resolve()
    metadata_path = pool_path.with_suffix(pool_path.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["output_sha256"] != checkpoint_fingerprint(pool_path):
        raise ValueError("Path-pool fingerprint differs from its metadata")
    row_model_path = Path(args.row_support_model).resolve()
    row_model = json.loads(row_model_path.read_text(encoding="utf-8"))
    models = {
        evidence_type: IsotonicModel.from_json(payload)
        for evidence_type, payload in row_model["models"].items()
    }
    content_keys_path = Path(args.content_keys).resolve()
    content_keys, content_keys_sha256 = load_evidence_content_keys(
        content_keys_path
    )
    recoveries = _recovery_rows(
        [Path(value) for value in args.recoveries], metadata["split"]
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    config_names = [
        f"g5_{kind}_d{threshold:g}"
        for kind in ("predicted", "oracle")
        for threshold in THRESHOLDS
    ]
    metrics = {name: _empty_metrics() for name in config_names}
    with pool_path.open(encoding="utf-8") as handle:
        for line in progress(handle, desc="Evaluate G5", unit="query"):
            record = json.loads(line)
            predicted_support = _row_support_predictions(
                str(record["query_id"]),
                record["paths_by_target"],
                store=store,
                models=models,
            )
            for kind in ("predicted", "oracle"):
                for threshold in THRESHOLDS:
                    name = f"g5_{kind}_d{threshold:g}"
                    ranked = _rank_g5(
                        record,
                        predicted_support=predicted_support,
                        oracle_recoveries=recoveries,
                        content_keys=content_keys,
                        evidence_budget=args.evidence_budget,
                        top_l=args.top_l,
                        threshold=threshold,
                        oracle=kind == "oracle",
                    )
                    _accumulate_query(
                        metrics[name],
                        record,
                        ranked,
                        recoveries,
                        query_rows=args.query_rows,
                        evidence_budget=args.evidence_budget,
                    )
    payload = {
        "format_version": 1,
        "system": metadata["system"],
        "path_pool": str(pool_path),
        "path_pool_sha256": metadata["output_sha256"],
        "row_support_model": str(row_model_path),
        "row_support_model_sha256": checkpoint_fingerprint(row_model_path),
        "evidence_content_keys": str(content_keys_path),
        "evidence_content_keys_sha256": content_keys_sha256,
        "input_path_score_space": metadata["path_score_space"],
        "edge_transform": "uncalibrated_sigmoid_of_raw_logit",
        "path_combination": "min",
        "candidate_evidence_per_target": args.top_l,
        "evidence_bundle_budget": args.evidence_budget,
        "content_deduplication": "exact UTF-8 text or image-file SHA-256",
        "results": {name: _finalize(value) for name, value in metrics.items()},
        "oracle_policy": (
            "Dev recovery row assignments are used only for the separately named "
            "oracle upper bound."
        ),
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "metrics.json", payload)
    (output_dir / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "pass", "output_dir": str(output_dir)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-pool", required=True)
    parser.add_argument("--row-support-model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--content-keys", required=True)
    parser.add_argument("--recoveries", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-rows", type=int, default=5)
    parser.add_argument("--evidence-budget", type=int, default=4)
    parser.add_argument("--top-l", type=int, default=20)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    args = parser.parse_args()
    if min(
        args.query_rows,
        args.evidence_budget,
        args.top_l,
        args.feature_cache_size,
    ) <= 0:
        parser.error("Budgets and cache size must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
