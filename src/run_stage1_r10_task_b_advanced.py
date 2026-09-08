#!/usr/bin/env python
"""Run R10 Task-B confidence-combination and G3/G4 fixed-pool scans."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mmdd_progress import progress

from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import checkpoint_fingerprint
from run_stage1_r10_task_b import (
    RECALL_KS,
    _accumulate_query,
    _empty_metrics,
    _finalize,
    _rank_pool,
    _recovery_rows,
    _write_json,
)


TEMPERATURES = (0.1, 0.3, 1.0)
POWERS = (2.0, 4.0, 8.0)
THRESHOLDS = (0.25, 0.5)


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def confidence_paths(
    paths_by_target: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Map raw Student evidence-edge logits to uncalibrated confidences."""

    transformed = {}
    for target_id, paths in paths_by_target.items():
        target_paths = []
        for source in paths:
            path = dict(source)
            if path["kind"] == "evidence":
                path["query_evidence_score"] = _sigmoid(
                    float(path["query_evidence_score"])
                )
                path["evidence_target_score"] = _sigmoid(
                    float(path["evidence_target_score"])
                )
            target_paths.append(path)
        transformed[str(target_id)] = target_paths
    return transformed


def aggregation_configs(top_k: int) -> list[dict[str, Any]]:
    configs = [
        {
            "name": "min_max",
            "family": "G1",
            "aggregator": PathAggregator("max", top_k, path_combination="min"),
        },
        {
            "name": "product_max",
            "family": "B3-product",
            "aggregator": PathAggregator(
                "max", top_k, path_combination="product"
            ),
        },
    ]
    for temperature in TEMPERATURES:
        for scope, lse, lme in (
            ("full", "logsumexp", "logmeanexp"),
            ("top4", "topk_logsumexp", "topk_logmeanexp"),
        ):
            for family, aggregation in (("lse", lse), ("lme", lme)):
                configs.append(
                    {
                        "name": f"min_{scope}_{family}_t{temperature:g}",
                        "family": "G0" if family == "lse" else "G2b",
                        "aggregator": PathAggregator(
                            aggregation,
                            top_k,
                            temperature=temperature,
                            path_combination="min",
                        ),
                    }
                )
    for power in POWERS:
        configs.append(
            {
                "name": f"g3_r{power:g}",
                "family": "G3",
                "aggregator": PathAggregator(
                    "fixed_power_mean",
                    top_k,
                    power=power,
                    path_combination="min",
                ),
            }
        )
        for threshold in THRESHOLDS:
            configs.append(
                {
                    "name": f"g4_d{threshold:g}_r{power:g}",
                    "family": "G4",
                    "aggregator": PathAggregator(
                        "fixed_power_mean",
                        top_k,
                        power=power,
                        path_combination="min",
                        threshold=threshold,
                    ),
                }
            )
    return configs


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# R10 Task B advanced scan: {payload['system']}",
        "",
        "Evidence-edge raw logits are mapped with an uncalibrated sigmoid, then "
        "combined with min or product. G3/G4 use B=4. Thresholds are empirical "
        "confidence thresholds on this fixed candidate distribution, not calibrated "
        "full-corpus probabilities.",
        "",
        "| Configuration | Family | Fused R@10 | Evidence R@10 | ValidPath@10,4 | RowSupport@10,4 | RecoverableRow@10,4 | Path-count correlation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config in payload["configurations"]:
        row = payload["results"][config["name"]]
        lines.append(
            f"| `{config['name']}` | {config['family']} | "
            f"{row['fused']['recall@10']:.2%} | "
            f"{row['evidence']['recall@10']:.2%} | "
            f"{row['valid_path']['valid_path_recall@10,4']['value']:.2%} | "
            f"{row['row_support']['row_support_coverage@10,4']:.2%} | "
            f"{row['recoverable_row_support']['recoverable_row_coverage@10,4']:.2%} | "
            f"{row['path_count_score_pearson'] or 0.0:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    pool_path = Path(args.path_pool).resolve()
    metadata_path = pool_path.with_suffix(pool_path.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["output_sha256"] != checkpoint_fingerprint(pool_path):
        raise ValueError("Path-pool fingerprint differs from its metadata")
    expected_budget = {
        "direct_k": 100,
        "evidence_k_per_modality": 20,
        "targets_per_evidence": 20,
        "evidence_types": ["text", "image"],
    }
    if metadata["retrieval_budget"] != expected_budget:
        raise ValueError("Task B requires the fixed R10 100/20/20 path pool")

    configs = aggregation_configs(args.evidence_budget)
    metrics = {config["name"]: _empty_metrics() for config in configs}
    recoveries = _recovery_rows(
        [Path(value) for value in args.recoveries], metadata["split"]
    )
    with pool_path.open(encoding="utf-8") as handle:
        for line in progress(handle, desc="Scan confidence path pool", unit="query"):
            record = json.loads(line)
            paths = confidence_paths(record["paths_by_target"])
            for config in configs:
                ranked = _rank_pool(paths, config["aggregator"])
                _accumulate_query(
                    metrics[config["name"]],
                    record,
                    ranked,
                    recoveries,
                    query_rows=args.query_rows,
                    evidence_budget=args.evidence_budget,
                )
    results = {name: _finalize(values) for name, values in metrics.items()}
    payload = {
        "format_version": 1,
        "system": metadata["system"],
        "path_pool": str(pool_path),
        "path_pool_sha256": metadata["output_sha256"],
        "input_path_score_space": metadata["path_score_space"],
        "edge_transform": "uncalibrated_sigmoid_of_raw_logit",
        "path_combination": "explicit_per_configuration",
        "retrieval_budget": metadata["retrieval_budget"],
        "evidence_bundle_budget": args.evidence_budget,
        "configurations": [
            {
                "name": config["name"],
                "family": config["family"],
                **config["aggregator"].config(),
            }
            for config in configs
        ],
        "results": results,
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "metrics.json", payload)
    (output_dir / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "pass", "output_dir": str(output_dir)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-pool", required=True)
    parser.add_argument("--recoveries", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-rows", type=int, default=5)
    parser.add_argument("--evidence-budget", type=int, default=4)
    args = parser.parse_args()
    if args.query_rows <= 0 or args.evidence_budget <= 0:
        parser.error("Row and evidence budgets must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
