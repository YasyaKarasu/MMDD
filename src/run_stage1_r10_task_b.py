#!/usr/bin/env python
"""Run R10 Task-B paired LSE/LME scans on one fixed Stage-1 path pool."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmdd_progress import progress

from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import checkpoint_fingerprint, rank_detailed_paths


TEMPERATURES = (0.1, 0.3, 1.0)
RECALL_KS = (10, 20, 50)


@dataclass
class Correlation:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        self.count += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_yy += y * y
        self.sum_xy += x * y

    def pearson(self) -> float | None:
        if self.count < 2:
            return None
        numerator = self.count * self.sum_xy - self.sum_x * self.sum_y
        denominator = math.sqrt(
            max(0.0, self.count * self.sum_xx - self.sum_x**2)
            * max(0.0, self.count * self.sum_yy - self.sum_y**2)
        )
        return numerator / denominator if denominator else None


def aggregation_configs(top_k: int) -> list[dict[str, Any]]:
    result = []
    for temperature in TEMPERATURES:
        for scope, lse, lme in (
            ("full", "logsumexp", "logmeanexp"),
            ("top4", "topk_logsumexp", "topk_logmeanexp"),
        ):
            for family, form in (("lse", lse), ("lme", lme)):
                result.append(
                    {
                        "name": f"{scope}_{family}_t{temperature:g}",
                        "scope": scope,
                        "family": family,
                        "temperature": temperature,
                        "aggregator": PathAggregator(
                            form, top_k, temperature=temperature
                        ),
                    }
                )
    return result


def _rank_pool(
    paths_by_target: dict[str, list[dict[str, Any]]], aggregator: PathAggregator
) -> dict[str, list[dict[str, Any]]]:
    return rank_detailed_paths(
        paths_by_target,
        aggregator=aggregator,
        rrf_k=60,
        fusion_mode="weighted_rrf",
        direct_weight=1.0,
        evidence_weight=0.05,
        gated_evidence_min_paths=2,
        gated_evidence_quantile=0.75,
    )


def _recall(ranking: list[str], positives: set[str], k: int) -> float:
    return len(set(ranking[:k]) & positives) / len(positives)


def _mrr(ranking: list[str], positives: set[str], k: int) -> float:
    for rank, target_id in enumerate(ranking[:k], 1):
        if target_id in positives:
            return 1.0 / rank
    return 0.0


def _recovery_rows(
    paths: list[Path], split: str
) -> dict[tuple[str, str, str], set[int]]:
    result: dict[tuple[str, str, str], set[int]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if str(row.get("split")) != split:
                    continue
                evidence_id = str(row.get("evidence", {}).get("asset_id", ""))
                key = (
                    str(row["query_table_id"]),
                    str(row["target_table_id"]),
                    evidence_id,
                )
                result.setdefault(key, set()).add(int(row["query_row_id"]))
    return result


def _selected_evidence(
    target: dict[str, Any] | None, evidence_budget: int
) -> list[str]:
    if target is None:
        return []
    if "selected_evidence_ids" in target:
        return [
            str(evidence_id)
            for evidence_id in target["selected_evidence_ids"][:evidence_budget]
        ]
    paths = sorted(
        (path for path in target["paths"] if path["kind"] == "evidence"),
        key=lambda path: (-float(path["path_score"]), str(path["evidence_id"])),
    )
    return [str(path["evidence_id"]) for path in paths[:evidence_budget]]


def _empty_metrics() -> dict[str, Any]:
    return {
        "queries": 0,
        "per_query": {
            channel: {f"recall@{k}": [] for k in RECALL_KS}
            for channel in ("fused", "direct", "evidence")
        },
        "mrr@50": [],
        "valid_path": {
            f"valid_path_recall@{k},4": [0, 0] for k in RECALL_KS
        },
        "row_support": {
            f"row_support_coverage@{k},4": [] for k in RECALL_KS
        },
        "recoverable_row_support": {
            f"recoverable_row_coverage@{k},4": [] for k in RECALL_KS
        },
        "attribution": {
            f"evidence_only_discovery@{k}": 0 for k in RECALL_KS
        }
        | {f"fused_rescued@{k}": 0 for k in RECALL_KS}
        | {f"fused_lost@{k}": 0 for k in RECALL_KS},
        "positive_targets": 0,
        "path_count_correlation": Correlation(),
    }


def _accumulate_query(
    metrics: dict[str, Any],
    record: dict[str, Any],
    ranked: dict[str, list[dict[str, Any]]],
    recoveries: dict[tuple[str, str, str], set[int]],
    *,
    query_rows: int,
    evidence_budget: int,
) -> None:
    metrics["queries"] += 1
    query_id = str(record["query_id"])
    positives = set(str(value) for value in record["positive_target_ids"])
    metrics["positive_targets"] += len(positives)
    rankings = {
        channel: [str(row["target_id"]) for row in ranked[channel]]
        for channel in ("fused", "direct", "evidence")
    }
    targets = {
        str(row["target_id"]): row for row in ranked["fused"]
    }
    valid_evidence = {
        str(target_id): set(str(value) for value in evidence_ids)
        for target_id, evidence_ids in record.get(
            "positive_evidence_by_target", {}
        ).items()
    }
    for k in RECALL_KS:
        for channel in rankings:
            metrics["per_query"][channel][f"recall@{k}"].append(
                _recall(rankings[channel], positives, k)
            )
        direct_hits = set(rankings["direct"][:k]) & positives
        evidence_hits = set(rankings["evidence"][:k]) & positives
        fused_hits = set(rankings["fused"][:k]) & positives
        metrics["attribution"][f"evidence_only_discovery@{k}"] += len(
            evidence_hits - direct_hits
        )
        metrics["attribution"][f"fused_rescued@{k}"] += len(
            fused_hits - direct_hits
        )
        metrics["attribution"][f"fused_lost@{k}"] += len(
            direct_hits - fused_hits
        )
        fused_top = set(rankings["fused"][:k])
        for target_id, evidence_ids in valid_evidence.items():
            valid_key = f"valid_path_recall@{k},4"
            metrics["valid_path"][valid_key][1] += 1
            selected = (
                _selected_evidence(targets.get(target_id), evidence_budget)
                if target_id in fused_top
                else []
            )
            valid_selected = set(selected) & evidence_ids
            metrics["valid_path"][valid_key][0] += bool(valid_selected)
            recoverable_rows = set().union(
                *(
                    recoveries.get((query_id, target_id, evidence_id), set())
                    for evidence_id in evidence_ids
                )
            )
            supported_rows = set().union(
                *(
                    recoveries.get((query_id, target_id, evidence_id), set())
                    for evidence_id in valid_selected
                )
            )
            metrics["row_support"][f"row_support_coverage@{k},4"].append(
                len(supported_rows) / query_rows
            )
            metrics["recoverable_row_support"][
                f"recoverable_row_coverage@{k},4"
            ].append(
                len(supported_rows) / len(recoverable_rows)
                if recoverable_rows
                else 0.0
            )
    metrics["mrr@50"].append(_mrr(rankings["fused"], positives, 50))
    correlation: Correlation = metrics["path_count_correlation"]
    for target in ranked["evidence"]:
        evidence_paths = sum(
            path["kind"] == "evidence" for path in target["paths"]
        )
        correlation.add(float(evidence_paths), float(target["evidence_score"]))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _finalize(metrics: dict[str, Any]) -> dict[str, Any]:
    correlation: Correlation = metrics.pop("path_count_correlation")
    result = {
        "queries": metrics["queries"],
        "positive_targets": metrics["positive_targets"],
        "path_count_score_pearson": correlation.pearson(),
        "path_count_score_pairs": correlation.count,
        "mrr@50": _mean(metrics["mrr@50"]),
        "attribution": metrics["attribution"],
    }
    for channel, values in metrics["per_query"].items():
        result[channel] = {name: _mean(rows) for name, rows in values.items()}
        result[channel]["per_query"] = values
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


def _synthetic_stress(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for count in (1, 2, 4, 8, 16, 32, 64):
        pool = {
            "strong": [{"kind": "evidence", "path_score": 1.5}],
            "weak": [
                {
                    "kind": "evidence",
                    "evidence_id": f"weak-{index}",
                    "path_score": 1.0,
                }
                for index in range(count)
            ],
        }
        for config in configs:
            evidence = _rank_pool(pool, config["aggregator"])["evidence"]
            score = {str(value["target_id"]): value["evidence_score"] for value in evidence}
            rows.append(
                {
                    "paths": count,
                    "configuration": config["name"],
                    "strong_score": score["strong"],
                    "weak_score": score["weak"],
                    "weak_outranks_strong": score["weak"] > score["strong"],
                }
            )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# R10 Task B: {payload['system']} fixed-pool LSE/LME scan",
        "",
        "All rows use the same ANN path pool and fixed 100/20/20 retrieval budget. "
        "Unverified paths are not labeled invalid; the synthetic stress table is a "
        "formula test rather than a real-task gain.",
        "",
        "| Configuration | Fused R@10 | Evidence R@10 | ValidPath@10,4 | RowSupport@10,4 | RecoverableRow@10,4 | Path-count correlation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["results"].items():
        lines.append(
            f"| `{name}` | {row['fused']['recall@10']:.2%} | "
            f"{row['evidence']['recall@10']:.2%} | "
            f"{row['valid_path']['valid_path_recall@10,4']['value']:.2%} | "
            f"{row['row_support']['row_support_coverage@10,4']:.2%} | "
            f"{row['recoverable_row_support']['recoverable_row_coverage@10,4']:.2%} | "
            f"{row['path_count_score_pearson'] or 0.0:.4f} |"
        )
    lines.extend(["", "## Synthetic weak-path inversion", ""])
    for scope in ("full", "top4"):
        for family in ("lse", "lme"):
            rows = [
                row
                for row in payload["synthetic_stress"]
                if row["configuration"] == f"{scope}_{family}_t1"
            ]
            first = next(
                (row["paths"] for row in rows if row["weak_outranks_strong"]),
                None,
            )
            lines.append(f"- `{scope}_{family}_t1`: first inversion N={first}.")
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
        for line in progress(handle, desc="Scan fixed path pool", unit="query"):
            record = json.loads(line)
            paths = record["paths_by_target"]
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
        "path_score_space": metadata["path_score_space"],
        "retrieval_budget": metadata["retrieval_budget"],
        "evidence_bundle_budget": args.evidence_budget,
        "results": results,
        "synthetic_stress": _synthetic_stress(configs),
        "interpretation": {
            "real_paths": (
                "Path-count correlation and retrieval metrics; unverified evidence "
                "is not treated as a confirmed negative."
            ),
            "synthetic_paths": (
                "Formula behavior only; no claim about real retrieval quality."
            ),
        },
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
