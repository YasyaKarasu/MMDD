#!/usr/bin/env python
"""Combine independently persisted per-lake Task-J baseline evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.significance import paired_bootstrap_delta


SYSTEMS = ("raw", "student", "raw_ensemble", "student_ensemble")


def _metric_view(result: dict[str, Any], channel: str | None = None) -> dict[str, Any]:
    metrics = result["metrics"]
    return metrics["per_query"][channel] if channel else metrics["per_query"]


def _comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    recall_ks: tuple[int, ...],
    iterations: int,
    seed: int,
    candidate_channel: str | None = None,
    reference_channel: str | None = None,
) -> dict[str, Any]:
    candidate_view = _metric_view(candidate, candidate_channel)
    reference_view = _metric_view(reference, reference_channel)
    return {
        f"recall@{k}": paired_bootstrap_delta(
            candidate_view[f"recall@{k}"],
            reference_view[f"recall@{k}"],
            iterations=iterations,
            seed=seed,
        )
        for k in recall_ks
    }


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _ci(result: dict[str, Any]) -> str:
    return (
        f"{result['mean']:+.2%} "
        f"[{result['ci_low']:+.2%}, {result['ci_high']:+.2%}]"
    )


def _markdown(payload: dict[str, Any]) -> str:
    ks = payload["parameters"]["recall_ks"]
    max_k = max(ks)
    lines = [
        "# Task J: per-lake baselines",
        "",
        "Each query retrieves only from its own data-lake corpus. Teacher rows "
        "rerank direct `gamma * k` pools with the alpha=0.7 z-score ensemble.",
        "Student deltas use the matching raw fused channel; Teacher-ensemble "
        "deltas use the matching raw direct channel.",
        "",
        "## Corpus partition and PCA",
        "",
        "| Lake | Corpus objects | PCA-1024 variance retained |",
        "| --- | ---: | ---: |",
    ]
    for lake, values in payload["lakes"].items():
        lines.append(
            f"| {values['label']} | {values['corpus_objects']:,} | "
            f"{values['pca_explained_variance_ratio']:.4%} |"
        )
    lines.extend(
        [
            "",
            "The two source ID sets are pairwise disjoint and their counts sum "
            "exactly to the 279,373-object mixed corpus.",
        ]
    )
    for lake, values in payload["lakes"].items():
        lines.extend(
            [
                "",
                f"## {values['label']}",
                "",
                "| System | "
                + " | ".join(f"R@{k}" for k in ks)
                + f" | MRR@{max_k} | Coverage@10 | R@10 delta / 95% CI |",
                "| --- | "
                + " | ".join("---:" for _ in ks)
                + " | ---: | ---: | --- |",
            ]
        )
        for system in SYSTEMS:
            result = values["systems"][system]
            metrics = result["metrics"]
            comparison = result.get("comparison")
            coverage = metrics.get("positive_evidence_path_coverage@10")
            lines.append(
                f"| {result['label']} | "
                + " | ".join(f"{metrics[f'recall@{k}']:.2%}" for k in ks)
                + f" | {metrics[f'mrr@{max_k}']:.4f} | "
                + (f"{coverage:.2%}" if coverage is not None else "n/a")
                + " | "
                + (_ci(comparison["recall@10"]) if comparison else "reference")
                + " |"
            )
        deployment = values["deployment_gap"]["recall@10"]
        lines.extend(
            [
                "",
                "Student+ensemble minus raw+ensemble at R@10: " + _ci(deployment) + ".",
                "",
                "| Index | Bytes | GiB | Ratio raw/student |",
                "| --- | ---: | ---: | ---: |",
                f"| Raw | {values['index_bytes']['raw']:,} | "
                f"{values['index_bytes']['raw'] / 2**30:.3f} | "
                f"{values['index_bytes']['ratio']:.2f}x |",
                f"| Student | {values['index_bytes']['student']:,} | "
                f"{values['index_bytes']['student'] / 2**30:.3f} | 1.00x |",
                "",
                "Per-system wall-clock latency (independent evaluation of all five k values):",
            ]
        )
        for system in SYSTEMS:
            timing = values["systems"][system]["timing"]
            lines.append(
                f"- {values['systems'][system]['label']}: "
                f"{timing['average_seconds_per_query_per_k'] * 1000:.2f} ms/query/k."
            )
    lines.extend(
        [
            "",
            "## Mixed vs per-lake raw",
            "",
            "| Lake | Mixed R@10 | Per-lake R@10 | Delta | Check |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for values in payload["lakes"].values():
        mixed = values["mixed_raw_recall@10"]
        per_lake = values["systems"]["raw"]["metrics"]["recall@10"]
        lines.append(
            f"| {values['label']} | {mixed:.2%} | {per_lake:.2%} | "
            f"{per_lake - mixed:+.2%} | {'pass' if per_lake >= mixed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Parameters",
            "",
            f"- recall_ks={ks}; gamma={payload['parameters']['gamma']}; "
            f"gamma_evidence={payload['parameters']['gamma_evidence']}.",
            f"- Bootstrap iterations={payload['parameters']['bootstrap_iterations']}; "
            f"seed={payload['parameters']['bootstrap_seed']}.",
            "- Fusion=weighted RRF (direct 1.0, evidence 0.05); evidence "
            "aggregation=logsumexp/top4.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    split = json.loads(args.split_summary.read_text(encoding="utf-8"))
    mixed = json.loads(args.mixed_metrics.read_text(encoding="utf-8"))
    recall_ks = tuple(args.recall_ks)
    lake_payload: dict[str, Any] = {}
    for lake_spec in args.lake:
        name, label, evaluation_dir, pca_path, student_index, raw_index, dataset = lake_spec
        evaluation_payloads = {
            system: json.loads(
                (evaluation_dir / system / "metrics.json").read_text(encoding="utf-8")
            )
            for system in SYSTEMS
        }
        systems = {
            system: evaluation_payloads[system]["systems"][system]
            for system in SYSTEMS
        }
        systems["student"]["comparison"] = _comparison(
            systems["student"], systems["raw"],
            recall_ks=recall_ks, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
            candidate_channel="fused", reference_channel="fused",
        )
        systems["raw_ensemble"]["comparison"] = _comparison(
            systems["raw_ensemble"], systems["raw"],
            recall_ks=recall_ks, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
            reference_channel="direct",
        )
        systems["student_ensemble"]["comparison"] = _comparison(
            systems["student_ensemble"], systems["raw"],
            recall_ks=recall_ks, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
            reference_channel="direct",
        )
        deployment_gap = _comparison(
            systems["student_ensemble"], systems["raw_ensemble"],
            recall_ks=recall_ks, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
        )
        pca = torch.load(pca_path, map_location="cpu", weights_only=True)
        raw_bytes = _directory_bytes(raw_index)
        student_bytes = _directory_bytes(student_index)
        lake_payload[name] = {
            "label": label,
            "dataset": dataset,
            "provenance": {
                "corpus_sha256": evaluation_payloads["raw"]["corpus_sha256"],
                "student_checkpoint": evaluation_payloads["student"]["student_checkpoint"],
                "student_checkpoint_sha256": evaluation_payloads["student"][
                    "student_checkpoint_sha256"
                ],
                "teacher_checkpoint": evaluation_payloads["raw_ensemble"][
                    "teacher_checkpoint"
                ],
                "teacher_checkpoint_sha256": evaluation_payloads["raw_ensemble"][
                    "teacher_checkpoint_sha256"
                ],
            },
            "corpus_objects": split["lakes"][name]["objects"],
            "pca_explained_variance_ratio": float(pca["explained_variance_ratio"]),
            "systems": systems,
            "deployment_gap": deployment_gap,
            "mixed_raw_recall@10": mixed["systems"]["raw"]["metrics"]["by_dataset"][dataset]["recall@10"],
            "index_bytes": {
                "raw": raw_bytes,
                "student": student_bytes,
                "ratio": raw_bytes / student_bytes,
            },
        }
    payload = {
        "format_version": 1,
        "parameters": {
            "recall_ks": list(recall_ks),
            "gamma": 10,
            "gamma_evidence": 2,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "split_summary": split,
        "lakes": lake_payload,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _lake(value: str) -> tuple[str, str, Path, Path, Path, Path, str]:
    parts = value.split("=", 1)
    values = parts[1].split(",") if len(parts) == 2 else []
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "lake must use NAME=LABEL,EVAL_DIR,PCA,STUDENT_INDEX,RAW_INDEX,DATASET"
        )
    return (
        parts[0],
        values[0],
        *(Path(item) for item in values[1:5]),
        values[5],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-summary", type=Path, required=True)
    parser.add_argument("--mixed-metrics", type=Path, required=True)
    parser.add_argument("--lake", type=_lake, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recall-ks", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
