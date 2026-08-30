#!/usr/bin/env python
"""Combine Task-J references with Task-L per-lake deployment evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.significance import paired_bootstrap_delta


SYSTEMS = ("raw", "student", "raw_ensemble", "student_ensemble")


def _read_system(path: Path, system: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("systems", {}).get(system)
    if result is None:
        raise ValueError(f"{path}: missing system {system!r}")
    return result


def _read_system_or_fallback(
    path: Path,
    system: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    return _read_system(path, system) if path.is_file() else fallback


def _view(result: dict[str, Any], channel: str | None = None) -> dict[str, Any]:
    per_query = result["metrics"]["per_query"]
    return per_query[channel] if channel else per_query


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
    candidate_view = _view(candidate, candidate_channel)
    reference_view = _view(reference, reference_channel)
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


def _ci(value: dict[str, Any]) -> str:
    return (
        f"{value['mean']:+.2%} "
        f"[{value['ci_low']:+.2%}, {value['ci_high']:+.2%}]"
    )


def _lake_spec(value: str) -> tuple[str, str, Path]:
    name, separator, raw_values = value.partition("=")
    values = raw_values.split(",") if separator else []
    if not name or len(values) != 2:
        raise argparse.ArgumentTypeError("lake must use NAME=LABEL,EVALUATION_DIR")
    return name, values[0], Path(values[1])


def _markdown(payload: dict[str, Any]) -> str:
    ks = payload["parameters"]["recall_ks"]
    max_k = max(ks)
    rows = [
        "# Task L: per-lake end-to-end deployment",
        "",
        "Student checkpoints are selected by the lake-local Task-K fused R@10 gate. "
        "Teacher rows rerank direct gamma*k pools; path rows use fused retrieval.",
    ]
    for lake in payload["lakes"].values():
        rows.extend(
            [
                "",
                f"## {lake['label']}",
                "",
                "| System | " + " | ".join(f"R@{k}" for k in ks)
                + f" | MRR@{max_k} | R@10 delta / 95% CI |",
                "| --- | " + " | ".join("---:" for _ in ks)
                + " | ---: | --- |",
            ]
        )
        for system in SYSTEMS:
            result = lake["systems"][system]
            metrics = result["metrics"]
            rows.append(
                f"| {result['label']} | "
                + " | ".join(f"{metrics[f'recall@{k}']:.2%}" for k in ks)
                + f" | {metrics[f'mrr@{max_k}']:.4f} | "
                + (_ci(result["comparison"]["recall@10"]) if "comparison" in result else "reference")
                + " |"
            )
        deployment = lake["deployment_gap"]["recall@10"]
        marginal = lake["rerank_marginal"]["recall@10"]
        rows.extend(
            [
                "",
                f"Deployment gap (Student+ensemble minus Raw+ensemble): {_ci(deployment)}.",
                f"Rerank marginal over Student direct retrieval: {_ci(marginal)}.",
                "Acceptance by CI lower bound >= -1pt: "
                + ("pass." if lake["acceptance_pass"] else "FAIL."),
                "",
                "| Index | GiB | Raw/student ratio |",
                "| --- | ---: | ---: |",
                f"| Raw | {lake['index_bytes']['raw'] / 2**30:.3f} | "
                f"{lake['index_bytes']['ratio']:.2f}x |",
                f"| Student | {lake['index_bytes']['student'] / 2**30:.3f} | 1.00x |",
            ]
        )
    rows.extend(
        [
            "",
            "## Decision",
            "",
            "Task L acceptance across both lakes: "
            + ("pass." if payload["all_lakes_pass"] else "FAIL; Task M is triggered."),
        ]
    )
    return "\n".join(rows) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    taskj = json.loads(args.taskj_metrics.read_text(encoding="utf-8"))
    taskk = json.loads(args.taskk_metrics.read_text(encoding="utf-8"))
    recall_ks = tuple(args.recall_ks)
    lakes: dict[str, Any] = {}
    for name, label, evaluation_dir in args.lake:
        taskj_lake = taskj["lakes"][name]
        systems = {
            "raw": taskj_lake["systems"]["raw"],
            "student": _read_system(
                evaluation_dir / "student" / "metrics.json", "student"
            ),
            "raw_ensemble": _read_system_or_fallback(
                evaluation_dir / "raw_ensemble" / "metrics.json",
                "raw_ensemble",
                taskj_lake["systems"]["raw_ensemble"],
            ),
            "student_ensemble": _read_system(
                evaluation_dir / "student_ensemble" / "metrics.json",
                "student_ensemble",
            ),
        }
        systems["student"]["comparison"] = _comparison(
            systems["student"], systems["raw"], recall_ks=recall_ks,
            iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
            candidate_channel="fused", reference_channel="fused",
        )
        systems["raw_ensemble"]["comparison"] = _comparison(
            systems["raw_ensemble"], systems["raw"], recall_ks=recall_ks,
            iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
            reference_channel="direct",
        )
        systems["student_ensemble"]["comparison"] = _comparison(
            systems["student_ensemble"], systems["raw"], recall_ks=recall_ks,
            iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
            reference_channel="direct",
        )
        deployment_gap = _comparison(
            systems["student_ensemble"], systems["raw_ensemble"],
            recall_ks=recall_ks, iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        rerank_marginal = _comparison(
            systems["student_ensemble"], systems["student"],
            recall_ks=recall_ks, iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed, reference_channel="direct",
        )
        selection_path = Path(taskk["selected_by_lake"][name]["selection"])
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        raw_bytes = int(taskj_lake["index_bytes"]["raw"])
        student_bytes = _directory_bytes(Path(selection["best_index"]))
        lakes[name] = {
            "label": label,
            "taskk_selection": taskk["selected_by_lake"][name],
            "systems": systems,
            "deployment_gap": deployment_gap,
            "rerank_marginal": rerank_marginal,
            "acceptance_pass": deployment_gap["recall@10"]["ci_low"] >= -0.01,
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
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "acceptance_ci_lower_bound": -0.01,
        },
        "lakes": lakes,
        "all_lakes_pass": all(lake["acceptance_pass"] for lake in lakes.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskj-metrics", type=Path, required=True)
    parser.add_argument("--taskk-metrics", type=Path, required=True)
    parser.add_argument("--lake", type=_lake_spec, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recall-ks", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
