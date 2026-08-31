#!/usr/bin/env python
"""Summarize completed r7 Task-Q joint-selection evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _rows(task_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows = []
    knockout_rows = []
    for path in sorted(task_root.glob("*/**/metrics.json")):
        payload = _load(path)
        run = str(path.parent.relative_to(task_root / str(payload["lake"])))
        selected_name = str(payload["selected"])
        for name, record in payload["metrics"].items():
            metrics = record["metrics"]
            delta = record["delta_vs_r5"]
            row = {
                "lake": payload["lake"],
                "run": run,
                "configuration": name,
                "selected": name == selected_name,
                "fused_recall@10": metrics["recall@10"],
                "direct_recall@10": metrics["direct"]["recall@10"],
                "evidence_recall@10": metrics["evidence"]["recall@10"],
                "coverage@10": metrics[
                    "positive_evidence_path_coverage@10"
                ],
                "mrr@50": metrics["mrr@50"],
                "delta_vs_r5": delta["delta_mean"],
                "ci_low_vs_r5": delta["ci_low"],
                "ci_high_vs_r5": delta["ci_high"],
                "r5_point_anchor_satisfied": delta["delta_mean"] >= -1e-12,
                "r5_ci_gate_satisfied": delta["ci_low"] >= -0.02,
            }
            knockout_rows.append(row)
            if name == selected_name:
                selected_rows.append(row)
    return selected_rows, knockout_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage-1 r7 Task Q: joint configuration selection",
        "",
        "The current artifacts cover the four retrieval-side combinations "
        "(2 fusion × 2 path aggregation). Relation weight and KD tau are fixed "
        "by each training checkpoint, so the planned 16-way joint sweep is not "
        "claimed complete.",
        "",
        "| Lake | Student run | Selected configuration | Fused R@10 | Δ vs r5 / 95% CI | Evidence R@10 | Coverage@10 | r5 point / CI gates |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda value: (value["lake"], value["run"])):
        lines.append(
            f"| {row['lake']} | {row['run']} | {row['configuration']} | "
            f"{row['fused_recall@10']:.2%} | {row['delta_vs_r5']:+.2%} "
            f"[{row['ci_low_vs_r5']:+.2%}, {row['ci_high_vs_r5']:+.2%}] | "
            f"{row['evidence_recall@10']:.2%} | {row['coverage@10']:.2%} | "
            f"{row['r5_point_anchor_satisfied']} / "
            f"{row['r5_ci_gate_satisfied']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    task_root = args.output_root / "taskQ_joint_selection"
    task_root.mkdir(parents=True, exist_ok=True)
    selected_rows, knockout_rows = _rows(task_root)
    _write_csv(task_root / "knockout.csv", knockout_rows)
    _write_results(task_root / "RESULTS.md", selected_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r7_20260831"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
