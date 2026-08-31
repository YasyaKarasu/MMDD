#!/usr/bin/env python
"""Summarize the r7 Task-R KD-temperature sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.significance import paired_bootstrap_delta


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def run(root: Path, *, iterations: int, seed: int) -> None:
    rows: list[dict[str, Any]] = []
    metrics_by_temp: dict[float, dict[str, Any]] = {}
    for metrics_path in sorted(
        root.glob("taskP_lowrank/*/*/t_*/mu_*/final_evaluation/metrics.json")
    ):
        metrics = _load(metrics_path)
        manifest = _load(metrics_path.parents[1] / "run_manifest.json")
        temperature = float(manifest["kd_temperature"])
        student = metrics["systems"]["student"]["metrics"]
        metrics_by_temp[temperature] = student
        rows.append(
            {
                "temperature": temperature,
                "best_fused_recall@10": student["recall@10"],
                "best_direct_recall@10": student["direct"]["recall@10"],
                "best_evidence_recall@10": student["evidence"]["recall@10"],
                "coverage@10": student["positive_evidence_path_coverage@10"],
                "mrr@50": student["mrr@50"],
            }
        )
    baseline_path = Path("work/stage1_optimization_r7_20260831/taskP_lowrank/wdc/k_256/mu_0/final_evaluation/metrics.json")
    baseline = _load(baseline_path)["systems"]["student"]["metrics"]
    baseline_per_query = baseline["per_query"]["fused"]["recall@10"]
    for row in rows:
        student = metrics_by_temp[row["temperature"]]
        delta = paired_bootstrap_delta(
            student["per_query"]["fused"]["recall@10"],
            baseline_per_query,
            iterations=iterations,
            seed=seed,
        )
        row.update(
            {
                "delta_vs_t1": delta["delta_mean"],
                "ci_low_vs_t1": delta["ci_low"],
                "ci_high_vs_t1": delta["ci_high"],
            }
        )
    rows.sort(key=lambda row: row["temperature"])
    output = root
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage-1 r7 Task R: KD temperature sweep",
        "",
        "WDC low-rank k=256, μ=0; deltas are paired bootstrap comparisons with the existing T=1.0 run (seed 13).",
        "",
        "| KD temperature | Fused R@10 | Δ vs T=1 / 95% CI | Direct R@10 | Evidence R@10 | Coverage@10 |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['temperature']:g} | {row['best_fused_recall@10']:.2%} | "
            f"{row['delta_vs_t1']:+.2%} [{row['ci_low_vs_t1']:+.2%}, {row['ci_high_vs_t1']:+.2%}] | "
            f"{row['best_direct_recall@10']:.2%} | {row['best_evidence_recall@10']:.2%} | "
            f"{row['coverage@10']:.2%} |"
        )
    lines.append("")
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run(
        Path("work/stage1_optimization_r7_20260831/taskR_kd_temperature"),
        iterations=10_000,
        seed=13,
    )
