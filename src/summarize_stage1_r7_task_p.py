#!/usr/bin/env python
"""Summarize completed r7 Task-P runs and export residual trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.significance import paired_bootstrap_delta

R5_FUSED_ANCHORS = {"entitables": 0.3774, "wdc": 0.6436}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _ci_vs_raw(
    metrics: dict[str, Any],
    channel: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    student = metrics["per_query"][channel]["recall@10"]
    raw = metrics["raw_embedding"]["per_query"][channel]["recall@10"]
    return paired_bootstrap_delta(
        student,
        raw,
        iterations=iterations,
        seed=seed,
    )


def _metric(metrics: dict[str, Any], channel: str) -> float:
    if channel == "fused":
        return float(metrics["recall@10"])
    return float(metrics[channel]["recall@10"])


def _run_rows(
    task_root: Path,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    trajectories = []
    for history_path in sorted(
        task_root.glob("*/k_*/mu_*/student_path.pt.history.json")
    ):
        history = _load(history_path)
        config = history["student_config"]
        run_dir = history_path.parent
        lake = run_dir.parents[1].name
        rank = int(config["relation_rank"])
        anchor = float(history["anchor_weight"])
        best_epoch = int(history["best_epoch"])
        epochs = history["epochs"]
        epoch_zero = next(row for row in epochs if int(row["epoch"]) == 0)
        direct_epoch_zero = _metric(epoch_zero["dev_retrieval"], "direct")
        direct_after = [
            _metric(row["dev_retrieval"], "direct")
            for row in epochs
            if int(row["epoch"]) > 0
        ]
        for row in epochs:
            metrics = row["dev_retrieval"]
            record = {
                "lake": lake,
                "rank": rank,
                "anchor_weight": anchor,
                "epoch": int(row["epoch"]),
                "is_best": int(row["epoch"]) == best_epoch,
                "fused_recall@10": _metric(metrics, "fused"),
                "direct_recall@10": _metric(metrics, "direct"),
                "evidence_recall@10": _metric(metrics, "evidence"),
                "coverage@10": float(
                    metrics["positive_evidence_path_coverage@10"]
                ),
                "mrr@50": float(metrics["mrr@50"]),
            }
            for channel in ("fused", "direct", "evidence"):
                ci = _ci_vs_raw(
                    metrics,
                    channel,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
                record[f"{channel}_delta_vs_raw"] = ci["delta_mean"]
                record[f"{channel}_ci_low_vs_raw"] = ci["ci_low"]
                record[f"{channel}_ci_high_vs_raw"] = ci["ci_high"]
            record.update(
                {
                    f"residual_frobenius_{key}": float(value)
                    for key, value in sorted(row["relation_drift"].items())
                }
            )
            trajectories.append(record)

        best = next(row for row in trajectories if row["lake"] == lake and row["rank"] == rank and row["anchor_weight"] == anchor and row["epoch"] == best_epoch)
        summaries.append(
            {
                **best,
                "epochs_evaluated": len(epochs),
                "direct_epoch_zero": direct_epoch_zero,
                "minimum_direct_after_epoch_zero": (
                    min(direct_after) if direct_after else direct_epoch_zero
                ),
                "direct_never_below_epoch_zero": all(
                    value >= direct_epoch_zero for value in direct_after
                ),
                "meets_r5_fused_anchor": (
                    best["fused_recall@10"] >= R5_FUSED_ANCHORS[lake]
                ),
                "run_complete": (run_dir / "run_manifest.json").is_file(),
            }
        )
    return summaries, trajectories


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage-1 r7 Task P: low-rank residual Student",
        "",
        "R@10 uses the fixed r6 retrieval pool and weighted-RRF training gate. "
        "Confidence intervals compare each epoch with the raw embedding system "
        "using paired bootstrap (10,000 iterations, seed 13).",
        "",
        "| Lake | k | μ | Best epoch | Fused R@10 | Direct R@10 | Evidence R@10 | Coverage@10 | Fused Δ vs raw / 95% CI | Direct never below e0 | ≥ r5 anchor | Complete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in sorted(
        rows, key=lambda value: (value["lake"], value["rank"], value["anchor_weight"])
    ):
        lines.append(
            f"| {row['lake']} | {row['rank']} | {row['anchor_weight']:g} | "
            f"{row['epoch']} | {row['fused_recall@10']:.2%} | "
            f"{row['direct_recall@10']:.2%} | {row['evidence_recall@10']:.2%} | "
            f"{row['coverage@10']:.2%} | {row['fused_delta_vs_raw']:+.2%} "
            f"[{row['fused_ci_low_vs_raw']:+.2%}, {row['fused_ci_high_vs_raw']:+.2%}] | "
            f"{row['direct_never_below_epoch_zero']} | "
            f"{row['meets_r5_fused_anchor']} | {row['run_complete']} |"
        )
    lines.extend(
        [
            "",
            "For d=1024, full-R has 9,437,184 relation parameters. Low-rank "
            "uses 18,432×k relation parameters (32× fewer at k=16, 8× at "
            "k=64, and 2× at k=256).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    task_root = args.output_root / "taskP_lowrank"
    task_root.mkdir(parents=True, exist_ok=True)
    summaries, trajectories = _run_rows(
        task_root,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_csv(task_root / "residual_trajectory.csv", trajectories)
    _write_csv(task_root / "k_sensitivity.csv", summaries)
    _write_results(task_root / "RESULTS.md", summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r7_20260831"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    values = parser.parse_args()
    if values.bootstrap_iterations <= 0:
        parser.error("--bootstrap-iterations must be positive")
    return values


if __name__ == "__main__":
    run(parse_args())
