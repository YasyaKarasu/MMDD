#!/usr/bin/env python
"""Summarize the round-3 ensemble-KD decision from completed histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.selection import write_json


def _epoch_view(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["dev_retrieval"]
    gate = record["gate"]
    constraint = gate["per_dataset"][0]
    bootstrap = constraint.get("bootstrap")
    wdc = metrics["by_dataset"][constraint["dataset"]]
    return {
        "epoch": int(record["epoch"]),
        "overall_recall@10": float(metrics["recall@10"]),
        "overall_direct_recall@10": float(metrics["direct"]["recall@10"]),
        "wdc_direct_recall@10": float(wdc["direct"]["recall@10"]),
        "gate_eligible": bool(gate["eligible"]),
        "gate_improved": bool(gate["improved"]),
        "wdc_delta": None if bootstrap is None else float(bootstrap["mean"]),
        "wdc_ci_low": None if bootstrap is None else float(bootstrap["ci_low"]),
        "wdc_ci_high": None if bootstrap is None else float(bootstrap["ci_high"]),
    }


def _run_view(weight: float, history_path: Path) -> dict[str, Any]:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [_epoch_view(record) for record in history["epochs"]]
    epoch_zero = epochs[0]
    best_epoch = int(history["best_epoch"])
    selected = next(record for record in epochs if record["epoch"] == best_epoch)
    peak = max(epochs[1:], key=lambda record: record["overall_recall@10"])
    success = (
        best_epoch > 0
        and selected["gate_eligible"]
        and selected["overall_recall@10"] > epoch_zero["overall_recall@10"]
    )
    selection_path = history_path.with_name("student_path.pt.selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    return {
        "distillation_weight": weight,
        "history": str(history_path.resolve()),
        "selection": str(selection_path.resolve()),
        "best_checkpoint": selection["best_checkpoint"],
        "best_checkpoint_sha256": selection["best_checkpoint_sha256"],
        "best_epoch": best_epoch,
        "stop_reason": history["stop_reason"],
        "gate_unsatisfied": bool(history["gate_unsatisfied"]),
        "teacher_logit_cache_hits": int(history["teacher_logit_cache_hits"]),
        "teacher_logit_caches": history["teacher_logit_caches"],
        "epoch_zero": epoch_zero,
        "selected": selected,
        "peak_trained": peak,
        "success": success,
        "epochs": epochs,
    }


def _ci(record: dict[str, Any]) -> str:
    if record["wdc_ci_low"] is None:
        return "n/a"
    return f"{record['wdc_delta']:+.2%} [{record['wdc_ci_low']:+.2%}, {record['wdc_ci_high']:+.2%}]"


def run(args: argparse.Namespace) -> None:
    if len(args.history) != len(args.weight):
        raise ValueError("--history and --weight must align")
    runs = [
        _run_view(weight, Path(history))
        for weight, history in zip(args.weight, args.history)
    ]
    accepted = [run for run in runs if run["success"]]
    winner = (
        max(accepted, key=lambda run: run["selected"]["overall_recall@10"])
        if accepted
        else None
    )
    fallback_selection_path = Path(args.fallback_selection)
    fallback = json.loads(fallback_selection_path.read_text(encoding="utf-8"))
    final_selection = winner["selection"] if winner is not None else str(fallback_selection_path.resolve())
    final_checkpoint = winner["best_checkpoint"] if winner is not None else fallback["best_checkpoint"]
    payload = {
        "format_version": 1,
        "teacher_ensemble_alpha": 0.7,
        "gate_tolerance": 0.02,
        "runs": runs,
        "accepted_weight": None if winner is None else winner["distillation_weight"],
        "kd_direction_closed": winner is None,
        "final_student_selection": final_selection,
        "final_student_checkpoint": final_checkpoint,
        "final_student_checkpoint_sha256": checkpoint_fingerprint(Path(final_checkpoint)),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", payload)
    write_json(
        output_dir / "final_student.json",
        {
            "selection": final_selection,
            "checkpoint": final_checkpoint,
            "checkpoint_sha256": payload["final_student_checkpoint_sha256"],
            "accepted_weight": payload["accepted_weight"],
        },
    )

    lines = [
        "# Task H: ensemble-KD",
        "",
        "Targets are `0.7 * z(Teacher) + 0.3 * z(frozen cosine)` within each direct/evidence candidate list. Training uses fresh PCA-1024, path-only in-batch negatives, mu=0.1, dataset alpha=0, gamma=10, and gamma_e=2.",
        "",
        "| KD weight | epoch-0 R@10 | peak trained R@10 | peak epoch | peak WDC direct R@10 | peak WDC delta / 95% CI | selected epoch | accepted |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for result in runs:
        peak = result["peak_trained"]
        lines.append(
            f"| {result['distillation_weight']:g} | {result['epoch_zero']['overall_recall@10']:.2%} | "
            f"{peak['overall_recall@10']:.2%} | {peak['epoch']} | {peak['wdc_direct_recall@10']:.2%} | "
            f"{_ci(peak)} | {result['best_epoch']} | {'yes' if result['success'] else 'no'} |"
        )
    lines += ["", "## Epoch details", ""]
    for result in runs:
        lines += [
            f"### KD weight {result['distillation_weight']:g}",
            "",
            "| Epoch | overall R@10 | WDC direct R@10 | WDC delta / 95% CI | Gate |",
            "| ---: | ---: | ---: | --- | --- |",
        ]
        for epoch in result["epochs"]:
            lines.append(
                f"| {epoch['epoch']} | {epoch['overall_recall@10']:.2%} | "
                f"{epoch['wdc_direct_recall@10']:.2%} | {_ci(epoch)} | "
                f"{'pass' if epoch['gate_eligible'] else 'fail'} |"
            )
        lines.append("")
    if winner is None:
        lines += [
            "## Decision",
            "",
            "Both ensemble-KD weights improve the overall point estimate only by moving along the same EntiTables/WDC trade-off; no trained epoch passes the WDC CI gate. Together with the R2 Teacher-logit KD result, the KD direction is closed for this formulation.",
            "",
            f"Final Student remains the R2 epoch-0 checkpoint `{final_checkpoint}` (`{payload['final_student_checkpoint_sha256']}`).",
        ]
    else:
        lines += [
            "## Decision",
            "",
            f"Accepted KD weight {winner['distillation_weight']:g}; final checkpoint `{final_checkpoint}`.",
        ]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="append", required=True)
    parser.add_argument("--weight", action="append", type=float, required=True)
    parser.add_argument("--fallback-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
