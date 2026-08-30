#!/usr/bin/env python
"""Summarize one lake-specific Teacher retrain and KD rerun from Task M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(value: dict[str, Any]) -> str:
    return (
        f"{value['delta_mean']:+.2%} "
        f"[{value['ci95_low']:+.2%}, {value['ci95_high']:+.2%}]"
    )


def _markdown(payload: dict[str, Any]) -> str:
    teacher = payload["teacher"]
    student = payload["student_rerun"]
    data = payload["data"]
    return "\n".join(
        [
            f"# Task M: {payload['label']}-only Teacher retrain and KD rerun",
            "",
            f"Retrieval-aligned records: {data['edge_records']} edge and "
            f"{data['target_records']} path lists; missing Teacher features: "
            f"{data['missing_teacher_objects']}.",
            "",
            "## Teacher rerank gate",
            "",
            "| Raw R@10 | Teacher R@10 | Delta | Required | Result |",
            "| ---: | ---: | ---: | ---: | --- |",
            f"| {teacher['raw_recall@10']:.2%} | "
            f"{teacher['reranked_recall@10']:.2%} | "
            f"{teacher['delta']:+.2%} | >{teacher['required_delta']:.0%} | "
            f"{'pass' if teacher['gate_pass'] else 'FAIL'} |",
            "",
            "## KD=0.3 rerun",
            "",
            "| System | Direct R@10 | Fused R@10 | Epoch | Gate |",
            "| --- | ---: | ---: | ---: | --- |",
            f"| Raw | {payload['references']['raw_direct_recall@10']:.2%} | "
            f"{payload['references']['raw_fused_recall@10']:.2%} | - | reference |",
            f"| {payload['label']} supervised | "
            f"{payload['references']['supervised_direct_recall@10']:.2%} | "
            f"{payload['references']['supervised_fused_recall@10']:.2%} | "
            f"{payload['references']['supervised_epoch']} | "
            f"{'fallback' if payload['references']['supervised_gate_unsatisfied'] else 'pass'} |",
            f"| Task-M KD 0.3 | {student['direct_recall@10']:.2%} | "
            f"{student['fused_recall@10']:.2%} | {student['best_epoch']} | "
            f"{'fallback' if student['gate_unsatisfied'] else 'pass'} |",
            "",
            "Task-M KD direct delta vs raw: "
            + _ci(student["direct_recall@10_vs_raw"])
            + ".",
            "Strict `KD > supervised > raw` chain: "
            + ("established." if payload["distillation_chain_established"] else "not established."),
            "",
            f"Teacher checkpoint: `{teacher['checkpoint']}`",
            f"Teacher SHA-256: `{teacher['checkpoint_sha256']}`",
            f"Student checkpoint: `{student['checkpoint']}`",
            f"Student SHA-256: `{student['checkpoint_sha256']}`",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    lake = getattr(args, "lake", "entitables")
    label = getattr(args, "label", "EntiTables")
    supervised_run = getattr(args, "supervised_run", f"{lake}_supervised")
    taskk = _read(args.taskk_metrics)
    preflight = _read(args.preflight)
    teacher_selection = _read(args.teacher_selection)
    diagnostic = _read(args.diagnostic)
    student_history = _read(args.student_history)
    student_selection = _read(args.student_selection)
    final_evaluation = _read(args.student_evaluation)["systems"]["student"]

    supervised = taskk["runs"][supervised_run]
    raw_direct = float(supervised["raw"]["direct_recall@10"])
    raw_fused = float(supervised["raw"]["fused_recall@10"])
    teacher_raw = float(diagnostic["raw_direct"]["recall@10"])
    teacher_reranked = float(diagnostic["teacher_reranked"]["recall@10"])
    teacher_delta = teacher_reranked - teacher_raw
    best_epoch = int(student_selection["best_epoch"])
    best_record = next(
        record
        for record in student_history["epochs"]
        if int(record["epoch"]) == best_epoch
    )
    retrieval = best_record["dev_retrieval"]
    gate = best_record["gate"]["per_dataset"][0]["bootstrap"]
    student_direct = float(retrieval["direct"]["recall@10"])
    student_fused = float(retrieval["recall@10"])
    supervised_direct = float(supervised["selected"]["direct_recall@10"])
    chain = student_direct > supervised_direct > raw_direct

    payload = {
        "format_version": 1,
        "lake": lake,
        "label": label,
        "trigger": {
            "taskk_required": bool(taskk["decisions"]["task_m_required"]),
            "taskk_distillation_chain": bool(
                taskk["decisions"]["entitables_distillation_chain"]
            ),
        },
        "data": preflight,
        "teacher": {
            "checkpoint": teacher_selection["best_checkpoint"],
            "checkpoint_sha256": teacher_selection["best_checkpoint_sha256"],
            "best_epoch": int(teacher_selection["best_epoch"]),
            "raw_recall@10": teacher_raw,
            "reranked_recall@10": teacher_reranked,
            "delta": teacher_delta,
            "required_delta": args.teacher_required_delta,
            "gate_pass": teacher_delta > args.teacher_required_delta,
            "ensemble_alpha": args.teacher_ensemble_alpha,
        },
        "references": {
            "raw_direct_recall@10": raw_direct,
            "raw_fused_recall@10": raw_fused,
            "supervised_direct_recall@10": supervised_direct,
            "supervised_fused_recall@10": float(
                supervised["selected"]["fused_recall@10"]
            ),
            "supervised_epoch": int(supervised["best_epoch"]),
            "supervised_gate_unsatisfied": bool(
                supervised.get("gate_unsatisfied", False)
            ),
        },
        "student_rerun": {
            "run_name": args.run_name,
            "directory": str(args.student_selection.parent.resolve()),
            "selection": str(args.student_selection.resolve()),
            "checkpoint": student_selection["best_checkpoint"],
            "checkpoint_sha256": student_selection["best_checkpoint_sha256"],
            "best_epoch": best_epoch,
            "gate_unsatisfied": bool(student_selection["gate_unsatisfied"]),
            "direct_recall@10": student_direct,
            "fused_recall@10": student_fused,
            "direct_recall@10_vs_raw": gate,
            "final_evaluation": final_evaluation,
        },
        "distillation_chain_established": chain,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    if lake == "entitables" and args.output_dir.name != "entitables":
        mirror = args.output_dir / "per_lake" / "entitables"
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (mirror / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskk-metrics", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--teacher-selection", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--student-history", type=Path, required=True)
    parser.add_argument("--student-selection", type=Path, required=True)
    parser.add_argument("--student-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lake", default="entitables")
    parser.add_argument("--label", default="EntiTables")
    parser.add_argument("--supervised-run", default="entitables_supervised")
    parser.add_argument("--run-name", default="entitables_kd0.3_taskm")
    parser.add_argument("--teacher-required-delta", type=float, default=0.03)
    parser.add_argument("--teacher-ensemble-alpha", type=float, default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
