#!/usr/bin/env python
"""Select the final per-lake Stage-1 r5 Student checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.selection import write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _task1_run(task1_root: Path, r4_root: Path, lake: str, tau: float) -> Path:
    if tau != 0.7:
        return task1_root / lake / f"tau_{tau:.1f}"
    if lake == "entitables":
        return r4_root / "taskM_entitables_teacher" / "student_kd0.3"
    return (
        r4_root
        / "taskM_entitables_teacher"
        / "per_lake"
        / "wdc"
        / "student_kd0.3"
    )


def _candidate(
    *,
    lake: str,
    tau: float,
    family: str,
    run_dir: Path,
    fused_recall: float,
    kd_teacher_checkpoint: Path | None,
    kd_teacher_label: str,
) -> dict[str, Any]:
    selection_path = run_dir / "student_path.pt.selection.json"
    selection = _read(selection_path)
    return {
        "lake": lake,
        "tau": tau,
        "family": family,
        "fused_recall@10": fused_recall,
        "run_directory": str(run_dir.resolve()),
        "student_selection": str(selection_path.resolve()),
        "student_checkpoint": selection["best_checkpoint"],
        "student_checkpoint_sha256": selection["best_checkpoint_sha256"],
        "kd_teacher_label": kd_teacher_label,
        "kd_teacher_checkpoint": (
            str(kd_teacher_checkpoint.resolve())
            if kd_teacher_checkpoint is not None
            else None
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    task1_root = Path(args.task1_root)
    task2_root = Path(args.task2_root)
    r4_root = Path(args.r4_root)
    task1 = _read(Path(args.task1_metrics))
    task2 = _read(Path(args.task2_metrics))
    current_teachers = {
        "entitables": r4_root
        / "taskM_entitables_teacher"
        / "checkpoints"
        / "teacher_path.pt",
        "wdc": r4_root
        / "taskM_entitables_teacher"
        / "per_lake"
        / "wdc"
        / "checkpoints"
        / "teacher_path.pt",
    }
    candidates: dict[str, list[dict[str, Any]]] = {
        "entitables": [],
        "wdc": [],
    }
    for row in task1["rows"]:
        lake = row["lake"]
        tau = float(row["tau"])
        teacher = None if tau == 0.0 else current_teachers[lake]
        candidates[lake].append(
            _candidate(
                lake=lake,
                tau=tau,
                family="task1_current_lake_teacher",
                run_dir=_task1_run(task1_root, r4_root, lake, tau),
                fused_recall=float(row["fused_recall@10"]),
                kd_teacher_checkpoint=teacher,
                kd_teacher_label=(
                    "frozen cosine only" if tau == 0.0 else "Task-M lake-local Teacher"
                ),
            )
        )

    repaired_teacher = task2_root / "checkpoints" / "teacher_path.pt"
    if task2["gate"]["passed"]:
        for tau in (0.3, 0.7, 1.0):
            run_dir = task2_root / f"wdc_tau_{tau:.1f}"
            evaluation_path = run_dir / "final_evaluation" / "metrics.json"
            if not evaluation_path.is_file():
                raise FileNotFoundError(
                    f"Task 2 passed but conditional tau run is missing: {evaluation_path}"
                )
            metrics = _read(evaluation_path)["systems"]["student"]["metrics"]
            candidates["wdc"].append(
                _candidate(
                    lake="wdc",
                    tau=tau,
                    family="task2_mixed_negative_teacher",
                    run_dir=run_dir,
                    fused_recall=float(metrics["recall@10"]),
                    kd_teacher_checkpoint=repaired_teacher,
                    kd_teacher_label="Task-2 mixed-negative WDC Teacher",
                )
            )

    selected = {
        lake: max(
            values,
            key=lambda value: (
                value["fused_recall@10"],
                -abs(value["tau"] - 0.7),
            ),
        )
        for lake, values in candidates.items()
    }
    selected["entitables"]["online_teacher_candidate"] = str(
        current_teachers["entitables"].resolve()
    )
    selected["entitables"]["online_teacher_provenance"] = (
        "Task-M EntiTables Teacher"
    )
    selected["wdc"]["online_teacher_candidate"] = (
        str(repaired_teacher.resolve()) if task2["gate"]["passed"] else None
    )
    selected["wdc"]["online_teacher_provenance"] = (
        "Task-2 mixed-negative WDC Teacher candidate"
        if task2["gate"]["passed"]
        else "no-op (Task-2 Teacher gate failed)"
    )
    payload = {
        "format_version": 1,
        "selection_metric": "fused recall@10",
        "tie_break": "closest tau to deployed r4 tau=0.7",
        "candidates": candidates,
        "selected": selected,
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task1-metrics", required=True)
    parser.add_argument("--task1-root", required=True)
    parser.add_argument("--task2-metrics", required=True)
    parser.add_argument("--task2-root", required=True)
    parser.add_argument("--r4-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
