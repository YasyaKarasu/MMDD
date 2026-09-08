#!/usr/bin/env python
"""Evaluate Task F2 with several retained tokens per table schema/row."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mmdd_stage1.experiment_process import run_logged_subprocess
from mmdd_stage1.significance import paired_bootstrap_delta


def _lake_inputs(root: Path, lake: str) -> dict[str, Path]:
    r4 = root / "work/stage1_optimization_r4_20260829"
    r6 = root / "work/stage1_optimization_r6_20260830"
    if lake == "entitables":
        return {
            "selection": r6
            / "taskE_modality_balance/entitables/relation_weight_2/student_path.pt.selection.json",
            "baseline_student": r6
            / "taskE_modality_balance/entitables/relation_weight_2/final_evaluation/metrics.json",
            "baseline_raw": r4
            / "taskJ_per_lake_baselines/per_lake/entitables/raw/metrics.json",
            "baseline_teacher": r4
            / "taskJ_per_lake_baselines/per_lake/entitables/raw_ensemble/metrics.json",
            "corpus": r4 / "taskJ_per_lake_baselines/corpora/entitables_corpus.jsonl",
            "dev_data": root
            / "work/stage1_stage2_entitables20k_v4_20260827/stage1_data/target_lists.jsonl",
            "teacher": r4 / "taskM_entitables_teacher/checkpoints/teacher_path.pt",
        }
    return {
        "selection": r6
        / "taskE_modality_balance/wdc/relation_weight_2/student_path.pt.selection.json",
        "baseline_student": r6
        / "taskE_modality_balance/wdc/relation_weight_2/final_evaluation/metrics.json",
        "baseline_raw": r4 / "taskJ_per_lake_baselines/per_lake/wdc/raw/metrics.json",
        "baseline_teacher": r4
        / "taskJ_per_lake_baselines/per_lake/wdc/raw_ensemble/metrics.json",
        "corpus": r4 / "taskJ_per_lake_baselines/corpora/wdc_corpus.jsonl",
        "dev_data": root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data/target_lists.jsonl",
        "teacher": r4
        / "taskM_entitables_teacher/per_lake/wdc/checkpoints/teacher_path.pt",
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _difference(left: Sequence[float], right: Sequence[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("Per-query metric lengths differ")
    return [float(a) - float(b) for a, b in zip(left, right)]


def summarize(
    enhanced_path: Path,
    baseline_student_path: Path,
    baseline_raw_path: Path,
    baseline_teacher_path: Path,
    output_dir: Path,
    *,
    table_tokens_per_group: int,
    evaluation_mode: str = "zero_shot",
    teacher_checkpoint: Path | None = None,
) -> dict[str, Any]:
    enhanced = _load(enhanced_path)
    baseline_student = _load(baseline_student_path)
    baseline_raw = _load(baseline_raw_path)
    baseline_teacher = _load(baseline_teacher_path)
    student = baseline_student["systems"]["student"]["metrics"]
    new_raw = enhanced["systems"]["raw"]["metrics"]
    old_raw = baseline_raw["systems"]["raw"]["metrics"]
    new_teacher = enhanced["systems"]["raw_ensemble"]["metrics"]
    old_teacher = baseline_teacher["systems"]["raw_ensemble"]["metrics"]
    new_raw_per_query = new_raw["per_query"]["direct"]["recall@10"]
    old_raw_per_query = old_raw["per_query"]["direct"]["recall@10"]
    if new_raw_per_query != old_raw_per_query:
        raise ValueError("Task F2 changed the fixed raw retrieval candidate pool")
    new_teacher_delta = _difference(
        new_teacher["per_query"]["recall@10"], new_raw_per_query
    )
    old_teacher_delta = _difference(
        old_teacher["per_query"]["recall@10"], old_raw_per_query
    )
    zero_delta = paired_bootstrap_delta(
        student["per_query"]["evidence"]["recall@10"],
        student["per_query"]["evidence"]["recall@10"],
        iterations=10_000,
        seed=13,
    )
    payload = {
        "format_version": 1,
        "configuration": {
            "max_rows": 12,
            "table_row_format": "values",
            "table_tokens_per_group": table_tokens_per_group,
            "pooling": "contiguous_mean_segments",
            "evaluation_mode": evaluation_mode,
            "teacher_checkpoint": (
                str(teacher_checkpoint.resolve())
                if teacher_checkpoint is not None
                else None
            ),
        },
        "student_evidence_recall@10": {
            "baseline": float(student["evidence"]["recall@10"]),
            "task_f": float(student["evidence"]["recall@10"]),
            "delta": zero_delta,
            "unchanged_by_construction": True,
        },
        "teacher_rerank_delta@10": {
            "baseline": sum(old_teacher_delta) / len(old_teacher_delta),
            "task_f": sum(new_teacher_delta) / len(new_teacher_delta),
            "difference_in_differences": paired_bootstrap_delta(
                new_teacher_delta,
                old_teacher_delta,
                iterations=10_000,
                seed=13,
            ),
        },
        "metrics": str(enhanced_path.resolve()),
        "baseline_student_metrics": str(baseline_student_path.resolve()),
        "baseline_raw_metrics": str(baseline_raw_path.resolve()),
        "baseline_teacher_metrics": str(baseline_teacher_path.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    evidence = payload["student_evidence_recall@10"]
    teacher = payload["teacher_rerank_delta@10"]
    teacher_delta = teacher["difference_in_differences"]
    (output_dir / "RESULTS.md").write_text(
        "\n".join(
            [
                f"# Stage-1 r6 Task F2: table token budget ({evaluation_mode})",
                "",
                "| Metric | Baseline | F2 | Delta / 95% CI |",
                "| --- | ---: | ---: | --- |",
                (
                    f"| Student evidence R@10 | {evidence['baseline']:.2%} | "
                    f"{evidence['task_f']:.2%} | unchanged by construction |"
                ),
                (
                    f"| Teacher rerank delta@10 | {teacher['baseline']:.2%} | "
                    f"{teacher['task_f']:.2%} | {teacher_delta['mean']:+.2%} "
                    f"[{teacher_delta['ci_low']:+.2%}, "
                    f"{teacher_delta['ci_high']:+.2%}] |"
                ),
                "",
                (
                    "F2 retains the original 12-row value serialization and keeps "
                    f"up to {table_tokens_per_group} contiguous pooled tokens per "
                    "schema/row group."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    inputs = _lake_inputs(root, args.lake)
    base_output_dir = (
        args.output_root
        / "taskF_table_representation"
        / f"tokens_per_group_{args.table_tokens_per_group}"
    )
    output_dir = (
        base_output_dir / "retrained" / args.lake
        if args.evaluation_mode == "retrained"
        else base_output_dir / args.lake
    )
    teacher_checkpoint = args.teacher_checkpoint or inputs["teacher"]
    evaluation = output_dir / "evaluation"
    metrics = evaluation / "metrics.json"
    if not metrics.is_file():
        run_logged_subprocess(
            [
                sys.executable,
                str(root / "src/evaluate_stage1_r3_baselines.py"),
                "--selection",
                str(inputs["selection"]),
                "--features",
                str(args.features),
                "--dev-data",
                str(inputs["dev_data"]),
                "--corpus",
                str(inputs["corpus"]),
                "--objects",
                str(args.objects),
                "--teacher-checkpoint",
                str(teacher_checkpoint),
                "--output-dir",
                str(evaluation),
                "--title",
                f"Stage-1 r6 Task F2 ({args.evaluation_mode}): {args.lake}",
                "--systems",
                "raw,raw_ensemble",
                "--recall-ks",
                "10",
                "--gamma",
                "10",
                "--gamma-evidence",
                "2",
                "--teacher-alpha",
                "0.7",
                "--teacher-batch-size",
                str(args.teacher_batch_size),
                "--teacher-table-tokens-per-group",
                str(args.table_tokens_per_group),
                "--feature-cache-size",
                "4000",
                "--bootstrap-iterations",
                "10000",
                "--bootstrap-seed",
                "13",
                "--device",
                args.device,
            ],
            output_dir / "evaluation.log",
            root=root,
        )
    payload = summarize(
        metrics,
        inputs["baseline_student"],
        inputs["baseline_raw"],
        inputs["baseline_teacher"],
        output_dir,
        table_tokens_per_group=args.table_tokens_per_group,
        evaluation_mode=args.evaluation_mode,
        teacher_checkpoint=teacher_checkpoint,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", choices=("entitables", "wdc"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--table-tokens-per-group", type=int, default=4)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--evaluation-mode",
        choices=("zero_shot", "retrained"),
        default="zero_shot",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r6_20260830"),
    )
    values = parser.parse_args()
    if values.table_tokens_per_group <= 1:
        parser.error("--table-tokens-per-group must be greater than one")
    if values.teacher_batch_size <= 0:
        parser.error("--teacher-batch-size must be positive")
    if values.evaluation_mode == "retrained" and values.teacher_checkpoint is None:
        parser.error("--teacher-checkpoint is required for retrained evaluation")
    return values


if __name__ == "__main__":
    run(parse_args())
