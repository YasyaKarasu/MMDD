#!/usr/bin/env python
"""Evaluate Task F1 with more serialized rows and named cells."""

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
            "selection": r6 / "taskE_modality_balance/entitables/relation_weight_2/student_path.pt.selection.json",
            "baseline_student": r6 / "taskE_modality_balance/entitables/relation_weight_2/final_evaluation/metrics.json",
            "baseline_raw": r4 / "taskJ_per_lake_baselines/per_lake/entitables/raw/metrics.json",
            "baseline_teacher": r4 / "taskJ_per_lake_baselines/per_lake/entitables/raw_ensemble/metrics.json",
            "corpus": r4 / "taskJ_per_lake_baselines/corpora/entitables_corpus.jsonl",
            "dev_data": root / "work/stage1_stage2_entitables20k_v4_20260827/stage1_data/target_lists.jsonl",
            "teacher": r4 / "taskM_entitables_teacher/checkpoints/teacher_path.pt",
        }
    return {
        "selection": r6 / "taskE_modality_balance/wdc/relation_weight_2/student_path.pt.selection.json",
        "baseline_student": r6 / "taskE_modality_balance/wdc/relation_weight_2/final_evaluation/metrics.json",
        "baseline_raw": r4 / "taskJ_per_lake_baselines/per_lake/wdc/raw/metrics.json",
        "baseline_teacher": r4 / "taskJ_per_lake_baselines/per_lake/wdc/raw_ensemble/metrics.json",
        "corpus": r4 / "taskJ_per_lake_baselines/corpora/wdc_corpus.jsonl",
        "dev_data": root / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data/target_lists.jsonl",
        "teacher": r4 / "taskM_entitables_teacher/per_lake/wdc/checkpoints/teacher_path.pt",
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
) -> dict[str, Any]:
    enhanced = _load(enhanced_path)
    baseline_student = _load(baseline_student_path)
    baseline_raw = _load(baseline_raw_path)
    baseline_teacher = _load(baseline_teacher_path)

    new_student = enhanced["systems"]["student"]["metrics"]
    old_student = baseline_student["systems"]["student"]["metrics"]
    new_raw = enhanced["systems"]["raw"]["metrics"]
    old_raw = baseline_raw["systems"]["raw"]["metrics"]
    new_teacher = enhanced["systems"]["raw_ensemble"]["metrics"]
    old_teacher = baseline_teacher["systems"]["raw_ensemble"]["metrics"]

    new_evidence_per_query = new_student["per_query"]["evidence"]["recall@10"]
    old_evidence_per_query = old_student["per_query"]["evidence"]["recall@10"]
    new_teacher_delta = _difference(
        new_teacher["per_query"]["recall@10"],
        new_raw["per_query"]["direct"]["recall@10"],
    )
    old_teacher_delta = _difference(
        old_teacher["per_query"]["recall@10"],
        old_raw["per_query"]["direct"]["recall@10"],
    )
    payload = {
        "format_version": 1,
        "configuration": {
            "max_rows": 20,
            "table_row_format": "named_cells",
            "table_tokens_per_group": 1,
        },
        "student_evidence_recall@10": {
            "baseline": float(old_student["evidence"]["recall@10"]),
            "task_f": float(new_student["evidence"]["recall@10"]),
            "delta": paired_bootstrap_delta(
                new_evidence_per_query,
                old_evidence_per_query,
                iterations=10_000,
                seed=13,
            ),
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
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    evidence = payload["student_evidence_recall@10"]
    teacher = payload["teacher_rerank_delta@10"]
    evidence_delta = evidence["delta"]
    teacher_delta = teacher["difference_in_differences"]
    (output_dir / "RESULTS.md").write_text(
        "\n".join(
            [
                "# Stage-1 r6 Task F1: row count and serialization",
                "",
                "| Metric | Baseline | Task F | Delta / 95% CI |",
                "| --- | ---: | ---: | --- |",
                (
                    f"| Student evidence R@10 | {evidence['baseline']:.2%} | "
                    f"{evidence['task_f']:.2%} | {evidence_delta['mean']:+.2%} "
                    f"[{evidence_delta['ci_low']:+.2%}, "
                    f"{evidence_delta['ci_high']:+.2%}] |"
                ),
                (
                    f"| Teacher rerank delta@10 | {teacher['baseline']:.2%} | "
                    f"{teacher['task_f']:.2%} | {teacher_delta['mean']:+.2%} "
                    f"[{teacher_delta['ci_low']:+.2%}, "
                    f"{teacher_delta['ci_high']:+.2%}] |"
                ),
                "",
                "F1 uses max_rows=20 and named-cell row serialization while retaining one token per schema/row group and the original dev lists and checkpoints.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    inputs = _lake_inputs(root, args.lake)
    output_dir = args.output_root / "taskF_table_representation" / args.lake
    output_dir.mkdir(parents=True, exist_ok=True)
    original_selection = _load(inputs["selection"])
    student_index = output_dir / "student_index"
    if not (student_index / "manifest.json").is_file():
        run_logged_subprocess(
            [
                sys.executable,
                str(root / "src/build_stage1_index.py"),
                "--features",
                str(args.features),
                "--student-checkpoint",
                str(original_selection["best_checkpoint"]),
                "--corpus",
                str(inputs["corpus"]),
                "--output-dir",
                str(student_index),
                "--device",
                args.device,
                "--feature-cache-size",
                "16000",
            ],
            output_dir / "build_student_index.log",
            root=root,
        )

    selection = dict(original_selection)
    selection["best_index"] = str(student_index.resolve())
    selection["latest_index"] = str(student_index.resolve())
    selection["raw_embedding_index"] = str((output_dir / "raw_index").resolve())
    selection_path = output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    evaluation = output_dir / "evaluation"
    metrics = evaluation / "metrics.json"
    if not metrics.is_file():
        run_logged_subprocess(
            [
                sys.executable,
                str(root / "src/evaluate_stage1_r3_baselines.py"),
                "--selection",
                str(selection_path),
                "--features",
                str(args.features),
                "--dev-data",
                str(inputs["dev_data"]),
                "--corpus",
                str(inputs["corpus"]),
                "--objects",
                str(args.objects),
                "--teacher-checkpoint",
                str(inputs["teacher"]),
                "--output-dir",
                str(evaluation),
                "--title",
                f"Stage-1 r6 Task F: {args.lake}",
                "--student-label",
                f"{args.lake} Task-F Student",
                "--systems",
                "raw,student,raw_ensemble",
                "--recall-ks",
                "10",
                "--gamma",
                "10",
                "--gamma-evidence",
                "2",
                "--teacher-alpha",
                "0.7",
                "--teacher-batch-size",
                "64",
                "--feature-cache-size",
                "16000",
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
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", choices=("entitables", "wdc"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r6_20260830"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
