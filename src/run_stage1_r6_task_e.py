#!/usr/bin/env python
"""Train and evaluate one Stage-1 r6 image-relation loss-weight variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mmdd_stage1.experiment_process import run_logged_subprocess


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    r4 = root / "work/stage1_optimization_r4_20260829"
    task_c = args.output_root / "taskC_adaptive_tau" / f"{args.lake}.json"
    adaptive = json.loads(task_c.read_text(encoding="utf-8"))
    tau = float(adaptive["selected_tau"])
    if args.lake == "entitables":
        source_data = root / "work/stage1_stage2_entitables20k_v4_20260827/stage1_data"
        corpus = r4 / "taskJ_per_lake_baselines/corpora/entitables_corpus.jsonl"
        raw_index = r4 / "taskJ_per_lake_baselines/epoch0/entitables/raw_index"
        pca = r4 / "taskJ_per_lake_baselines/pca/entitables_pca_1024.pt"
        teacher = r4 / "taskM_entitables_teacher/checkpoints/teacher_path.pt"
        dataset = "entitables20k_v4"
    else:
        source_data = (
            root
            / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data"
        )
        corpus = r4 / "taskJ_per_lake_baselines/corpora/wdc_corpus.jsonl"
        raw_index = r4 / "taskJ_per_lake_baselines/epoch0/wdc/raw_index"
        pca = r4 / "taskJ_per_lake_baselines/pca/wdc_pca_1024.pt"
        teacher = (
            r4
            / "taskM_entitables_teacher/per_lake/wdc/checkpoints/teacher_path.pt"
        )
        dataset = "wdc2k_v2"

    features = (
        root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    )
    target_data = source_data / "target_lists.jsonl"
    weight_label = f"{args.relation_weight:g}"
    run_dir = (
        args.output_root
        / "taskE_modality_balance"
        / args.lake
        / f"relation_weight_{weight_label}"
    )
    checkpoint = run_dir / "student_path.pt"
    selection = checkpoint.with_suffix(checkpoint.suffix + ".selection.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not selection.is_file():
        command = [
            sys.executable,
            str(root / "src/train_stage1.py"),
            "student-path",
            "--features",
            str(features),
            "--base-data",
            str(target_data),
            "--dev-data",
            str(target_data),
            "--student-initialization",
            "pca",
            "--student-pca-basis",
            str(pca),
            "--student-dim",
            "1024",
            "--freeze-projection",
            "--corpus",
            str(corpus),
            "--raw-index-root",
            str(raw_index),
            "--teacher-checkpoint",
            str(teacher),
            "--teacher-logit-cache",
            str(run_dir / "teacher_logits"),
            "--teacher-logit-batch-size",
            "32",
            "--teacher-amp",
            "off",
            "--kd-target-teacher-alpha",
            f"{tau:g}",
            "--distillation-weight",
            "0.3",
            "--in-batch-negatives",
            "--in-batch-max-negatives",
            "256",
            "--anchor-weight",
            "0.1",
            "--anchor-weight-evidence",
            "0.1",
            "--learning-rate",
            "1e-4",
            "--relation-learning-rate",
            "1e-5",
            "--relation-loss-weight",
            f"table_to_image={args.relation_weight:g}",
            f"image_to_table={args.relation_weight:g}",
            "--dataset-sampling-alpha",
            "0",
            "--epochs",
            "12",
            "--batch-size",
            "64",
            "--patience",
            "4",
            "--primary-metric",
            "recall@10",
            "--per-dataset-gate",
            f"{dataset}:recall@10>=0",
            "--gate-tolerance",
            "0.02",
            "--bootstrap-iterations",
            "10000",
            "--bootstrap-seed",
            "13",
            "--recall-ks",
            "10,20,30,40,50",
            "--train-eval-ks",
            "10,50",
            "--gamma",
            "10",
            "--gamma-evidence",
            "2",
            "--evidence-aggregation",
            "logsumexp",
            "--evidence-top-k",
            "4",
            "--evidence-types",
            "text",
            "image",
            "--fusion-mode",
            "weighted_rrf",
            "--direct-weight",
            "1",
            "--evidence-weight",
            "0.05",
            "--min-dev-evidence-path-queries",
            "1",
            "--min-dev-evidence-path-coverage",
            "0.07",
            "--min-dev-evidence-path-coverage-by-dataset",
            "0.01",
            "--device",
            args.device,
            "--output",
            str(checkpoint),
        ]
        run_logged_subprocess(command, run_dir / "train.log", root=root)

    evaluation = run_dir / "final_evaluation"
    if not (evaluation / "metrics.json").is_file():
        command = [
            sys.executable,
            str(root / "src/evaluate_stage1_r3_baselines.py"),
            "--selection",
            str(selection),
            "--features",
            str(features),
            "--dev-data",
            str(target_data),
            "--corpus",
            str(corpus),
            "--teacher-checkpoint",
            str(teacher),
            "--output-dir",
            str(evaluation),
            "--title",
            f"Stage-1 r6 Task E: {args.lake} relation weight {weight_label}",
            "--student-label",
            f"{args.lake} image relation weight {weight_label}",
            "--systems",
            "student",
            "--recall-ks",
            "10,20,30,40,50",
            "--gamma",
            "10",
            "--gamma-evidence",
            "2",
            "--evidence-types",
            "text",
            "image",
            "--feature-cache-size",
            "60000",
            "--bootstrap-iterations",
            "10000",
            "--bootstrap-seed",
            "13",
            "--device",
            args.device,
        ]
        run_logged_subprocess(
            command, run_dir / "final_evaluation.log", root=root
        )

    _write_manifest = {
        "format_version": 1,
        "lake": args.lake,
        "dataset": dataset,
        "relation_loss_weights": {
            "table_to_image": args.relation_weight,
            "image_to_table": args.relation_weight,
        },
        "evidence_types": ["text", "image"],
        "adaptive_tau": tau,
        "adaptive_tau_source": str(task_c.resolve()),
        "teacher_checkpoint": str(teacher.resolve()),
        "student_selection": str(selection.resolve()),
        "evaluation": str((evaluation / "metrics.json").resolve()),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(_write_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", required=True, choices=["entitables", "wdc"])
    parser.add_argument("--device", required=True)
    parser.add_argument("--relation-weight", type=float, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r6_20260830"),
    )
    values = parser.parse_args()
    if values.relation_weight <= 0:
        parser.error("--relation-weight must be positive")
    return values


if __name__ == "__main__":
    run(parse_args())
