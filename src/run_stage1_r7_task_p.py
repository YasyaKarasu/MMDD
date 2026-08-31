#!/usr/bin/env python
"""Train and evaluate one Stage-1 r7 low-rank residual Student variant."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _run(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            list(command),
            cwd=root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _lake_inputs(root: Path, lake: str) -> dict[str, Any]:
    r4 = root / "work/stage1_optimization_r4_20260829"
    r6 = root / "work/stage1_optimization_r6_20260830"
    if lake == "entitables":
        data = root / "work/stage1_stage2_entitables20k_v4_20260827/stage1_data"
        teacher = r4 / "taskM_entitables_teacher/checkpoints/teacher_path.pt"
        teacher_logits = (
            r6
            / "taskE_modality_balance/entitables/relation_weight_2/teacher_logits"
        )
        dataset = "entitables20k_v4"
        tau = 0.7
    else:
        data = (
            root
            / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data"
        )
        teacher = (
            r4
            / "taskM_entitables_teacher/per_lake/wdc/checkpoints/teacher_path.pt"
        )
        teacher_logits = (
            r6 / "taskE_modality_balance/wdc/relation_weight_1/teacher_logits"
        )
        dataset = "wdc2k_v2"
        tau = 0.5
    return {
        "data": data,
        "target_data": data / "target_lists.jsonl",
        "corpus": r4 / f"taskJ_per_lake_baselines/corpora/{lake}_corpus.jsonl",
        "raw_index": r4 / f"taskJ_per_lake_baselines/epoch0/{lake}/raw_index",
        "pca": r4 / f"taskJ_per_lake_baselines/pca/{lake}_pca_1024.pt",
        "teacher": teacher,
        "teacher_logits": teacher_logits,
        "dataset": dataset,
        "tau": tau,
    }


def _label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    inputs: dict[str, Any],
    selection: Path,
    evaluation: Path,
) -> None:
    payload = {
        "format_version": 1,
        "task": "P",
        "lake": args.lake,
        "dataset": inputs["dataset"],
        "relation_param": "lowrank",
        "relation_rank": args.rank,
        "anchor_weight": args.anchor_weight,
        "adaptive_tau": inputs["tau"],
        "kd_temperature": args.kd_temperature,
        "distillation_weight": 0.3,
        "in_batch_max_negatives": 256,
        "teacher_checkpoint": str(inputs["teacher"].resolve()),
        "teacher_logit_cache": str(inputs["teacher_logits"].resolve()),
        "student_selection": str(selection.resolve()),
        "evaluation": str(evaluation.resolve()),
        "max_dev_queries": args.max_dev_queries,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    inputs = _lake_inputs(root, args.lake)
    features = (
        root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    )
    rank_root = args.output_root / "taskP_lowrank" / args.lake / f"k_{args.rank}"
    if args.kd_temperature == 1.0:
        run_dir = rank_root / f"mu_{_label(args.anchor_weight)}"
    else:
        run_dir = (
            rank_root
            / f"t_{_label(args.kd_temperature)}"
            / f"mu_{_label(args.anchor_weight)}"
        )
    checkpoint = run_dir / "student_path.pt"
    selection = checkpoint.with_suffix(checkpoint.suffix + ".selection.json")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep the full training set but optionally cap dev evaluation for large lakes.
    dev_data = inputs["target_data"]
    if args.max_dev_queries is not None:
        dev_data = run_dir / f"dev_subset_{args.max_dev_queries}.jsonl"
        if not dev_data.is_file():
            kept = 0
            with inputs["target_data"].open(encoding="utf-8") as source, dev_data.open(
                "w", encoding="utf-8"
            ) as target:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("split") != "dev":
                        continue
                    target.write(line)
                    kept += 1
                    if kept >= args.max_dev_queries:
                        break
            if kept < args.max_dev_queries:
                raise ValueError(
                    f"requested {args.max_dev_queries} dev queries, found {kept}"
                )

    if not selection.is_file():
        command = [
            sys.executable,
            str(root / "src/train_stage1.py"),
            "student-path",
            "--features",
            str(features),
            "--base-data",
            str(inputs["target_data"]),
            "--dev-data",
            str(dev_data),
            "--student-initialization",
            "pca",
            "--student-pca-basis",
            str(inputs["pca"]),
            "--student-dim",
            "1024",
            "--freeze-projection",
            "--relation-param",
            "lowrank",
            "--relation-rank",
            str(args.rank),
            "--corpus",
            str(inputs["corpus"]),
            "--raw-index-root",
            str(inputs["raw_index"]),
            "--teacher-checkpoint",
            str(inputs["teacher"]),
            "--teacher-logit-cache",
            str(inputs["teacher_logits"]),
            "--teacher-logit-batch-size",
            "32",
            "--teacher-amp",
            "off",
            "--kd-target-teacher-alpha",
            f"{inputs['tau']:g}",
            "--distillation-weight",
            "0.3",
            "--temperature",
            f"{args.kd_temperature:g}",
            "--in-batch-negatives",
            "--in-batch-max-negatives",
            "256",
            "--anchor-weight",
            f"{args.anchor_weight:g}",
            "--anchor-weight-evidence",
            f"{args.anchor_weight:g}",
            "--learning-rate",
            "1e-4",
            "--relation-learning-rate",
            "1e-5",
            "--dataset-sampling-alpha",
            "0",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            "64",
            "--patience",
            "4",
            "--primary-metric",
            "recall@10",
            "--per-dataset-gate",
            f"{inputs['dataset']}:recall@10>=0",
            "--gate-tolerance",
            "0.02",
            "--bootstrap-iterations",
            str(args.bootstrap_iterations),
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
        _run(command, run_dir / "train.log")

    evaluation = run_dir / "final_evaluation"
    metrics_path = evaluation / "metrics.json"
    if not metrics_path.is_file():
        command = [
            sys.executable,
            str(root / "src/evaluate_stage1_r3_baselines.py"),
            "--selection",
            str(selection),
            "--features",
            str(features),
            "--dev-data",
            str(dev_data),
            "--corpus",
            str(inputs["corpus"]),
            "--teacher-checkpoint",
            str(inputs["teacher"]),
            "--output-dir",
            str(evaluation),
            "--title",
            f"Stage-1 r7 Task P: {args.lake} lowrank k={args.rank}",
            "--student-label",
            f"{args.lake} lowrank k={args.rank} mu={args.anchor_weight:g}",
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
            str(args.bootstrap_iterations),
            "--bootstrap-seed",
            "13",
            "--device",
            args.device,
        ]
        _run(command, run_dir / "evaluation.log")

    _write_manifest(
        run_dir / "run_manifest.json",
        args=args,
        inputs=inputs,
        selection=selection,
        evaluation=metrics_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", required=True, choices=["entitables", "wdc"])
    parser.add_argument("--rank", required=True, type=int, choices=[16, 64, 256])
    parser.add_argument("--anchor-weight", required=True, type=float, choices=[0.0, 0.1])
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--kd-temperature",
        type=float,
        default=1.0,
        help="Temperature used by the listwise KD loss (Task R sweep).",
    )
    parser.add_argument(
        "--max-dev-queries",
        type=int,
        default=None,
        help="Cap dev queries for exploratory runs; training data remains full.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r7_20260831"),
    )
    values = parser.parse_args()
    if values.epochs <= 0 or values.bootstrap_iterations <= 0:
        parser.error("--epochs and --bootstrap-iterations must be positive")
    if values.max_dev_queries is not None and values.max_dev_queries <= 0:
        parser.error("--max-dev-queries must be positive")
    if values.kd_temperature <= 0:
        parser.error("--kd-temperature must be positive")
    return values


if __name__ == "__main__":
    run(parse_args())
