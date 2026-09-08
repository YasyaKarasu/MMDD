#!/usr/bin/env python
"""Run the R10 Task-E Student-only controlled comparisons under E01."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from mmdd_stage1.experiment_process import run_logged_subprocess
from mmdd_stage1.retrieval import checkpoint_fingerprint


ARMS: dict[str, dict[str, str | float | bool]] = {
    "e01": {
        "edge_bce_weight": 0.0,
        "distillation_weight": 0.3,
        "freeze_projection": False,
        "positive_loss_mode": "sum_probability",
    },
    "student_bce_on": {
        "edge_bce_weight": 1.0,
        "distillation_weight": 0.3,
        "freeze_projection": False,
        "positive_loss_mode": "sum_probability",
    },
    "kd_off": {
        "edge_bce_weight": 0.0,
        "distillation_weight": 0.0,
        "freeze_projection": False,
        "positive_loss_mode": "sum_probability",
    },
    "p_frozen": {
        "edge_bce_weight": 0.0,
        "distillation_weight": 0.3,
        "freeze_projection": True,
        "positive_loss_mode": "sum_probability",
    },
    "mean_positive": {
        "edge_bce_weight": 0.0,
        "distillation_weight": 0.3,
        "freeze_projection": False,
        "positive_loss_mode": "mean_log_probability",
    },
}

AGGREGATION_FLAGS = [
    "--evidence-aggregation",
    "logmeanexp",
    "--evidence-top-k",
    "4",
    "--evidence-temperature",
    "0.1",
    "--evidence-power",
    "2",
    "--path-combination",
    "min",
    "--evidence-threshold",
    "0",
    "--evidence-target-temperature",
    "0.1",
]


def _selection_path(output: Path) -> Path:
    return Path(str(output) + ".selection.json")


def _history_path(output: Path) -> Path:
    return Path(str(output) + ".history.json")


def _last_checkpoint(output: Path) -> Path:
    return output.with_name(output.stem + ".last.pt")


def _run_stage(command: list[str], output: Path, log_path: Path, root: Path) -> None:
    if _selection_path(output).is_file() and _last_checkpoint(output).is_file():
        return
    run_logged_subprocess(command, log_path, root=root)
    if not _selection_path(output).is_file() or not _last_checkpoint(output).is_file():
        raise RuntimeError(f"Training stage did not complete: {output}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage_record(output: Path) -> dict[str, Any]:
    selection = _load_json(_selection_path(output))
    history = _load_json(_history_path(output))
    epochs = history.get("epochs", [])
    endpoint = epochs[-1] if epochs else None
    return {
        "selected_checkpoint": str(output),
        "selected_checkpoint_sha256": checkpoint_fingerprint(output),
        "endpoint_checkpoint": str(_last_checkpoint(output)),
        "endpoint_checkpoint_sha256": checkpoint_fingerprint(
            _last_checkpoint(output)
        ),
        "best_epoch": selection.get("best_epoch"),
        "best_metrics": selection.get("best_metrics"),
        "endpoint_record": endpoint,
        "positive_loss_mode": history.get("positive_loss_mode"),
    }


def _projection_flag(frozen: bool) -> str:
    return "--freeze-projection" if frozen else "--no-freeze-projection"


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    r10 = Path(args.r10_root).resolve()
    output_root = Path(args.output_root).resolve()
    e01_root = r10 / "taskE_matched" / "t0" / "e01"
    t0_root = r10 / "taskE_matched" / "t0"
    features = r10 / "features_qwen3_vl_embedding_8b"
    corpus = r10 / "stage1_data" / "stage1_corpus.jsonl"
    raw_index = r10 / "taskA_protocol" / "baselines" / "raw_index"
    edge_train = r10 / "taskD_edge_labels" / "lists" / "edge_lists.train_fit.jsonl"
    edge_dev = r10 / "taskD_edge_labels" / "lists" / "edge_lists.dev.jsonl"
    path_train = r10 / "taskD_edge_labels" / "lists" / "target_lists.train_fit.jsonl"
    path_dev = r10 / "taskD_edge_labels" / "lists" / "target_lists.dev.jsonl"
    c4_student = (
        r10
        / "taskC_pr_ablation"
        / "c4_r_warmup_then_p1e-6_r1e-5"
        / "phase2_unfrozen"
        / "student_path.pt"
    )
    edge_teacher = t0_root / "teacher_edge.last.pt"
    path_teacher = e01_root / "teacher_path.last.pt"
    hard_targets = e01_root / "hard_targets.jsonl"
    edge_teacher_cache = t0_root / "teacher_logits" / "edge"
    path_teacher_cache = e01_root / "teacher_logits" / "path"
    baseline = e01_root / "student_path.pt"
    required = [
        features / "merge_summary.json",
        corpus,
        raw_index / "manifest.json",
        edge_train,
        edge_dev,
        path_train,
        path_dev,
        c4_student,
        edge_teacher,
        path_teacher,
        hard_targets,
        baseline,
        _last_checkpoint(baseline),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Task-E Student-control prerequisites:\n" + "\n".join(missing)
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    python = sys.executable
    train = str(root / "src" / "train_stage1.py")
    output_root.mkdir(parents=True, exist_ok=True)

    for arm_name in args.arms:
        config = ARMS[arm_name]
        arm_root = output_root / arm_name
        arm_root.mkdir(parents=True, exist_ok=True)
        edge_output = arm_root / "student_edge.pt"
        common_student_flags = [
            "--seed",
            str(args.seed),
            "--student-confidence-transform",
            "--student-score-space",
            "confidence",
            "--positive-loss-mode",
            str(config["positive_loss_mode"]),
            _projection_flag(bool(config["freeze_projection"])),
            "--distillation-weight",
            str(config["distillation_weight"]),
            "--edge-bce-weight",
            str(config["edge_bce_weight"]),
            "--in-batch-negatives",
            "--in-batch-max-negatives",
            "256",
            "--anchor-weight",
            "0.1",
            "--anchor-weight-evidence",
            "0.1",
            "--learning-rate",
            "1e-6",
            "--relation-learning-rate",
            "1e-5",
            "--dataset-sampling-alpha",
            "0",
            "--batch-size",
            "64",
            "--patience",
            "0",
            "--device",
            "cuda:0",
        ]
        edge_command = [
            python,
            train,
            "student-edge",
            "--features",
            str(features),
            "--base-data",
            str(edge_train),
            "--dev-data",
            str(edge_dev),
            "--student-checkpoint",
            str(c4_student),
            "--teacher-checkpoint",
            str(edge_teacher),
            "--teacher-score-space",
            "confidence",
            "--teacher-logit-cache",
            str(edge_teacher_cache),
            "--teacher-logit-batch-size",
            "32",
            "--teacher-amp",
            "off",
            "--edge-type-oversample",
            "--epochs",
            "2",
            "--output",
            str(edge_output),
            *common_student_flags,
        ]
        _run_stage(edge_command, edge_output, arm_root / "student_edge.log", root)

        path_output = arm_root / "student_path.pt"
        path_command = [
            python,
            train,
            "student-path",
            "--features",
            str(features),
            "--base-data",
            str(path_train),
            "--hard-data",
            str(hard_targets),
            "--hard-source-checkpoint",
            str(c4_student),
            "--hard-fraction",
            "0.25",
            "--hard-learning-rate",
            "1e-6",
            "--dev-data",
            str(path_dev),
            "--student-checkpoint",
            str(_last_checkpoint(edge_output)),
            "--corpus",
            str(corpus),
            "--raw-index-root",
            str(raw_index),
            "--teacher-checkpoint",
            str(path_teacher),
            "--teacher-score-space",
            "confidence",
            "--teacher-logit-cache",
            str(path_teacher_cache),
            "--teacher-logit-batch-size",
            "32",
            "--teacher-amp",
            "off",
            "--continuous-edge-data",
            str(edge_train),
            "--continuous-edge-dev-data",
            str(edge_dev),
            "--continuous-edge-teacher-checkpoint",
            str(edge_teacher),
            "--continuous-edge-teacher-logit-cache",
            str(edge_teacher_cache),
            "--continuous-edge-weight",
            "1",
            "--continuous-edge-batch-size",
            "64",
            "--epochs",
            "10",
            "--max-optimizer-updates",
            "356",
            "--primary-metric",
            "valid_path_recall@10,4",
            "--per-dataset-gate",
            "entitables-final-layout-gaussian-v9:recall@10>=0",
            "--gate-tolerance",
            "0.02",
            "--bootstrap-iterations",
            "10000",
            "--bootstrap-seed",
            "13",
            "--recall-ks",
            "10,20,50",
            "--train-eval-ks",
            "10,20,50",
            "--direct-k",
            "100",
            "--evidence-k",
            "20",
            "--targets-per-evidence",
            "20",
            "--gamma",
            "10",
            "--gamma-evidence",
            "2",
            "--fusion-mode",
            "weighted_rrf",
            "--direct-weight",
            "1",
            "--evidence-weight",
            "0.05",
            "--min-dev-evidence-path-queries",
            "1",
            "--min-dev-evidence-path-coverage",
            "0",
            "--min-dev-evidence-path-coverage-by-dataset",
            "0",
            "--output",
            str(path_output),
            *common_student_flags,
            *AGGREGATION_FLAGS,
        ]
        _run_stage(path_command, path_output, arm_root / "student_path.log", root)

        _write_json(
            arm_root / "manifest.json",
            {
                "format_version": 1,
                "experiment": "R10 Task E Student-only controls under E01",
                "arm": arm_name,
                "seed": args.seed,
                "seed_scope": "Student edge/path randomness conditional on fixed Teacher, C4 initialization and mining pool",
                "factors": config,
                "fixed_budget": {
                    "student_edge_epochs": 2,
                    "student_path_optimizer_updates": 356,
                    "hard_fraction": 0.25,
                },
                "fixed_inputs": {
                    "student_initialization": str(c4_student),
                    "student_initialization_sha256": checkpoint_fingerprint(
                        c4_student
                    ),
                    "teacher_edge": str(edge_teacher),
                    "teacher_edge_sha256": checkpoint_fingerprint(edge_teacher),
                    "teacher_path": str(path_teacher),
                    "teacher_path_sha256": checkpoint_fingerprint(path_teacher),
                    "hard_targets": str(hard_targets),
                    "hard_targets_sha256": checkpoint_fingerprint(hard_targets),
                    "aggregation": "g2b_logmeanexp_min",
                    "baseline": str(baseline),
                    "baseline_sha256": checkpoint_fingerprint(baseline),
                    "baseline_endpoint": str(_last_checkpoint(baseline)),
                    "baseline_endpoint_sha256": checkpoint_fingerprint(
                        _last_checkpoint(baseline)
                    ),
                },
                "student_edge": _stage_record(edge_output),
                "student_path": _stage_record(path_output),
            },
        )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_r10 = root / "work" / "stage1_optimization_r10_20260907"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=sorted(ARMS),
        default=sorted(name for name in ARMS if name != "e01"),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--r10-root", default=str(default_r10))
    parser.add_argument(
        "--output-root",
        default=str(default_r10 / "taskE_student_controls"),
    )
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
