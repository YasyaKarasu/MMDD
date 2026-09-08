#!/usr/bin/env python
"""Run one fixed-step row of the R10 Task-E matched training table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from mmdd_stage1.experiment_process import run_logged_subprocess
from mmdd_stage1.retrieval import checkpoint_fingerprint


AGGREGATIONS: dict[str, dict[str, str | int | float]] = {
    "g0": {
        "evidence_aggregation": "logsumexp",
        "evidence_top_k": 4,
        "evidence_temperature": 0.1,
        "evidence_power": 2.0,
        "path_combination": "min",
        "evidence_threshold": 0.0,
        "evidence_target_temperature": 0.1,
    },
    "g2b": {
        "evidence_aggregation": "logmeanexp",
        "evidence_top_k": 4,
        "evidence_temperature": 0.1,
        "evidence_power": 2.0,
        "path_combination": "min",
        "evidence_threshold": 0.0,
        "evidence_target_temperature": 0.1,
    },
    "g3": {
        "evidence_aggregation": "fixed_power_mean",
        "evidence_top_k": 4,
        "evidence_temperature": 1.0,
        "evidence_power": 2.0,
        "path_combination": "min",
        "evidence_threshold": 0.0,
        "evidence_target_temperature": 0.1,
    },
}

CELL_IDS = {
    "t0": {"g0": "e00", "g2b": "e01", "g3": "e02"},
    "t1": {"g0": "e10", "g2b": "e11", "g3": "e12", "g5": "e13"},
}


def _selection_path(output: Path) -> Path:
    return Path(str(output) + ".selection.json")


def _last_checkpoint(output: Path) -> Path:
    return output.with_name(output.stem + ".last.pt")


def _run_stage(command: list[str], output: Path, log_path: Path, root: Path) -> None:
    if _selection_path(output).is_file() and _last_checkpoint(output).is_file():
        return
    run_logged_subprocess(command, log_path, root=root)
    if not _selection_path(output).is_file() or not _last_checkpoint(output).is_file():
        raise RuntimeError(f"Training stage did not complete: {output}")


def _aggregation_flags(config: dict[str, str | int | float]) -> list[str]:
    flags = [
        "--evidence-aggregation",
        str(config["evidence_aggregation"]),
        "--evidence-top-k",
        str(config["evidence_top_k"]),
        "--evidence-temperature",
        str(config["evidence_temperature"]),
        "--evidence-power",
        str(config["evidence_power"]),
        "--path-combination",
        str(config["path_combination"]),
        "--evidence-threshold",
        str(config["evidence_threshold"]),
        "--evidence-target-temperature",
        str(config["evidence_target_temperature"]),
    ]
    if config.get("row_support_model") is not None:
        flags.extend(
            [
                "--row-support-model",
                str(config["row_support_model"]),
                "--row-support-top-l",
                str(config["row_support_top_l"]),
                "--evidence-content-keys",
                str(config["evidence_content_keys"]),
            ]
        )
    return flags


def _validate_scored_pool(
    target_path: Path,
    edge_path: Path,
    teacher_checkpoint: Path,
    config: dict[str, str | int | float],
) -> bool:
    if not target_path.is_file() or not edge_path.is_file():
        return False
    target_metadata_path = target_path.with_suffix(target_path.suffix + ".metadata.json")
    edge_metadata_path = edge_path.with_suffix(edge_path.suffix + ".metadata.json")
    if not target_metadata_path.is_file() or not edge_metadata_path.is_file():
        return False
    target_metadata = json.loads(target_metadata_path.read_text(encoding="utf-8"))
    edge_metadata = json.loads(edge_metadata_path.read_text(encoding="utf-8"))
    expected_training = {
        **config,
        "teacher_score_space": "confidence",
        "student_score_space": "confidence",
        "teacher_ensemble_alpha": None,
    }
    return (
        target_metadata == edge_metadata
        and target_metadata.get("teacher_scoring") == "complete"
        and target_metadata.get("teacher_checkpoint_sha256")
        == checkpoint_fingerprint(teacher_checkpoint)
        and target_metadata.get("training_score_config") == expected_training
    )


def _write_manifest(
    path: Path,
    *,
    row: str,
    edge_bce_weight: float,
    edge_teacher: Path,
    edge_student: Path,
    cells: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "format_version": 1,
        "experiment": "R10 Task E matched Teacher/Student training",
        "row": row,
        "edge_bce_weight": edge_bce_weight,
        "teacher_score_space": "confidence",
        "student_score_space": "confidence",
        "path_combination": "min",
        "evidence_target_temperature": 0.1,
        "shared_edge_stages": {
            "teacher_checkpoint": str(edge_teacher),
            "teacher_checkpoint_sha256": checkpoint_fingerprint(edge_teacher),
            "student_checkpoint": str(edge_student),
            "student_checkpoint_sha256": checkpoint_fingerprint(edge_student),
            "reuse_reason": (
                "Edge ranking/KD/BCE objectives do not depend on the cross-path "
                "aggregation operator."
            ),
        },
        "fixed_step_budget": {
            "teacher_edge_epochs": 2,
            "teacher_path_epochs": 2,
            "student_edge_epochs": 2,
            "student_path_optimizer_updates": 356,
            "hard_fraction": 0.25,
        },
        "cells": cells,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    r10 = Path(args.r10_root).resolve()
    output_root = Path(args.output_root).resolve()
    row_root = output_root / args.row
    features = r10 / "features_qwen3_vl_embedding_8b"
    corpus = r10 / "stage1_data" / "stage1_corpus.jsonl"
    raw_index = r10 / "taskA_protocol" / "baselines" / "raw_index"
    edge_train = r10 / "taskD_edge_labels" / "lists" / "edge_lists.train_fit.jsonl"
    edge_dev = r10 / "taskD_edge_labels" / "lists" / "edge_lists.dev.jsonl"
    path_train = r10 / "taskD_edge_labels" / "lists" / "target_lists.train_fit.jsonl"
    path_dev = r10 / "taskD_edge_labels" / "lists" / "target_lists.dev.jsonl"
    c4_root = (
        r10
        / "taskC_pr_ablation"
        / "c4_r_warmup_then_p1e-6_r1e-5"
        / "phase2_unfrozen"
    )
    c4_student = c4_root / "student_path.pt"
    c4_index = c4_root / "student_path.dev_indices" / "epoch_000"
    pending_root = r10 / "taskD4_hard_mining"
    pending_targets = pending_root / "hard_targets_round1_4pool.filtered.pending.jsonl"
    pending_edges = pending_root / "hard_edges_round1_4pool.filtered.pending.jsonl"
    row_support_model = r10 / "taskB_g5" / "row_support_isotonic.json"
    evidence_content_keys = r10 / "taskB_g5" / "evidence_content_keys.jsonl"
    required = [
        features / "merge_summary.json",
        corpus,
        raw_index / "manifest.json",
        edge_train,
        edge_dev,
        path_train,
        path_dev,
        c4_student,
        c4_index / "manifest.json",
        pending_targets,
        pending_edges,
        row_support_model,
        evidence_content_keys,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Task-E prerequisites:\n" + "\n".join(missing))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    python = sys.executable
    train = str(root / "src" / "train_stage1.py")
    refresh = str(root / "src" / "refresh_stage1_hard_negatives.py")
    row_root.mkdir(parents=True, exist_ok=True)
    edge_bce_weight = 1.0 if args.row == "t1" else 0.0

    edge_teacher_output = row_root / "teacher_edge.pt"
    edge_teacher_command = [
        python,
        train,
        "teacher-edge",
        "--features",
        str(features),
        "--base-data",
        str(edge_train),
        "--dev-data",
        str(edge_dev),
        "--teacher-confidence-transform",
        "--teacher-score-space",
        "confidence",
        "--edge-bce-weight",
        str(edge_bce_weight),
        "--learning-rate",
        "5e-5",
        "--epochs",
        "2",
        "--batch-size",
        "8",
        "--patience",
        "0",
        "--feature-cache-size",
        "24000",
        "--teacher-amp",
        "off",
        "--seed",
        "13",
        "--device",
        "cuda:0",
        "--output",
        str(edge_teacher_output),
    ]
    _run_stage(
        edge_teacher_command,
        edge_teacher_output,
        row_root / "teacher_edge.log",
        root,
    )
    edge_teacher = _last_checkpoint(edge_teacher_output)

    edge_student_output = row_root / "student_edge.pt"
    edge_student_cache = row_root / "teacher_logits" / "edge"
    edge_student_command = [
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
        "--student-confidence-transform",
        "--student-score-space",
        "confidence",
        "--teacher-checkpoint",
        str(edge_teacher),
        "--teacher-score-space",
        "confidence",
        "--teacher-logit-cache",
        str(edge_student_cache),
        "--teacher-logit-batch-size",
        "32",
        "--teacher-amp",
        "off",
        "--distillation-weight",
        "0.3",
        "--edge-bce-weight",
        str(edge_bce_weight),
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
        "--epochs",
        "2",
        "--batch-size",
        "64",
        "--patience",
        "0",
        "--device",
        "cuda:0",
        "--output",
        str(edge_student_output),
    ]
    _run_stage(
        edge_student_command,
        edge_student_output,
        row_root / "student_edge.log",
        root,
    )
    edge_student = _last_checkpoint(edge_student_output)

    aggregations = dict(AGGREGATIONS)
    if args.row == "t1":
        aggregations["g5"] = {
            "evidence_aggregation": "greedy_row_support",
            "evidence_top_k": 4,
            "evidence_temperature": 1.0,
            "evidence_power": 2.0,
            "path_combination": "min",
            "evidence_threshold": 0.5,
            "evidence_target_temperature": 0.1,
            "row_support_model": str(row_support_model.resolve()),
            "row_support_model_sha256": checkpoint_fingerprint(
                row_support_model
            ),
            "row_support_top_l": 20,
            "evidence_content_keys": str(evidence_content_keys.resolve()),
            "evidence_content_keys_sha256": checkpoint_fingerprint(
                evidence_content_keys
            ),
        }

    cells: dict[str, dict[str, Any]] = {}
    for aggregation_name, config in aggregations.items():
        cell_id = CELL_IDS[args.row][aggregation_name]
        cell_root = row_root / cell_id
        cell_root.mkdir(parents=True, exist_ok=True)
        aggregation_flags = _aggregation_flags(config)

        teacher_path_output = cell_root / "teacher_path.pt"
        teacher_path_command = [
            python,
            train,
            "teacher-path",
            "--features",
            str(features),
            "--base-data",
            str(path_train),
            "--hard-data",
            str(pending_targets),
            "--hard-fraction",
            "0.25",
            "--hard-learning-rate",
            "5e-5",
            "--dev-data",
            str(path_dev),
            "--teacher-checkpoint",
            str(edge_teacher),
            "--teacher-confidence-transform",
            "--teacher-score-space",
            "confidence",
            "--learning-rate",
            "5e-5",
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--patience",
            "0",
            "--feature-cache-size",
            "32000",
            "--teacher-amp",
            "off",
            "--seed",
            "13",
            "--device",
            "cuda:0",
            "--output",
            str(teacher_path_output),
            *aggregation_flags,
        ]
        _run_stage(
            teacher_path_command,
            teacher_path_output,
            cell_root / "teacher_path.log",
            root,
        )
        teacher_path = _last_checkpoint(teacher_path_output)

        scored_targets = cell_root / "hard_targets.jsonl"
        scored_edges = cell_root / "hard_edges.jsonl"
        if not _validate_scored_pool(
            scored_targets, scored_edges, teacher_path, config
        ):
            refresh_command = [
                python,
                refresh,
                "--features",
                str(features),
                "--teacher-checkpoint",
                str(teacher_path),
                "--teacher-score-space",
                "confidence",
                "--student-checkpoint",
                str(c4_student),
                "--student-score-space",
                "raw_logit",
                "--training-student-score-space",
                "confidence",
                "--index-dir",
                str(c4_index),
                "--corpus",
                str(corpus),
                "--target-lists",
                str(path_train),
                "--output-target-lists",
                str(scored_targets),
                "--output-edge-lists",
                str(scored_edges),
                "--pending-target-lists",
                str(pending_targets),
                "--pending-edge-lists",
                str(pending_edges),
                "--teacher-batch-size",
                "16",
                "--feature-cache-size",
                "8000",
                "--device",
                "cuda:0",
                "--mining-round",
                "1",
                "--hard-targets-per-query",
                "8",
                "--hard-evidence-per-type",
                "8",
                "--hard-targets-per-positive-evidence",
                "8",
                "--hard-paths-per-query",
                "8",
                "--direct-k",
                "100",
                "--evidence-k",
                "20",
                "--targets-per-evidence",
                "20",
                "--evidence-types",
                "text",
                "image",
                *aggregation_flags,
            ]
            run_logged_subprocess(
                refresh_command,
                cell_root / "hard_rescore.log",
                root=root,
            )
            if not _validate_scored_pool(
                scored_targets, scored_edges, teacher_path, config
            ):
                raise RuntimeError(f"Hard-pool scoring metadata mismatch: {cell_id}")

        student_path_output = cell_root / "student_path.pt"
        student_path_cache = cell_root / "teacher_logits" / "path"
        student_path_command = [
            python,
            train,
            "student-path",
            "--features",
            str(features),
            "--base-data",
            str(path_train),
            "--hard-data",
            str(scored_targets),
            "--hard-source-checkpoint",
            str(c4_student),
            "--hard-fraction",
            "0.25",
            "--hard-learning-rate",
            "1e-6",
            "--dev-data",
            str(path_dev),
            "--student-checkpoint",
            str(edge_student),
            "--student-confidence-transform",
            "--student-score-space",
            "confidence",
            "--corpus",
            str(corpus),
            "--raw-index-root",
            str(raw_index),
            "--teacher-checkpoint",
            str(teacher_path),
            "--teacher-score-space",
            "confidence",
            "--teacher-logit-cache",
            str(student_path_cache),
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
            str(edge_student_cache),
            "--continuous-edge-weight",
            "1",
            "--continuous-edge-batch-size",
            "64",
            "--distillation-weight",
            "0.3",
            "--edge-bce-weight",
            str(edge_bce_weight),
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
            "--epochs",
            "10",
            "--max-optimizer-updates",
            "356",
            "--batch-size",
            "64",
            "--patience",
            "0",
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
            "--seed",
            "13",
            "--device",
            "cuda:0",
            "--output",
            str(student_path_output),
            *aggregation_flags,
        ]
        _run_stage(
            student_path_command,
            student_path_output,
            cell_root / "student_path.log",
            root,
        )
        selected_student = student_path_output
        cells[cell_id] = {
            "aggregation_name": aggregation_name,
            "aggregation": config,
            "teacher_checkpoint": str(teacher_path),
            "teacher_checkpoint_sha256": checkpoint_fingerprint(teacher_path),
            "hard_targets": str(scored_targets),
            "hard_targets_sha256": checkpoint_fingerprint(scored_targets),
            "student_selected_checkpoint": str(selected_student),
            "student_selected_checkpoint_sha256": checkpoint_fingerprint(
                selected_student
            ),
            "student_endpoint_checkpoint": str(
                _last_checkpoint(student_path_output)
            ),
            "student_endpoint_checkpoint_sha256": checkpoint_fingerprint(
                _last_checkpoint(student_path_output)
            ),
        }

    _write_manifest(
        row_root / "manifest.json",
        row=args.row,
        edge_bce_weight=edge_bce_weight,
        edge_teacher=edge_teacher,
        edge_student=edge_student,
        cells=cells,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_r10 = root / "work" / "stage1_optimization_r10_20260907"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", required=True, choices=sorted(CELL_IDS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--r10-root", default=str(default_r10))
    parser.add_argument(
        "--output-root",
        default=str(default_r10 / "taskE_matched"),
    )
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
