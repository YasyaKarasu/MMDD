#!/usr/bin/env python
"""Run one R11 Task-B historical global-positive-mask diagnostic arm."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mmdd_stage1.experiment_process import run_logged_subprocess
from mmdd_stage1.retrieval import checkpoint_fingerprint


def _last_checkpoint(path: Path) -> Path:
    return path.with_name(path.stem + ".last.pt")


def _selection_path(path: Path) -> Path:
    return Path(str(path) + ".selection.json")


def _history_path(path: Path) -> Path:
    return Path(str(path) + ".history.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(command: list[str], output: Path, log: Path, root: Path) -> float:
    if _selection_path(output).is_file() and _last_checkpoint(output).is_file():
        return 0.0
    started = time.monotonic()
    run_logged_subprocess(command, log, root=root)
    elapsed = time.monotonic() - started
    if not _selection_path(output).is_file() or not _last_checkpoint(output).is_file():
        raise RuntimeError(f"Training did not produce complete artifacts: {output}")
    return elapsed


def _stage_record(output: Path, elapsed_seconds: float) -> dict[str, Any]:
    history = _load_json(_history_path(output))
    selection = _load_json(_selection_path(output))
    epochs = history.get("epochs", [])
    return {
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "selected_checkpoint": str(output),
        "selected_checkpoint_sha256": checkpoint_fingerprint(output),
        "endpoint_checkpoint": str(_last_checkpoint(output)),
        "endpoint_checkpoint_sha256": checkpoint_fingerprint(_last_checkpoint(output)),
        "best_epoch": selection.get("best_epoch"),
        "best_metrics": selection.get("best_metrics"),
        "endpoint_record": epochs[-1] if epochs else None,
    }


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    r10 = root / "work" / "stage1_optimization_r10_20260907"
    r11 = root / "work" / "stage1_optimization_r11_20260908"
    supervision = r11 / "taskA_protocol" / "supervision"
    output_root = r11 / "taskB_historical_mask" / args.arm
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    previous_manifest = _load_json(manifest_path) if manifest_path.is_file() else {}

    def recorded_elapsed(stage: str, current: float) -> float:
        if current:
            return current
        previous = previous_manifest.get(stage, {})
        return float(previous.get("elapsed_seconds_this_invocation", 0.0))

    features = r10 / "features_qwen3_vl_embedding_8b"
    corpus = r10 / "stage1_data" / "stage1_corpus.jsonl"
    raw_index = r10 / "taskA_protocol" / "baselines" / "raw_index"
    edge_train = supervision / "edge_lists.train_fit.jsonl"
    edge_dev = supervision / "edge_lists.dev.jsonl"
    path_train = supervision / "target_lists.train_fit.jsonl"
    path_dev = supervision / "target_lists.dev.jsonl"
    student_init = (
        r10
        / "taskC_pr_ablation"
        / "c4_r_warmup_then_p1e-6_r1e-5"
        / "phase2_unfrozen"
        / "student_path.pt"
    )
    edge_teacher = r10 / "taskE_matched" / "t0" / "teacher_edge.last.pt"
    path_teacher = (
        r10 / "taskE_matched" / "t0" / "e01" / "teacher_path.last.pt"
    )
    required = [
        features / "merge_summary.json",
        corpus,
        raw_index / "manifest.json",
        edge_train,
        edge_dev,
        path_train,
        path_dev,
        student_init,
        edge_teacher,
        path_teacher,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Task-B inputs:\n" + "\n".join(missing))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    python = sys.executable
    train = str(root / "src" / "train_stage1.py")
    mask_flag = "--global-positive-mask" if args.arm == "b1" else "--no-global-positive-mask"
    common = [
        "--seed", "13",
        "--student-confidence-transform",
        "--student-score-space", "confidence",
        "--positive-loss-mode", "sum_probability",
        "--no-freeze-projection",
        "--distillation-weight", "0.3",
        "--edge-bce-weight", "0",
        "--in-batch-negatives",
        "--in-batch-max-negatives", "256",
        mask_flag,
        "--anchor-weight", "0.1",
        "--anchor-weight-evidence", "0.1",
        "--learning-rate", "1e-6",
        "--relation-learning-rate", "1e-5",
        "--weight-decay", "0.01",
        "--dataset-sampling-alpha", "0",
        "--batch-size", "64",
        "--patience", "0",
        "--device", "cuda:0",
    ]

    edge_output = output_root / "student_edge.pt"
    edge_command = [
        python,
        train,
        "student-edge",
        "--features", str(features),
        "--base-data", str(edge_train),
        "--dev-data", str(edge_dev),
        "--student-checkpoint", str(student_init),
        "--teacher-checkpoint", str(edge_teacher),
        "--teacher-score-space", "confidence",
        "--teacher-logit-cache", str(output_root / "teacher_logits" / "edge"),
        "--teacher-logit-batch-size", "32",
        "--teacher-amp", "off",
        "--edge-ranking-weight", "1",
        "--epochs", "2",
        "--output", str(edge_output),
        *common,
    ]
    edge_elapsed = _run(
        edge_command, edge_output, output_root / "student_edge.log", root
    )

    edge_eval_output = output_root / "edge_endpoint_eval.pt"
    edge_eval_command = [
        python,
        train,
        "student-path",
        "--features", str(features),
        "--base-data", str(path_train),
        "--dev-data", str(path_dev),
        "--student-checkpoint", str(_last_checkpoint(edge_output)),
        "--corpus", str(corpus),
        "--raw-index-root", str(raw_index),
        "--index-root", str(output_root / "edge_endpoint_eval.dev_indices"),
        "--distillation-weight", "0",
        "--no-in-batch-negatives",
        "--initialize-only",
        "--eval-epoch-zero",
        "--selection-order", "evidence_funnel.valid_pool_count:max",
        "--selection-order", "evidence_funnel.row_b:max",
        "--selection-order", "evidence_funnel.valid_b_count:max",
        "--selection-order", "direct.recall@10:max",
        "--student-confidence-transform",
        "--student-score-space", "confidence",
        "--batch-size", "64",
        "--device", "cuda:0",
        "--recall-ks", "10,20,50",
        "--train-eval-ks", "10,20,50",
        "--direct-k", "100",
        "--evidence-k", "20",
        "--targets-per-evidence", "20",
        "--gamma", "10",
        "--gamma-evidence", "2",
        "--fusion-mode", "weighted_rrf",
        "--direct-weight", "1",
        "--evidence-weight", "0.05",
        "--evidence-aggregation", "logmeanexp",
        "--evidence-top-k", "4",
        "--evidence-temperature", "0.1",
        "--path-combination", "min",
        "--evidence-threshold", "0",
        "--evidence-target-temperature", "0.1",
        "--output", str(edge_eval_output),
    ]
    edge_eval_elapsed = _run(
        edge_eval_command,
        edge_eval_output,
        output_root / "edge_endpoint_eval.log",
        root,
    )

    path_output = output_root / "student_path.pt"
    path_command = [
        python,
        train,
        "student-path",
        "--features", str(features),
        "--base-data", str(path_train),
        "--dev-data", str(path_dev),
        "--student-checkpoint", str(_last_checkpoint(edge_output)),
        "--corpus", str(corpus),
        "--raw-index-root", str(raw_index),
        "--index-root", str(output_root / "student_path.dev_indices"),
        "--teacher-checkpoint", str(path_teacher),
        "--teacher-score-space", "confidence",
        "--teacher-logit-cache", str(output_root / "teacher_logits" / "path"),
        "--teacher-logit-batch-size", "32",
        "--teacher-amp", "off",
        "--continuous-edge-data", str(edge_train),
        "--continuous-edge-dev-data", str(edge_dev),
        "--continuous-edge-teacher-checkpoint", str(edge_teacher),
        "--continuous-edge-teacher-logit-cache", str(output_root / "teacher_logits" / "edge"),
        "--continuous-edge-weight", "1",
        "--continuous-edge-batch-size", "64",
        "--epochs", "1",
        "--max-optimizer-updates", "178",
        "--primary-metric", "valid_path_recall@10,4",
        "--recall-ks", "10,20,50",
        "--train-eval-ks", "10,20,50",
        "--direct-k", "100",
        "--evidence-k", "20",
        "--targets-per-evidence", "20",
        "--gamma", "10",
        "--gamma-evidence", "2",
        "--fusion-mode", "weighted_rrf",
        "--direct-weight", "1",
        "--evidence-weight", "0.05",
        "--evidence-aggregation", "logmeanexp",
        "--evidence-top-k", "4",
        "--evidence-temperature", "0.1",
        "--evidence-power", "2",
        "--path-combination", "min",
        "--evidence-threshold", "0",
        "--evidence-target-temperature", "0.1",
        "--output", str(path_output),
        *common,
    ]
    path_elapsed = _run(
        path_command, path_output, output_root / "student_path.log", root
    )

    _write_json(
        output_root / "manifest.json",
        {
            "format_version": 1,
            "experiment": "R11 Task B historical global-positive-mask diagnostic",
            "arm": args.arm,
            "global_positive_mask": args.arm == "b1",
            "gpu": args.gpu,
            "seed": 13,
            "scope": "Historical causal diagnostic; R10 weight ancestry is not source-clean.",
            "fixed_budget": {
                "edge_epochs": 2,
                "edge_updates_per_epoch_expected": 659,
                "path_optimizer_updates": 178,
            },
            "fixed_inputs": {
                "edge_train": str(edge_train),
                "edge_train_sha256": checkpoint_fingerprint(edge_train),
                "path_train": str(path_train),
                "path_train_sha256": checkpoint_fingerprint(path_train),
                "student_initialization": str(student_init),
                "student_initialization_sha256": checkpoint_fingerprint(student_init),
                "edge_teacher": str(edge_teacher),
                "edge_teacher_sha256": checkpoint_fingerprint(edge_teacher),
                "path_teacher": str(path_teacher),
                "path_teacher_sha256": checkpoint_fingerprint(path_teacher),
            },
            "commands": {
                "edge": edge_command,
                "edge_endpoint_eval": edge_eval_command,
                "path": path_command,
            },
            "edge": _stage_record(
                edge_output, recorded_elapsed("edge", edge_elapsed)
            ),
            "edge_endpoint_eval": _stage_record(
                edge_eval_output,
                recorded_elapsed("edge_endpoint_eval", edge_eval_elapsed),
            ),
            "path": _stage_record(
                path_output, recorded_elapsed("path", path_elapsed)
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("b0", "b1"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parents[1])
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
