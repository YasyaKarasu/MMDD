#!/usr/bin/env python
"""Run one reproducible R11 Task-C clean-Teacher or Student probe chain."""

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


STUDENT_ARMS = {
    "c0": ("raw_logit", 1.0),
    "c1": ("confidence", 1.0),
    "c2": ("confidence", 0.1),
}


def _last(path: Path) -> Path:
    return path.with_name(path.stem + ".last.pt")


def _history(path: Path) -> Path:
    return Path(str(path) + ".history.json")


def _selection(path: Path) -> Path:
    return Path(str(path) + ".selection.json")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(command: list[str], output: Path, log: Path, root: Path) -> float:
    if _selection(output).is_file() and _last(output).is_file():
        return 0.0
    started = time.monotonic()
    run_logged_subprocess(command, log, root=root)
    elapsed = time.monotonic() - started
    if not _selection(output).is_file() or not _last(output).is_file():
        raise RuntimeError(f"Incomplete training artifacts: {output}")
    return elapsed


def _compact_retrieval(record: dict[str, Any]) -> dict[str, Any] | None:
    metrics = record.get("dev_retrieval")
    if not isinstance(metrics, dict):
        return None
    funnel = metrics.get("evidence_funnel", {})
    return {
        "recall@10": metrics.get("recall@10"),
        "direct_recall@10": metrics.get("direct", {}).get("recall@10"),
        "evidence_recall@10": metrics.get("evidence", {}).get("recall@10"),
        "valid_path_recall@10,4": metrics.get("valid_path_recall@10,4"),
        "evidence_funnel": {
            key: funnel.get(key)
            for key in (
                "implicit_positive_pairs",
                "valid_pool_count",
                "valid_pool",
                "valid_b_count",
                "valid_b",
                "row_b",
                "q_to_e_pair_recall",
                "e_to_t_pair_recall_given_q_to_e",
            )
        },
    }


def _stage_record(path: Path, elapsed: float) -> dict[str, Any]:
    history = _load_json(_history(path))
    epochs = history.get("epochs", [])
    endpoint = epochs[-1] if epochs else {}
    return {
        "elapsed_seconds_this_invocation": elapsed,
        "selected_checkpoint": str(path),
        "selected_checkpoint_sha256": checkpoint_fingerprint(path),
        "endpoint_checkpoint": str(_last(path)),
        "endpoint_checkpoint_sha256": checkpoint_fingerprint(_last(path)),
        "best_epoch": history.get("best_epoch"),
        "stop_reason": history.get("stop_reason"),
        "endpoint": {
            key: endpoint.get(key)
            for key in (
                "epoch",
                "optimizer_updates",
                "cumulative_optimizer_updates",
                "optimizer_update_budget_exhausted",
                "loss",
                "dev_loss",
                "dev_edge",
                "dev_target_lists",
                "in_batch_expansion",
                "projection_drift",
                "relation_drift",
            )
            if key in endpoint
        },
        "endpoint_retrieval": _compact_retrieval(endpoint),
    }


def _paths(root: Path) -> dict[str, Path]:
    r10 = root / "work" / "stage1_optimization_r10_20260907"
    r11 = root / "work" / "stage1_optimization_r11_20260908"
    supervision = r11 / "taskA_protocol" / "supervision"
    return {
        "r11": r11,
        "features": r10 / "features_qwen3_vl_embedding_8b",
        "corpus": r10 / "stage1_data" / "stage1_corpus.jsonl",
        "raw_index": r10 / "taskA_protocol" / "baselines" / "raw_index",
        "pca_init": r11 / "taskA_protocol" / "baselines" / "pca_init.pt",
        "edge_train": supervision / "edge_lists.train_fit.jsonl",
        "edge_dev": supervision / "edge_lists.dev.jsonl",
        "path_train": supervision / "target_lists.train_fit.jsonl",
        "path_dev": supervision / "target_lists.dev.jsonl",
    }


def _path_flags() -> list[str]:
    return [
        "--evidence-aggregation", "logsumexp",
        "--evidence-temperature", "1",
        "--path-combination", "sum",
        "--evidence-threshold", "0",
        "--evidence-target-temperature", "1",
    ]


def _retrieval_flags(paths: dict[str, Path], index_root: Path) -> list[str]:
    return [
        "--corpus", str(paths["corpus"]),
        "--raw-index-root", str(paths["raw_index"]),
        "--index-root", str(index_root),
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
    ]


def run_teacher(args: argparse.Namespace, root: Path, paths: dict[str, Path]) -> None:
    teacher_name = "teacher" if args.seed == 13 else f"teacher_seed{args.seed}"
    out = paths["r11"] / "taskC_clean" / teacher_name
    out.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    train = str(root / "src" / "train_stage1.py")
    common = [
        "--features", str(paths["features"]),
        "--teacher-score-space", "raw_logit",
        "--teacher-dim", "512",
        "--teacher-layers", "3",
        "--teacher-heads", "8",
        "--text-latents", "16",
        "--image-latents", "24",
        "--dropout", "0.1",
        "--learning-rate", "5e-5",
        "--weight-decay", "0.01",
        "--batch-size", "8",
        "--teacher-amp", "off",
        "--seed", str(args.seed),
        "--patience", "0",
        "--device", "cuda:0",
    ]
    edge = out / "teacher_edge.pt"
    edge_command = [
        python, train, "teacher-edge",
        "--base-data", str(paths["edge_train"]),
        "--dev-data", str(paths["edge_dev"]),
        "--epochs", "2",
        "--feature-cache-size", "24000",
        "--selection-order", "dev_edge.macro_recall@1:max",
        "--selection-order", "dev_loss:min",
        "--output", str(edge),
        *common,
    ]
    edge_elapsed = _run(edge_command, edge, out / "teacher_edge.log", root)

    path = out / "teacher_path.pt"
    path_command = [
        python, train, "teacher-path",
        "--base-data", str(paths["path_train"]),
        "--dev-data", str(paths["path_dev"]),
        "--teacher-checkpoint", str(edge),
        "--epochs", "2",
        "--feature-cache-size", "32000",
        "--selection-order", "dev_target_lists.evidence.recall@1:max",
        "--selection-order", "dev_target_lists.direct.recall@1:max",
        "--selection-order", "dev_loss:min",
        "--output", str(path),
        *common,
        *_path_flags(),
    ]
    path_elapsed = _run(path_command, path, out / "teacher_path.log", root)
    _write_json(
        out / "manifest.json",
        {
            "format_version": 1,
            "experiment": "R11 Task C source-clean token Teacher",
            "seed": args.seed,
            "gpu": args.gpu,
            "commands": {"edge": edge_command, "path": path_command},
            "edge": _stage_record(edge, edge_elapsed),
            "path": _stage_record(path, path_elapsed),
        },
    )


def _student_common(paths: dict[str, Path], *, seed: int) -> list[str]:
    return [
        "--features", str(paths["features"]),
        "--seed", str(seed),
        "--student-score-space", "raw_logit",
        "--positive-loss-mode", "sum_probability",
        "--no-freeze-projection",
        "--distillation-weight", "0.3",
        "--temperature", "1",
        "--edge-bce-weight", "0",
        "--in-batch-negatives",
        "--in-batch-max-negatives", "256",
        "--global-positive-mask",
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


def run_student(args: argparse.Namespace, root: Path, paths: dict[str, Path]) -> None:
    teacher = paths["r11"] / "taskC_clean" / "teacher"
    edge_teacher = teacher / "teacher_edge.pt"
    path_teacher = teacher / "teacher_path.pt"
    if not edge_teacher.is_file() or not path_teacher.is_file():
        raise FileNotFoundError("Run the clean Teacher chain before Student probes")
    suffix = "_long" if args.long else ""
    seed_suffix = "" if args.seed == 13 else f"_seed{args.seed}"
    out = paths["r11"] / "taskC_clean" / f"{args.arm}{suffix}{seed_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    train = str(root / "src" / "train_stage1.py")
    common = _student_common(paths, seed=args.seed)
    commands: dict[str, list[str]] = {}
    records: dict[str, Any] = {}

    if args.arm in STUDENT_ARMS:
        ranking_space, ranking_temperature = STUDENT_ARMS[args.arm]
        edge = out / "student_edge.pt"
        edge_command = [
            python, train, "student-edge",
            "--base-data", str(paths["edge_train"]),
            "--dev-data", str(paths["edge_dev"]),
            "--student-checkpoint", str(paths["pca_init"]),
            "--teacher-checkpoint", str(edge_teacher),
            "--teacher-score-space", "raw_logit",
            "--teacher-logit-cache", str(teacher / "teacher_logits" / "edge"),
            "--teacher-logit-batch-size", "32",
            "--teacher-amp", "off",
            "--edge-ranking-weight", "1",
            "--edge-ranking-score-space", ranking_space,
            "--edge-ranking-temperature", str(ranking_temperature),
            "--epochs", "2" if args.long else "1",
            "--selection-order", "dev_edge.macro_recall@1:max",
            "--selection-order", "dev_loss:min",
            "--output", str(edge),
            *common,
        ]
        if not args.long:
            edge_command.extend(["--max-optimizer-updates", "178"])
        commands["edge"] = edge_command
        elapsed = _run(edge_command, edge, out / "student_edge.log", root)
        records["edge"] = _stage_record(edge, elapsed)
        evaluation_source = _last(edge)
        needs_endpoint_eval = not args.long
        if args.long:
            path = out / "student_path.pt"
            path_command = [
                python, train, "student-path",
                "--base-data", str(paths["path_train"]),
                "--dev-data", str(paths["path_dev"]),
                "--student-checkpoint", str(evaluation_source),
                "--teacher-checkpoint", str(path_teacher),
                "--teacher-score-space", "raw_logit",
                "--teacher-logit-cache", str(teacher / "teacher_logits" / "path"),
                "--teacher-logit-batch-size", "32",
                "--teacher-amp", "off",
                "--epochs", "2",
                "--max-optimizer-updates", "356",
                "--selection-order", "evidence_funnel.valid_pool_count:max",
                "--selection-order", "evidence_funnel.row_b:max",
                "--selection-order", "evidence_funnel.valid_b_count:max",
                "--selection-order", "direct.recall@10:max",
                "--output", str(path),
                *common,
                *_path_flags(),
                *_retrieval_flags(paths, out / "student_path.dev_indices"),
            ]
            commands["path"] = path_command
            elapsed = _run(path_command, path, out / "student_path.log", root)
            records["path"] = _stage_record(path, elapsed)
    else:
        path = out / "student_path.pt"
        path_command = [
            python, train, "student-path",
            "--base-data", str(paths["path_train"]),
            "--dev-data", str(paths["path_dev"]),
            "--student-checkpoint", str(paths["pca_init"]),
            "--teacher-checkpoint", str(path_teacher),
            "--teacher-score-space", "raw_logit",
            "--teacher-logit-cache", str(teacher / "teacher_logits" / "path"),
            "--teacher-logit-batch-size", "32",
            "--teacher-amp", "off",
            "--epochs", "1",
            "--max-optimizer-updates", "178",
            "--selection-order", "evidence_funnel.valid_pool_count:max",
            "--selection-order", "evidence_funnel.row_b:max",
            "--selection-order", "evidence_funnel.valid_b_count:max",
            "--selection-order", "direct.recall@10:max",
            "--output", str(path),
            *common,
            *_path_flags(),
            *_retrieval_flags(paths, out / "student_path.dev_indices"),
        ]
        commands["path"] = path_command
        elapsed = _run(path_command, path, out / "student_path.log", root)
        records["path"] = _stage_record(path, elapsed)
        evaluation_source = _last(path)
        needs_endpoint_eval = False

    if needs_endpoint_eval:
        evaluation = out / "endpoint_eval.pt"
        evaluation_command = [
            python, train, "student-path",
            "--base-data", str(paths["path_train"]),
            "--dev-data", str(paths["path_dev"]),
            "--student-checkpoint", str(evaluation_source),
            "--features", str(paths["features"]),
            "--distillation-weight", "0",
            "--no-in-batch-negatives",
            "--initialize-only",
            "--eval-epoch-zero",
            "--selection-order", "evidence_funnel.valid_pool_count:max",
            "--selection-order", "evidence_funnel.row_b:max",
            "--selection-order", "evidence_funnel.valid_b_count:max",
            "--selection-order", "direct.recall@10:max",
            "--output", str(evaluation),
            "--seed", str(args.seed),
            "--student-score-space", "raw_logit",
            "--batch-size", "64",
            "--device", "cuda:0",
            *_path_flags(),
            *_retrieval_flags(paths, out / "endpoint_eval.dev_indices"),
        ]
        commands["endpoint_eval"] = evaluation_command
        elapsed = _run(
            evaluation_command, evaluation, out / "endpoint_eval.log", root
        )
        records["endpoint_eval"] = _stage_record(evaluation, elapsed)
    _write_json(
        out / "manifest.json",
        {
            "format_version": 1,
            "experiment": "R11 Task C Student fixed-budget probe",
            "arm": args.arm,
            "regime": "long" if args.long else "probe",
            "seed": args.seed,
            "gpu": args.gpu,
            "commands": commands,
            "fixed_inputs": {
                "pca_initialization": str(paths["pca_init"]),
                "pca_initialization_sha256": checkpoint_fingerprint(paths["pca_init"]),
                "edge_teacher_sha256": checkpoint_fingerprint(edge_teacher),
                "path_teacher_sha256": checkpoint_fingerprint(path_teacher),
            },
            **records,
        },
    )


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    paths = _paths(root)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    required = [
        paths["features"] / "merge_summary.json",
        paths["corpus"],
        paths["raw_index"] / "manifest.json",
        paths["edge_train"],
        paths["edge_dev"],
        paths["path_train"],
        paths["path_dev"],
        paths["pca_init"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Task-C inputs:\n" + "\n".join(missing))
    if args.arm == "teacher":
        run_teacher(args, root, paths)
    else:
        run_student(args, root, paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("teacher", *STUDENT_ARMS, "path_only"), required=True
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--long",
        action="store_true",
        help="Run two edge epochs followed by 356 raw-score path updates.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parents[1])
    )
    args = parser.parse_args()
    if args.arm in {"teacher", "path_only"} and args.long:
        parser.error("--long is only valid for c0/c1/c2")
    return args


if __name__ == "__main__":
    run(parse_args())
