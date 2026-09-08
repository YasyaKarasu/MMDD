#!/usr/bin/env python
"""Run R11 Task-D KD-off and projection-frozen Student controls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from mmdd_stage1.retrieval import checkpoint_fingerprint
from run_stage1_r11_task_c import (
    STUDENT_ARMS,
    _last,
    _path_flags,
    _paths,
    _retrieval_flags,
    _run,
    _stage_record,
    _student_common,
    _write_json,
)


def _replace_flag(values: list[str], flag: str, replacement: str) -> None:
    index = values.index(flag)
    values[index + 1] = replacement


def _control_common(
    paths: dict[str, Path], *, arm: str, seed: int
) -> list[str]:
    values = _student_common(paths, seed=seed)
    if arm == "d1":
        _replace_flag(values, "--distillation-weight", "0")
    elif arm == "d2":
        values[values.index("--no-freeze-projection")] = "--freeze-projection"
    return values


def _fixed_inputs(
    paths: dict[str, Path], teacher: Path, ranking_arm: str
) -> dict[str, Any]:
    d0 = paths["r11"] / "taskC_clean" / ranking_arm / "manifest.json"
    return {
        "pca_initialization": str(paths["pca_init"]),
        "pca_initialization_sha256": checkpoint_fingerprint(paths["pca_init"]),
        "edge_teacher": str(teacher / "teacher_edge.pt"),
        "edge_teacher_sha256": checkpoint_fingerprint(teacher / "teacher_edge.pt"),
        "path_teacher": str(teacher / "teacher_path.pt"),
        "path_teacher_sha256": checkpoint_fingerprint(teacher / "teacher_path.pt"),
        "shared_d0_manifest": str(d0),
        "shared_d0_manifest_sha256": (
            checkpoint_fingerprint(d0) if d0.is_file() else None
        ),
    }


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    paths = _paths(root)
    teacher = paths["r11"] / "taskC_clean" / "teacher"
    required = [
        paths["pca_init"],
        paths["edge_train"],
        paths["edge_dev"],
        paths["path_train"],
        paths["path_dev"],
        teacher / "teacher_edge.pt",
        teacher / "teacher_path.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Task-D inputs:\n" + "\n".join(missing))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    python = sys.executable
    train = str(root / "src" / "train_stage1.py")
    ranking_space, ranking_temperature = STUDENT_ARMS[args.ranking_arm]
    regime = "long" if args.long else "probe"
    seed_suffix = "" if args.seed == 13 else f"_seed{args.seed}"
    out = (
        paths["r11"]
        / "taskD_controls"
        / f"{args.ranking_arm}_{args.arm}_{regime}{seed_suffix}"
    )
    out.mkdir(parents=True, exist_ok=True)
    common = _control_common(paths, arm=args.arm, seed=args.seed)
    commands: dict[str, list[str]] = {}
    records: dict[str, Any] = {}

    edge = out / "student_edge.pt"
    edge_command = [
        python,
        train,
        "student-edge",
        "--base-data",
        str(paths["edge_train"]),
        "--dev-data",
        str(paths["edge_dev"]),
        "--student-checkpoint",
        str(paths["pca_init"]),
        "--edge-ranking-weight",
        "1",
        "--edge-ranking-score-space",
        ranking_space,
        "--edge-ranking-temperature",
        str(ranking_temperature),
        "--epochs",
        "2" if args.long else "1",
        "--selection-order",
        "dev_edge.macro_recall@1:max",
        "--selection-order",
        "dev_loss:min",
        "--output",
        str(edge),
        *common,
    ]
    if args.arm == "d2":
        edge_command.extend(
            [
                "--teacher-checkpoint",
                str(teacher / "teacher_edge.pt"),
                "--teacher-score-space",
                "raw_logit",
                "--teacher-logit-cache",
                str(teacher / "teacher_logits" / "edge"),
                "--teacher-logit-batch-size",
                "32",
                "--teacher-amp",
                "off",
            ]
        )
    if not args.long:
        edge_command.extend(["--max-optimizer-updates", "178"])
    commands["edge"] = edge_command
    elapsed = _run(edge_command, edge, out / "student_edge.log", root)
    records["edge"] = _stage_record(edge, elapsed)

    if args.long:
        path = out / "student_path.pt"
        path_command = [
            python,
            train,
            "student-path",
            "--base-data",
            str(paths["path_train"]),
            "--dev-data",
            str(paths["path_dev"]),
            "--student-checkpoint",
            str(_last(edge)),
            "--epochs",
            "2",
            "--max-optimizer-updates",
            "356",
            "--selection-order",
            "evidence_funnel.valid_pool_count:max",
            "--selection-order",
            "evidence_funnel.row_b:max",
            "--selection-order",
            "evidence_funnel.valid_b_count:max",
            "--selection-order",
            "direct.recall@10:max",
            "--output",
            str(path),
            *common,
            *_path_flags(),
            *_retrieval_flags(paths, out / "student_path.dev_indices"),
        ]
        if args.arm == "d2":
            path_command.extend(
                [
                    "--teacher-checkpoint",
                    str(teacher / "teacher_path.pt"),
                    "--teacher-score-space",
                    "raw_logit",
                    "--teacher-logit-cache",
                    str(teacher / "teacher_logits" / "path"),
                    "--teacher-logit-batch-size",
                    "32",
                    "--teacher-amp",
                    "off",
                ]
            )
        commands["path"] = path_command
        elapsed = _run(path_command, path, out / "student_path.log", root)
        records["path"] = _stage_record(path, elapsed)
    else:
        evaluation = out / "endpoint_eval.pt"
        evaluation_command = [
            python,
            train,
            "student-path",
            "--base-data",
            str(paths["path_train"]),
            "--dev-data",
            str(paths["path_dev"]),
            "--student-checkpoint",
            str(_last(edge)),
            "--features",
            str(paths["features"]),
            "--distillation-weight",
            "0",
            "--no-in-batch-negatives",
            "--initialize-only",
            "--eval-epoch-zero",
            "--selection-order",
            "evidence_funnel.valid_pool_count:max",
            "--selection-order",
            "evidence_funnel.row_b:max",
            "--selection-order",
            "evidence_funnel.valid_b_count:max",
            "--selection-order",
            "direct.recall@10:max",
            "--output",
            str(evaluation),
            "--seed",
            str(args.seed),
            "--student-score-space",
            "raw_logit",
            "--batch-size",
            "64",
            "--device",
            "cuda:0",
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
            "experiment": "R11 Task D controlled Student ablation",
            "arm": args.arm,
            "ranking_arm": args.ranking_arm,
            "regime": regime,
            "seed": args.seed,
            "gpu": args.gpu,
            "single_factor_change": (
                "distillation_weight: 0.3 -> 0"
                if args.arm == "d1"
                else "projection: trainable -> frozen"
            ),
            "commands": commands,
            "fixed_inputs": _fixed_inputs(paths, teacher, args.ranking_arm),
            **records,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("d1", "d2"), required=True)
    parser.add_argument("--ranking-arm", choices=tuple(STUDENT_ARMS), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parents[1])
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
