#!/usr/bin/env python
"""Retrain and evaluate the per-lake Task F2 Teachers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _run(command: Sequence[str], log_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def _lake_inputs(root: Path, lake: str) -> dict[str, Path]:
    r4 = root / "work/stage1_optimization_r4_20260829"
    if lake == "entitables":
        teacher_root = r4 / "taskM_entitables_teacher"
        return {
            "edge_data": teacher_root / "data/edge_lists.jsonl",
            "path_data": teacher_root / "data/target_lists.jsonl",
            "dev_data": root
            / "work/stage1_stage2_entitables20k_v4_20260827/stage1_data/target_lists.jsonl",
            "corpus": r4
            / "taskJ_per_lake_baselines/corpora/entitables_corpus.jsonl",
            "raw_index": r4
            / "taskJ_per_lake_baselines/epoch0/entitables/raw_index",
        }
    teacher_root = r4 / "taskM_entitables_teacher/per_lake/wdc"
    return {
        "edge_data": teacher_root / "data/edge_lists.jsonl",
        "path_data": teacher_root / "data/target_lists.jsonl",
        "dev_data": root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data/target_lists.jsonl",
        "corpus": r4 / "taskJ_per_lake_baselines/corpora/wdc_corpus.jsonl",
        "raw_index": r4 / "taskJ_per_lake_baselines/epoch0/wdc/raw_index",
    }


def _complete(checkpoint: Path) -> bool:
    return checkpoint.is_file() and checkpoint.with_suffix(
        checkpoint.suffix + ".history.json"
    ).is_file()


def run(args: argparse.Namespace) -> dict[str, str | int | float]:
    root = Path(__file__).resolve().parents[1]
    inputs = _lake_inputs(root, args.lake)
    output_dir = (
        args.output_root
        / "taskF_table_representation"
        / f"tokens_per_group_{args.table_tokens_per_group}"
        / "retrained"
        / args.lake
    )
    checkpoints = output_dir / "checkpoints"
    edge_checkpoint = checkpoints / "teacher_edge.pt"
    path_checkpoint = checkpoints / "teacher_path.pt"

    common = [
        "--features",
        str(args.features),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--patience",
        str(args.patience),
        "--feature-cache-gb",
        str(args.feature_cache_gb),
        "--teacher-table-tokens-per-group",
        str(args.table_tokens_per_group),
    ]
    if not _complete(edge_checkpoint):
        _run(
            [
                sys.executable,
                str(root / "src/train_stage1.py"),
                "teacher-edge",
                "--base-data",
                str(inputs["edge_data"]),
                "--dev-data",
                str(inputs["edge_data"]),
                "--output",
                str(edge_checkpoint),
                "--feature-cache-size",
                "24000",
                *common,
            ],
            output_dir / "teacher_edge.log",
        )

    if not _complete(path_checkpoint):
        _run(
            [
                sys.executable,
                str(root / "src/train_stage1.py"),
                "teacher-path",
                "--base-data",
                str(inputs["path_data"]),
                "--dev-data",
                str(inputs["path_data"]),
                "--output",
                str(path_checkpoint),
                "--teacher-checkpoint",
                str(edge_checkpoint),
                "--corpus",
                str(inputs["corpus"]),
                "--raw-index-root",
                str(inputs["raw_index"]),
                "--feature-cache-size",
                "32000",
                "--teacher-rerank",
                "--teacher-rerank-dev-data",
                str(inputs["dev_data"]),
                "--teacher-rerank-top-k",
                "100",
                "--teacher-rerank-batch-size",
                str(args.teacher_rerank_batch_size),
                "--teacher-rerank-interval",
                "2",
                "--primary-metric",
                "teacher_rerank.recall@10",
                *common,
            ],
            output_dir / "teacher_path.log",
        )

    _run(
        [
            sys.executable,
            str(root / "src/run_stage1_r6_task_f_tokens.py"),
            "--lake",
            args.lake,
            "--features",
            str(args.features),
            "--objects",
            str(args.objects),
            "--device",
            args.device,
            "--table-tokens-per-group",
            str(args.table_tokens_per_group),
            "--teacher-batch-size",
            str(args.teacher_rerank_batch_size),
            "--teacher-checkpoint",
            str(path_checkpoint),
            "--evaluation-mode",
            "retrained",
            "--output-root",
            str(args.output_root),
        ],
        output_dir / "evaluation_pipeline.log",
    )
    payload: dict[str, str | int | float] = {
        "status": "complete",
        "lake": args.lake,
        "table_tokens_per_group": args.table_tokens_per_group,
        "edge_checkpoint": str(edge_checkpoint.resolve()),
        "path_checkpoint": str(path_checkpoint.resolve()),
        "summary": str((output_dir / "summary.json").resolve()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", choices=("entitables", "wdc"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--table-tokens-per-group", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--feature-cache-gb", type=float, default=24.0)
    parser.add_argument("--teacher-rerank-batch-size", type=int, default=16)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r6_20260830"),
    )
    values = parser.parse_args()
    if values.table_tokens_per_group <= 1:
        parser.error("--table-tokens-per-group must be greater than one")
    if values.epochs <= 0 or values.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if values.learning_rate <= 0 or values.feature_cache_gb <= 0:
        parser.error("learning rate and feature cache budget must be positive")
    if values.patience < 0 or values.teacher_rerank_batch_size <= 0:
        parser.error("patience must be non-negative and rerank batch size positive")
    return values


if __name__ == "__main__":
    run(parse_args())
