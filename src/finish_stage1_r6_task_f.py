#!/usr/bin/env python
"""Finish Task F after two table-feature workers complete."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _run(command: Sequence[str], log_path: Path, *, environment: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            list(command),
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _wait_for_cache(cache: Path, expected: int, timeout_seconds: int) -> None:
    started = time.monotonic()
    while True:
        base = _line_count(cache / "manifest.jsonl")
        teacher = _line_count(cache / "teacher_manifest.jsonl")
        if base == expected and teacher == expected:
            return
        if base > expected or teacher > expected:
            raise ValueError(f"{cache}: feature manifest exceeds expected count {expected}")
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(
                f"{cache}: timed out at base={base}, teacher={teacher}, expected={expected}"
            )
        print(
            json.dumps(
                {
                    "event": "waiting_for_feature_shard",
                    "cache": str(cache),
                    "base": base,
                    "teacher": teacher,
                    "expected": expected,
                }
            ),
            flush=True,
        )
        time.sleep(30)


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    task_f = root / "work/stage1_optimization_r6_20260830/taskF_table_representation"
    work = task_f / "feature_work"
    cache = task_f / "features_qwen3_vl_embedding_8b"
    shard_counts = (20_228, 20_185)
    shard_caches = [work / f"cache_shard_{index:02d}" for index in range(2)]
    for shard_cache, expected in zip(shard_caches, shard_counts):
        _wait_for_cache(shard_cache, expected, args.timeout_seconds)

    free_bytes = shutil.disk_usage(root).free
    if free_bytes < args.minimum_free_gib * 1024**3:
        raise OSError(
            f"Task F requires at least {args.minimum_free_gib} GiB free before merge/evaluation; "
            f"found {free_bytes / 1024**3:.1f} GiB"
        )

    _run(
        [
            sys.executable,
            str(root / "src/merge_stage1_feature_cache.py"),
            "--cache-dir",
            str(cache),
            "--staging-dirs",
            *(str(path) for path in shard_caches),
        ],
        task_f / "merge.log",
    )
    _run(
        [
            sys.executable,
            str(root / "src/cache_stage1_features.py"),
            "--input-jsonl",
            str(work / "stage1_objects.jsonl"),
            "--output-dir",
            str(cache),
            "--model-dir",
            str(root / "hf_models/Qwen3-VL-Embedding-8B"),
            "--device",
            "cpu",
            "--dtype",
            "bf16",
            "--teacher-data",
            str(work / "table_shard_00.jsonl"),
            str(work / "table_shard_01.jsonl"),
            "--teacher-split",
            "all",
        ],
        task_f / "validate.log",
    )

    processes = []
    for gpu, lake in enumerate(("entitables", "wdc")):
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONPATH"] = str(root / "src")
        log_path = task_f / lake / "pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(root / "src/run_stage1_r6_task_f.py"),
            "--lake",
            lake,
            "--features",
            str(cache),
            "--objects",
            str(work / "stage1_objects.jsonl"),
            "--device",
            "cuda:0",
        ]
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((lake, process, handle))
    failures = []
    for lake, process, handle in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append((lake, return_code))
    if failures:
        raise RuntimeError(f"Task F lake evaluations failed: {failures}")

    _run(
        [
            sys.executable,
            str(root / "src/summarize_stage1_r6.py"),
            "finalize",
        ],
        task_f / "summarize.log",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "task_f": str(task_f.resolve()),
                "free_gib": shutil.disk_usage(root).free / 1024**3,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=10_800)
    parser.add_argument("--minimum-free-gib", type=int, default=30)
    values = parser.parse_args()
    if values.timeout_seconds <= 0 or values.minimum_free_gib <= 0:
        parser.error("timeout and minimum free space must be positive")
    return values


if __name__ == "__main__":
    run(parse_args())
