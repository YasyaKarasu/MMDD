#!/usr/bin/env python
"""Run teacher active-learning rounds with a Flask human annotation UI."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from build_teacher_training_data import run as build_teacher_training_data
from select_hitl_batch import run as select_hitl_batch
from stage1_io import iter_jsonl, load_json, update_stage1_manifest
from train_teacher import run as train_teacher


def existing_round_ids(stage1_dir: Path) -> list[int]:
    ids = []
    for path in stage1_dir.glob("human_labels_template_round_*.jsonl"):
        try:
            ids.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def next_round_id(stage1_dir: Path) -> int:
    ids = existing_round_ids(stage1_dir)
    return ids[-1] + 1 if ids else 0


def status_path(stage1_dir: Path, round_id: int) -> Path:
    return stage1_dir / f"hitl_round_{round_id}_annotation_status.json"


def is_round_merged(stage1_dir: Path, round_id: int) -> bool:
    path = status_path(stage1_dir, round_id)
    if not path.exists():
        selected_path = stage1_dir / f"hitl_selected_round_{round_id}.jsonl"
        human_path = stage1_dir / "human_labeled_paths.jsonl"
        if not selected_path.exists() or not human_path.exists():
            return False
        selected_ids = {rec["path_id"] for rec in iter_jsonl(selected_path) if rec.get("path_id")}
        human_ids = {rec["path_id"] for rec in iter_jsonl(human_path) if rec.get("path_id")}
        return bool(selected_ids) and selected_ids.issubset(human_ids)
    try:
        return bool(load_json(path).get("merged"))
    except (json.JSONDecodeError, OSError):
        return False


def run_teacher_cycle(args: argparse.Namespace, round_id: int, final: bool = False) -> None:
    train_pairs = Path(args.stage1_dir) / (f"train_pairs_final_round_{round_id}.jsonl" if final else f"train_pairs_round_{round_id}.jsonl")
    teacher_dir = Path(args.stage1_dir) / ("teacher_final" if final else f"teacher_round_{round_id}")
    build_teacher_training_data(
        argparse.Namespace(
            stage1_dir=args.stage1_dir,
            output=str(train_pairs),
            include_pseudo_labels=args.include_pseudo_labels,
            pseudo_pos_threshold=args.pseudo_pos_threshold,
            pseudo_neg_threshold=args.pseudo_neg_threshold,
            seed=args.seed,
        )
    )
    train_teacher(
        argparse.Namespace(
            stage1_dir=args.stage1_dir,
            embedding_dir=args.embedding_dir,
            train_pairs=str(train_pairs),
            output_dir=str(teacher_dir),
            epochs=args.teacher_epochs,
            batch_size=args.teacher_batch_size,
            lr=args.teacher_lr,
            path_composition=args.path_composition,
            seed=args.seed,
        )
    )


def select_round(args: argparse.Namespace, round_id: int) -> None:
    select_hitl_batch(
        argparse.Namespace(
            stage1_dir=args.stage1_dir,
            teacher_scores=str(Path(args.stage1_dir) / "teacher_scores.jsonl"),
            round_id=round_id,
            batch_size=args.batch_size,
            candidate_top_n=args.candidate_top_n,
            seed=args.seed + round_id,
            allow_reselect_previous=args.allow_reselect_previous,
            allow_reselect_labeled=args.allow_reselect_labeled,
        )
    )


def launch_gui(args: argparse.Namespace, round_id: int) -> subprocess.Popen[str]:
    script = Path(__file__).resolve().parent / "hitl_annotation_app.py"
    cmd = [
        sys.executable,
        str(script),
        "--stage1_dir",
        args.stage1_dir,
        "--round_id",
        str(round_id),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    proc = subprocess.Popen(cmd, text=True)
    print(f"Annotation GUI for round {round_id}: http://{args.host}:{args.port}")
    return proc


def wait_for_merge(args: argparse.Namespace, round_id: int, proc: subprocess.Popen[str] | None) -> None:
    deadline = None if args.wait_timeout_seconds <= 0 else time.time() + args.wait_timeout_seconds
    try:
        while True:
            if is_round_merged(Path(args.stage1_dir), round_id):
                return
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"Annotation GUI exited before round {round_id} was merged")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for human labels for round {round_id}")
            time.sleep(args.poll_seconds)
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def run(args: argparse.Namespace) -> None:
    stage1_dir = Path(args.stage1_dir)
    start = args.start_round if args.start_round is not None else next_round_id(stage1_dir)
    completed = []
    for offset in range(args.rounds):
        round_id = start + offset
        print(f"=== HITL round {round_id}: train teacher ===")
        run_teacher_cycle(args, round_id)
        print(f"=== HITL round {round_id}: select uncertain samples ===")
        select_round(args, round_id)
        proc = launch_gui(args, round_id) if args.gui else None
        if not args.gui:
            print(
                "Fill "
                f"{stage1_dir / f'human_labels_template_round_{round_id}.jsonl'} "
                "with labels, then run scripts/hitl_annotation_app.py or merge_human_labels.py."
            )
        if args.wait_for_labels:
            wait_for_merge(args, round_id, proc)
            completed.append(round_id)
        elif proc is not None:
            print(f"GUI started in process {proc.pid}; this runner is not waiting for labels.")
    if args.final_retrain and completed:
        print("=== Final teacher retrain with latest human labels ===")
        run_teacher_cycle(args, completed[-1], final=True)
    update_stage1_manifest(
        stage1_dir,
        "hitl_training_loop",
        {
            "rounds_requested": args.rounds,
            "start_round": start,
            "completed_rounds": completed,
            "final_retrain": bool(args.final_retrain and completed),
            "args": vars(args),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--embedding_dir", default="output_stage1_logic/embeddings")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--start_round", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--candidate_top_n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--teacher_epochs", type=int, default=5)
    parser.add_argument("--teacher_batch_size", type=int, default=256)
    parser.add_argument("--teacher_lr", type=float, default=1e-4)
    parser.add_argument("--path_composition", choices=["min", "product"], default="min")
    parser.add_argument("--include_pseudo_labels", default="false")
    parser.add_argument("--pseudo_pos_threshold", type=float, default=0.9)
    parser.add_argument("--pseudo_neg_threshold", type=float, default=0.1)
    parser.add_argument("--allow_reselect_previous", action="store_true")
    parser.add_argument("--allow_reselect_labeled", action="store_true")
    parser.add_argument("--gui", dest="gui", action="store_true", default=True)
    parser.add_argument("--no_gui", dest="gui", action="store_false")
    parser.add_argument("--wait_for_labels", dest="wait_for_labels", action="store_true", default=True)
    parser.add_argument("--no_wait_for_labels", dest="wait_for_labels", action="store_false")
    parser.add_argument("--final_retrain", dest="final_retrain", action="store_true", default=True)
    parser.add_argument("--no_final_retrain", dest="final_retrain", action="store_false")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--poll_seconds", type=float, default=2.0)
    parser.add_argument("--wait_timeout_seconds", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
