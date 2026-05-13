#!/usr/bin/env python
"""Run Stage-1 teacher/HITL training and student ranking distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_hitl_training_rounds import run as run_hitl_rounds
from run_hitl_training_rounds import run_teacher_cycle
from stage1_gui import resolve_gui_host
from stage1_io import update_stage1_manifest
from stage1_training_cache import clear_training_outputs
from train_student import run as train_student


def train_teacher_and_hitl(args: argparse.Namespace) -> None:
    if getattr(args, "table_only", False):
        print("=== Stage-1 train: table-only teacher without HITL rounds ===")
        run_teacher_cycle(
            argparse.Namespace(
                stage1_dir=args.stage1_dir,
                embedding_dir=args.embedding_dir,
                include_pseudo_labels="false",
                pseudo_pos_threshold=args.pseudo_pos_threshold,
                pseudo_neg_threshold=args.pseudo_neg_threshold,
                seed=args.seed,
                teacher_epochs=args.teacher_epochs,
                teacher_batch_size=args.teacher_batch_size,
                teacher_lr=args.teacher_lr,
                path_composition=args.path_composition,
                table_only=True,
            ),
            round_id=0,
            final=True,
        )
        return
    if args.hitl_rounds > 0:
        run_hitl_rounds(
            argparse.Namespace(
                stage1_dir=args.stage1_dir,
                embedding_dir=args.embedding_dir,
                rounds=args.hitl_rounds,
                start_round=args.start_round,
                batch_size=args.hitl_batch_size,
                candidate_top_n=args.candidate_top_n,
                seed=args.seed,
                teacher_epochs=args.teacher_epochs,
                teacher_batch_size=args.teacher_batch_size,
                teacher_lr=args.teacher_lr,
                path_composition=args.path_composition,
                include_pseudo_labels=args.include_pseudo_labels,
                pseudo_pos_threshold=args.pseudo_pos_threshold,
                pseudo_neg_threshold=args.pseudo_neg_threshold,
                allow_reselect_previous=args.allow_reselect_previous,
                allow_reselect_labeled=args.allow_reselect_labeled,
                gui=args.gui,
                wait_for_labels=args.wait_for_labels,
                final_retrain=args.final_teacher_retrain,
                force_retrain=False,
                reset_human_labels=False,
                host=args.host,
                lan=getattr(args, "lan", False),
                port=args.port,
                poll_seconds=args.poll_seconds,
                wait_timeout_seconds=args.wait_timeout_seconds,
                table_only=False,
            )
        )
        return
    print("=== Stage-1 train: teacher without HITL rounds ===")
    run_teacher_cycle(
        argparse.Namespace(
            stage1_dir=args.stage1_dir,
            embedding_dir=args.embedding_dir,
            include_pseudo_labels=args.include_pseudo_labels,
            pseudo_pos_threshold=args.pseudo_pos_threshold,
            pseudo_neg_threshold=args.pseudo_neg_threshold,
            seed=args.seed,
            teacher_epochs=args.teacher_epochs,
            teacher_batch_size=args.teacher_batch_size,
            teacher_lr=args.teacher_lr,
            path_composition=args.path_composition,
            table_only=False,
        ),
        round_id=0,
        final=True,
    )


def run(args: argparse.Namespace) -> None:
    if args.embedding_dir is None:
        args.embedding_dir = str(Path(args.stage1_dir) / "embeddings")
    if args.student_dir is None:
        args.student_dir = str(Path(args.stage1_dir) / "student")
    args.host = resolve_gui_host(getattr(args, "host", None), getattr(args, "lan", False))
    if getattr(args, "force_retrain", False):
        removed = clear_training_outputs(Path(args.stage1_dir), Path(args.student_dir), getattr(args, "reset_human_labels", False))
        print(f"Force retrain cleanup removed {len(removed)} training artifact(s).")
    train_teacher_and_hitl(args)
    if args.hitl_rounds > 0 and args.gui and not args.wait_for_labels:
        print("Skipping student distillation because --no_wait_for_labels was used.")
        return
    if args.skip_student:
        print("Skipping student distillation because --skip_student was set.")
        return
    print("=== Stage-1 train: student ranking distillation ===")
    train_student(
        argparse.Namespace(
            stage1_dir=args.stage1_dir,
            embedding_dir=args.embedding_dir,
            teacher_scores=str(Path(args.stage1_dir) / "teacher_scores.jsonl"),
            output_dir=args.student_dir,
            epochs=args.student_epochs,
            batch_size=args.student_batch_size,
            lr=args.student_lr,
            student_dim=args.student_dim,
            distill_loss=args.distill_loss,
            ranking_temperature=args.ranking_temperature,
            max_pairs_per_group=args.max_pairs_per_group,
            pairwise_min_delta=args.pairwise_min_delta,
            seed=args.seed,
            progress=getattr(args, "progress", True),
            table_only=getattr(args, "table_only", False),
        )
    )
    update_stage1_manifest(Path(args.stage1_dir), "stage1_train_models", {"args": vars(args)})
    print(json.dumps({"stage1_dir": args.stage1_dir, "student_dir": args.student_dir}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--embedding_dir", default=None)
    parser.add_argument("--student_dir", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hitl_rounds", type=int, default=1)
    parser.add_argument("--start_round", type=int, default=None)
    parser.add_argument("--hitl_batch_size", type=int, default=50)
    parser.add_argument("--candidate_top_n", type=int, default=500)
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
    parser.add_argument("--final_teacher_retrain", dest="final_teacher_retrain", action="store_true", default=True)
    parser.add_argument("--no_final_teacher_retrain", dest="final_teacher_retrain", action="store_false")
    parser.add_argument("--force_retrain", action="store_true", help="Clear generated teacher/student/HITL training outputs before training.")
    parser.add_argument("--reset_human_labels", action="store_true", help="With --force_retrain, also delete merged human labels.")
    parser.add_argument("--table_only", action="store_true", help="Train only table-table connectivity; skip HITL/path samples and multimodal distillation.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--lan", action="store_true", help="Expose the HITL GUI on the LAN by binding to 0.0.0.0.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--poll_seconds", type=float, default=2.0)
    parser.add_argument("--wait_timeout_seconds", type=float, default=0.0)
    parser.add_argument("--skip_student", action="store_true")
    parser.add_argument("--student_epochs", type=int, default=10)
    parser.add_argument("--student_batch_size", type=int, default=512)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--student_dim", type=int, default=128)
    parser.add_argument("--distill_loss", choices=["pairwise", "listwise"], default="pairwise")
    parser.add_argument("--ranking_temperature", type=float, default=1.0)
    parser.add_argument("--max_pairs_per_group", type=int, default=2048)
    parser.add_argument("--pairwise_min_delta", type=float, default=1e-4)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
