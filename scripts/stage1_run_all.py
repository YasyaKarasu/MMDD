#!/usr/bin/env python
"""Run the consolidated Stage-1 pipeline end to end."""

from __future__ import annotations

import argparse

from stage1_index_eval import run as run_index_eval
from stage1_prepare_data import run as run_prepare_data
from stage1_gui import resolve_gui_host
from stage1_train_models import run as run_train_models


def run(args: argparse.Namespace) -> None:
    if args.embedding_dir is None:
        args.embedding_dir = f"{args.stage1_dir}/embeddings"
    if args.student_dir is None:
        args.student_dir = f"{args.stage1_dir}/student"
    if args.hnsw_dir is None:
        args.hnsw_dir = f"{args.stage1_dir}/hnsw_indices"
    if args.qrels is None:
        args.qrels = f"{args.stage1_dir}/qrels.jsonl"
    args.host = resolve_gui_host(getattr(args, "host", None), getattr(args, "lan", False))
    if not args.train_only:
        run_prepare_data(
            argparse.Namespace(
                input_dir=args.input_dir,
                stage1_dir=args.stage1_dir,
                seed=args.seed,
                min_rows_per_fragment=args.min_rows_per_fragment,
                max_chains_per_table=args.max_chains_per_table,
                max_bridges_per_anchor=args.max_bridges_per_anchor,
                max_target_attrs=args.max_target_attrs,
                max_query_context_attrs=args.max_query_context_attrs,
                min_ab_purity=args.min_ab_purity,
                min_bc_purity=args.min_bc_purity,
                min_support=args.min_support,
                max_bridge_unique_ratio=args.max_bridge_unique_ratio,
                skip_embeddings=args.skip_embeddings,
                encoder_path=args.encoder_path,
                embedding_batch_size=args.embedding_batch_size,
                device=args.device,
                dtype=args.dtype,
                max_table_rows=args.max_table_rows,
                max_text_chars=args.max_text_chars,
                max_image_pixels=args.max_image_pixels,
                force_recompute_embeddings=args.force_recompute_embeddings,
            )
        )
    if not args.prepare_only:
        run_train_models(
            argparse.Namespace(
                stage1_dir=args.stage1_dir,
                embedding_dir=args.embedding_dir,
                student_dir=args.student_dir,
                seed=args.seed,
                hitl_rounds=args.hitl_rounds,
                start_round=args.start_round,
                hitl_batch_size=args.hitl_batch_size,
                candidate_top_n=args.candidate_top_n,
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
                final_teacher_retrain=args.final_teacher_retrain,
                force_retrain=getattr(args, "force_retrain", False),
                reset_human_labels=getattr(args, "reset_human_labels", False),
                host=args.host,
                lan=getattr(args, "lan", False),
                port=args.port,
                poll_seconds=args.poll_seconds,
                wait_timeout_seconds=args.wait_timeout_seconds,
                skip_student=args.skip_student,
                student_epochs=args.student_epochs,
                student_batch_size=args.student_batch_size,
                student_lr=args.student_lr,
                student_dim=args.student_dim,
                distill_loss=args.distill_loss,
                ranking_temperature=args.ranking_temperature,
                max_pairs_per_group=args.max_pairs_per_group,
                pairwise_min_delta=args.pairwise_min_delta,
                progress=getattr(args, "progress", True),
            )
        )
    if not args.prepare_only and not args.train_only:
        run_index_eval(
            argparse.Namespace(
                stage1_dir=args.stage1_dir,
                student_dir=args.student_dir,
                hnsw_dir=args.hnsw_dir,
                qrels=args.qrels,
                seed=args.seed,
                skip_index=args.skip_index,
                skip_eval=args.skip_eval,
                hnsw_space=args.hnsw_space,
                hnsw_m=args.hnsw_m,
                hnsw_ef_construction=args.hnsw_ef_construction,
                hnsw_ef_search=args.hnsw_ef_search,
                topk=args.topk,
                max_hops=args.max_hops,
                beam_width=args.beam_width,
                beam_neighbors=args.beam_neighbors,
                path_composition=args.path_composition,
                progress=getattr(args, "progress", True),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output_medium")
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--embedding_dir", default=None)
    parser.add_argument("--student_dir", default=None)
    parser.add_argument("--hnsw_dir", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--min_rows_per_fragment", type=int, default=5)
    parser.add_argument("--max_chains_per_table", type=int, default=10)
    parser.add_argument("--max_bridges_per_anchor", type=int, default=5)
    parser.add_argument("--max_target_attrs", type=int, default=2)
    parser.add_argument("--max_query_context_attrs", type=int, default=2)
    parser.add_argument("--min_ab_purity", type=float, default=0.95)
    parser.add_argument("--min_bc_purity", type=float, default=0.85)
    parser.add_argument("--min_support", type=int, default=6)
    parser.add_argument("--max_bridge_unique_ratio", type=float, default=0.85)
    parser.add_argument("--skip_embeddings", action="store_true")
    parser.add_argument("--encoder_path", default="./Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max_table_rows", type=int, default=5)
    parser.add_argument("--max_text_chars", type=int, default=2048)
    parser.add_argument("--max_image_pixels", type=int, default=178_956_970)
    parser.add_argument("--force_recompute_embeddings", action="store_true")
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
    parser.add_argument("--skip_index", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--hnsw_space", default="cosine")
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--hnsw_ef_construction", type=int, default=200)
    parser.add_argument("--hnsw_ef_search", type=int, default=100)
    parser.add_argument("--topk", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--beam_width", type=int, default=64)
    parser.add_argument("--beam_neighbors", type=int, default=50)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
