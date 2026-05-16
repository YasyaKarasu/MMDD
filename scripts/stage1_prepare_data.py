#!/usr/bin/env python
"""Prepare all non-training Stage-1 artifacts from output_medium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_stage1_embeddings import run as build_embeddings
from build_stage1_evidence_paths import run as build_evidence_paths
from build_stage1_logic_connectivity import run as build_logic_connectivity
from generate_weak_labels import run as generate_weak_labels
from stage1_io import update_stage1_manifest


def run(args: argparse.Namespace) -> None:
    stage1_dir = Path(args.stage1_dir)
    if getattr(args, "webtable_mode", False):
        args.table_only = True
    print("=== Stage-1 prepare: logic connectivity ===")
    build_logic_connectivity(
        argparse.Namespace(
            input_dir=args.input_dir,
            output_dir=args.stage1_dir,
            min_rows_per_fragment=args.min_rows_per_fragment,
            max_chains_per_table=args.max_chains_per_table,
            max_bridges_per_anchor=args.max_bridges_per_anchor,
            max_target_attrs=args.max_target_attrs,
            max_query_context_attrs=args.max_query_context_attrs,
            seed=args.seed,
            min_ab_purity=args.min_ab_purity,
            min_bc_purity=args.min_bc_purity,
            min_support=args.min_support,
            max_bridge_unique_ratio=args.max_bridge_unique_ratio,
            table_only=getattr(args, "table_only", False),
            webtable_mode=getattr(args, "webtable_mode", False),
            webtable_query_file=getattr(args, "webtable_query_file", None),
            webtable_ground_truth_file=getattr(args, "webtable_ground_truth_file", None),
            webtable_table_dir=getattr(args, "webtable_table_dir", None),
            webtable_max_rows=getattr(args, "webtable_max_rows", 200),
            webtable_recursive_lookup=getattr(args, "webtable_recursive_lookup", False),
            webtable_split_ratios=getattr(args, "webtable_split_ratios", [0.7, 0.1, 0.2]),
            webtable_split_seed=getattr(args, "webtable_split_seed", args.seed),
        )
    )
    if getattr(args, "table_only", False):
        print("=== Stage-1 prepare: table-only mode skips evidence paths and weak labels ===")
    else:
        print("=== Stage-1 prepare: evidence paths ===")
        build_evidence_paths(argparse.Namespace(input_dir=args.input_dir, stage1_dir=args.stage1_dir, seed=args.seed))
        print("=== Stage-1 prepare: weak labels / HITL pool ===")
        generate_weak_labels(argparse.Namespace(stage1_dir=args.stage1_dir))
    if not args.skip_embeddings:
        print("=== Stage-1 prepare: frozen embeddings ===")
        build_embeddings(
            argparse.Namespace(
                input_dir=args.input_dir,
                stage1_dir=args.stage1_dir,
                encoder_path=args.encoder_path,
                batch_size=args.embedding_batch_size,
                device=args.device,
                dtype=args.dtype,
                max_table_rows=args.max_table_rows,
                max_text_chars=args.max_text_chars,
                max_image_pixels=args.max_image_pixels,
                embedding_prompt_mode=getattr(args, "embedding_prompt_mode", "connectivity"),
                force_recompute=args.force_recompute_embeddings,
                table_only=getattr(args, "table_only", False),
                progress=bool(getattr(args, "progress", True)),
            )
        )
    update_stage1_manifest(stage1_dir, "stage1_prepare_data", {"args": vars(args)})
    print(json.dumps({"stage1_dir": args.stage1_dir, "embeddings": not args.skip_embeddings}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output_medium")
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--seed", type=int, default=13)
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
    parser.add_argument(
        "--embedding_prompt_mode",
        choices=["connectivity", "content_only"],
        default="connectivity",
        help="Embedding instruction style: connectivity preserves cross-asset retrieval prompts; content_only encodes each table/text/image by its own content.",
    )
    parser.add_argument("--force_recompute_embeddings", action="store_true")
    parser.add_argument("--table_only", action="store_true", help="Build only table-fragment artifacts and embeddings; skip multimodal evidence artifacts.")
    parser.add_argument("--webtable_mode", action="store_true", help="Read WebTable benchmark format and automatically use table-only Stage-1 mode.")
    parser.add_argument("--webtable_query_file", default=None, help="Path to webtable_join_query.csv. Defaults to --input_dir/webtable_join_query.csv.")
    parser.add_argument("--webtable_ground_truth_file", default=None, help="Path to webtable_join_ground_truth.csv. Defaults to --input_dir/webtable_join_ground_truth.csv.")
    parser.add_argument("--webtable_table_dir", default=None, help="Directory containing WebTable CSV files. Defaults to --input_dir/data/benchmark/webtable/large/split_1.")
    parser.add_argument("--webtable_max_rows", type=int, default=200, help="Maximum rows loaded per WebTable CSV; 0 keeps all rows.")
    parser.add_argument("--webtable_recursive_lookup", action="store_true", help="Recursively search --webtable_table_dir when a listed CSV is not found directly.")
    parser.add_argument("--webtable_split_ratios", nargs=3, type=float, default=[0.7, 0.1, 0.2], metavar=("TRAIN", "DEV", "TEST"))
    parser.add_argument("--webtable_split_seed", type=int, default=13, help="Seed for deterministic WebTable query_table train/dev/test split.")
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
