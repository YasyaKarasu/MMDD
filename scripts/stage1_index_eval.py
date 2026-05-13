#!/usr/bin/env python
"""Build Stage-1 ANN indexes and run recall evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_hnsw_indices import run as build_hnsw_indices
from eval_stage1_recall import run as eval_stage1_recall
from stage1_io import update_stage1_manifest


def run(args: argparse.Namespace) -> None:
    if args.student_dir is None:
        args.student_dir = str(Path(args.stage1_dir) / "student")
    if args.hnsw_dir is None:
        args.hnsw_dir = str(Path(args.stage1_dir) / "hnsw_indices")
    if args.qrels is None:
        args.qrels = str(Path(args.stage1_dir) / "qrels.jsonl")
    if not args.skip_index:
        print("=== Stage-1 index/eval: build HNSW indexes ===")
        build_hnsw_indices(
            argparse.Namespace(
                stage1_dir=args.stage1_dir,
                student_dir=args.student_dir,
                space=args.hnsw_space,
                m=args.hnsw_m,
                ef_construction=args.hnsw_ef_construction,
                ef_search=args.hnsw_ef_search,
                table_only=getattr(args, "table_only", False),
            )
        )
    if not args.skip_eval:
        print("=== Stage-1 index/eval: recall evaluation ===")
        eval_stage1_recall(
            argparse.Namespace(
                stage1_dir=args.stage1_dir,
                student_dir=args.student_dir,
                hnsw_dir=args.hnsw_dir,
                qrels=args.qrels,
                topk=[str(k) for k in args.topk],
                max_hops=args.max_hops,
                beam_width=args.beam_width,
                beam_neighbors=args.beam_neighbors,
                path_composition=args.path_composition,
                progress=getattr(args, "progress", True),
                seed=args.seed,
                table_only=getattr(args, "table_only", False),
                table_hnsw_k=args.table_hnsw_k,
                recall_records=args.recall_records,
                write_recall_records=getattr(args, "write_recall_records", True),
            )
        )
    update_stage1_manifest(Path(args.stage1_dir), "stage1_index_eval", {"args": vars(args)})
    print(json.dumps({"stage1_dir": args.stage1_dir, "hnsw_dir": args.hnsw_dir}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--student_dir", default=None)
    parser.add_argument("--hnsw_dir", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--seed", type=int, default=13)
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
    parser.add_argument("--path_composition", choices=["min", "product"], default="min")
    parser.add_argument("--table_only", action="store_true", help="Build/evaluate only direct table-to-table retrieval; skip path-aware metrics.")
    parser.add_argument("--recall_records", default=None, help="Output JSONL path for per-query recalled targets and bridge paths.")
    parser.add_argument("--no_recall_records", dest="write_recall_records", action="store_false", help="Skip writing per-query recall_rankings.jsonl.")
    parser.add_argument(
        "--table_hnsw_k",
        type=int,
        default=0,
        help="right_target table neighbors to retrieve for table-only HNSW eval; 0 means all indexed target tables.",
    )
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True, write_recall_records=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
