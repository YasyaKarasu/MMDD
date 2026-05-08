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
                seed=args.seed,
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
