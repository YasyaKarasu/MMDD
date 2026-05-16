#!/usr/bin/env python
"""Build one HNSW index per stage-1 object type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage1_io import iter_jsonl, update_stage1_manifest, write_json


def right_target_fragment_ids(stage1_dir: Path) -> set[str]:
    return {
        rec["fragment_id"]
        for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl")
        if rec.get("role") == "right_target" and rec.get("fragment_id")
    }


def run(args: argparse.Namespace) -> None:
    try:
        import hnswlib
    except ImportError as exc:
        raise SystemExit("hnswlib is required. Install it in MMDD, e.g. conda run -n MMDD python -m pip install hnswlib") from exc

    stage1_dir = Path(args.stage1_dir)
    raw_embedding_hnsw = bool(getattr(args, "raw_embedding_hnsw", False))
    emb_dir = Path(getattr(args, "embedding_dir", "") or stage1_dir / "embeddings") if raw_embedding_hnsw else Path(args.student_dir) / "index_embeddings"
    out_dir = Path(getattr(args, "hnsw_dir", "") or stage1_dir / "hnsw_indices")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    object_types = ("table_fragment",) if getattr(args, "table_only", False) else ("table_fragment", "text_asset", "image_asset")
    right_target_ids = right_target_fragment_ids(stage1_dir)
    for object_type in object_types:
        npy = emb_dir / f"{object_type}.npy"
        ids_path = emb_dir / f"{object_type}_ids.json"
        if not npy.exists() or not ids_path.exists():
            continue
        arr = np.load(npy).astype("float32")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        indexed_role = None
        if object_type == "table_fragment":
            keep_indices = [idx for idx, object_id in enumerate(ids) if object_id in right_target_ids]
            arr = arr[keep_indices]
            ids = [ids[idx] for idx in keep_indices]
            indexed_role = "right_target"
        if arr.shape[0] == 0:
            continue
        index = hnswlib.Index(space=args.space, dim=int(arr.shape[1]))
        index.init_index(max_elements=int(arr.shape[0]), ef_construction=args.ef_construction, M=args.m)
        labels = np.arange(arr.shape[0], dtype=np.int64)
        index.add_items(arr, labels)
        index.set_ef(args.ef_search)
        index.save_index(str(out_dir / f"{object_type}.bin"))
        write_json(out_dir / f"{object_type}_ids.json", ids)
        item = {"object_type": object_type, "count": len(ids), "dimension": int(arr.shape[1])}
        if indexed_role:
            item["indexed_role"] = indexed_role
        stats.append(item)
    write_json(
        out_dir / "hnsw_stats.json",
        {
            "space": args.space,
            "m": args.m,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "table_only": bool(getattr(args, "table_only", False)),
            "embedding_backend": "raw" if raw_embedding_hnsw else "student",
            "embedding_dir": str(emb_dir),
            "objects": stats,
        },
    )
    update_stage1_manifest(stage1_dir, "hnsw_indices", {"hnsw_dir": str(out_dir), "objects": stats, "args": vars(args)})
    print(json.dumps({"objects": stats, "hnsw_dir": str(out_dir)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--student_dir", default="output_stage1_logic/student")
    parser.add_argument("--embedding_dir", default=None)
    parser.add_argument("--hnsw_dir", default=None)
    parser.add_argument("--space", "--hnsw_space", dest="space", default="cosine")
    parser.add_argument("--m", "--hnsw_m", dest="m", type=int, default=32)
    parser.add_argument("--ef_construction", "--hnsw_ef_construction", dest="ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", "--hnsw_ef_search", dest="ef_search", type=int, default=100)
    parser.add_argument("--table_only", action="store_true", help="Build only the table_fragment index.")
    parser.add_argument("--raw_embedding_hnsw", action="store_true", help="Build HNSW indexes directly from frozen raw embeddings instead of student projections.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
