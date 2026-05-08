#!/usr/bin/env python
"""Build one HNSW index per stage-1 object type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage1_io import update_stage1_manifest, write_json


def run(args: argparse.Namespace) -> None:
    try:
        import hnswlib
    except ImportError as exc:
        raise SystemExit("hnswlib is required. Install it in MMDD, e.g. conda run -n MMDD python -m pip install hnswlib") from exc

    stage1_dir = Path(args.stage1_dir)
    emb_dir = Path(args.student_dir) / "index_embeddings"
    out_dir = stage1_dir / "hnsw_indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for object_type in ("table_fragment", "text_asset", "image_asset"):
        npy = emb_dir / f"{object_type}.npy"
        ids_path = emb_dir / f"{object_type}_ids.json"
        if not npy.exists() or not ids_path.exists():
            continue
        arr = np.load(npy).astype("float32")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if arr.shape[0] == 0:
            continue
        index = hnswlib.Index(space=args.space, dim=int(arr.shape[1]))
        index.init_index(max_elements=int(arr.shape[0]), ef_construction=args.ef_construction, M=args.m)
        labels = np.arange(arr.shape[0], dtype=np.int64)
        index.add_items(arr, labels)
        index.set_ef(args.ef_search)
        index.save_index(str(out_dir / f"{object_type}.bin"))
        write_json(out_dir / f"{object_type}_ids.json", ids)
        stats.append({"object_type": object_type, "count": len(ids), "dimension": int(arr.shape[1])})
    write_json(out_dir / "hnsw_stats.json", {"space": args.space, "m": args.m, "ef_construction": args.ef_construction, "ef_search": args.ef_search, "objects": stats})
    update_stage1_manifest(stage1_dir, "hnsw_indices", {"hnsw_dir": str(out_dir), "objects": stats, "args": vars(args)})
    print(json.dumps({"objects": stats, "hnsw_dir": str(out_dir)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--student_dir", default="output_stage1_logic/student")
    parser.add_argument("--space", "--hnsw_space", dest="space", default="cosine")
    parser.add_argument("--m", "--hnsw_m", dest="m", type=int, default=32)
    parser.add_argument("--ef_construction", "--hnsw_ef_construction", dest="ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", "--hnsw_ef_search", dest="ef_search", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
