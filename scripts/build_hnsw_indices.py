#!/usr/bin/env python
"""Build one HNSW index per stage-1 object type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stage1_io import iter_jsonl, update_stage1_manifest, write_json

DATA_LAKE_SPLIT_MODE_CHOICES = ("auto", "strict", "query_corpus")


def right_target_fragment_ids(stage1_dir: Path) -> set[str]:
    return {
        rec["fragment_id"]
        for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl")
        if rec.get("role") == "right_target" and rec.get("fragment_id")
    }


def right_target_fragment_splits(stage1_dir: Path) -> dict[str, str]:
    return {
        rec["fragment_id"]: str(rec.get("split", "unknown"))
        for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl")
        if rec.get("role") == "right_target" and rec.get("fragment_id")
    }


def split_index_name(object_type: str, split: str) -> str:
    safe_split = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(split))
    return f"{object_type}_{safe_split or 'unknown'}"


def infer_data_lake_split_mode(stage1_dir: Path, requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    manifest_path = stage1_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mode = manifest.get("logic_connectivity", {}).get("data_lake_split_mode")
            if mode in {"strict", "query_corpus"}:
                return mode
        except json.JSONDecodeError:
            pass
    target_splits = {
        rec.get("split")
        for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl")
        if rec.get("role") == "right_target"
    }
    return "query_corpus" if "corpus" in target_splits else "strict"


def right_target_split_counts(stage1_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl"):
        if rec.get("role") != "right_target":
            continue
        split = str(rec.get("split", "unknown"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def save_hnsw_index(hnswlib: Any, args: argparse.Namespace, out_dir: Path, index_name: str, arr: np.ndarray, ids: list[str]) -> None:
    index = hnswlib.Index(space=args.space, dim=int(arr.shape[1]))
    index.init_index(max_elements=int(arr.shape[0]), ef_construction=args.ef_construction, M=args.m)
    labels = np.arange(arr.shape[0], dtype=np.int64)
    index.add_items(arr, labels)
    index.set_ef(args.ef_search)
    index.save_index(str(out_dir / f"{index_name}.bin"))
    write_json(out_dir / f"{index_name}_ids.json", ids)


def remove_hnsw_index_files(out_dir: Path, index_name: str) -> None:
    for path in (out_dir / f"{index_name}.bin", out_dir / f"{index_name}_ids.json"):
        if path.exists():
            path.unlink()


def remove_split_hnsw_index_files(out_dir: Path, object_type: str) -> None:
    for path in [*out_dir.glob(f"{object_type}_*.bin"), *out_dir.glob(f"{object_type}_*_ids.json")]:
        path.unlink()


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
    data_lake_split_mode = infer_data_lake_split_mode(stage1_dir, getattr(args, "data_lake_split_mode", "auto"))
    right_target_ids = right_target_fragment_ids(stage1_dir)
    right_target_splits = right_target_fragment_splits(stage1_dir)
    for object_type in object_types:
        npy = emb_dir / f"{object_type}.npy"
        ids_path = emb_dir / f"{object_type}_ids.json"
        if not npy.exists() or not ids_path.exists():
            continue
        arr = np.load(npy).astype("float32")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        indexed_role = None
        if object_type == "table_fragment":
            if data_lake_split_mode == "strict":
                remove_hnsw_index_files(out_dir, object_type)
                remove_split_hnsw_index_files(out_dir, object_type)
            else:
                remove_split_hnsw_index_files(out_dir, object_type)
            keep_indices = [idx for idx, object_id in enumerate(ids) if object_id in right_target_ids]
            arr = arr[keep_indices]
            ids = [ids[idx] for idx in keep_indices]
            indexed_role = "right_target"
        if arr.shape[0] == 0:
            continue
        if object_type == "table_fragment" and data_lake_split_mode == "strict":
            indices_by_split: dict[str, list[int]] = {}
            for idx, object_id in enumerate(ids):
                indices_by_split.setdefault(right_target_splits.get(object_id, "unknown"), []).append(idx)
            for split, split_indices in sorted(indices_by_split.items()):
                split_arr = arr[split_indices]
                split_ids = [ids[idx] for idx in split_indices]
                index_name = split_index_name(object_type, split)
                save_hnsw_index(hnswlib, args, out_dir, index_name, split_arr, split_ids)
                stats.append(
                    {
                        "object_type": object_type,
                        "index_name": index_name,
                        "split": split,
                        "count": len(split_ids),
                        "dimension": int(split_arr.shape[1]),
                        "indexed_role": indexed_role,
                    }
                )
            continue
        save_hnsw_index(hnswlib, args, out_dir, object_type, arr, ids)
        item = {"object_type": object_type, "index_name": object_type, "count": len(ids), "dimension": int(arr.shape[1])}
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
            "data_lake_split_mode": data_lake_split_mode,
            "target_split_counts": right_target_split_counts(stage1_dir),
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
    parser.add_argument(
        "--data_lake_split_mode",
        choices=DATA_LAKE_SPLIT_MODE_CHOICES,
        default="auto",
        help="Candidate data lake split mode recorded in HNSW stats; auto reads stage metadata.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
