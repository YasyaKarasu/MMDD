"""Encode text/image evidence (and optionally lake tables) into pure content tokens.

One process per GPU shard; shards are disjoint by object-id hash and each
publishes immutable chunk files atomically.  Prompts are the encoder's own
``EMBEDDING_INSTRUCTIONS``.  No task parameter is created, read or written here.

    python -m mmdd_stage1.content_encoder --dataset-root D --backbone-dir B \
        --cache-dir ENCODER --out-dir OUT --shard 0 --num-shards 2 --kinds text,image
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

from cache_stage1_features import EMBEDDING_INSTRUCTIONS

from . import content
from .construction import serialize_table_parts
from .data import iter_jsonl

# Preparation-stage imaging contract: decode whatever the dataset ships instead
# of letting PIL's heuristic bomb guard turn a real object into a NULL token
# (bad images are reported, never zero-filled).
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings  # noqa: E402

warnings.simplefilter("ignore", Image.DecompressionBombWarning)

TARGET_MAX_ROWS = 20
TARGET_MAX_CELL_CHARS = 1024


def shard_of(object_id: str, num_shards: int) -> int:
    return int.from_bytes(hashlib.sha256(object_id.encode("utf-8")).digest()[:8], "big") % num_shards


def load_source_tables(dataset_root: Path, needed: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in iter_jsonl(dataset_root / "source_tables" / "part-00000.jsonl"):
        sid = str(row["source_table_id"])
        if sid in needed:
            out[sid] = row
    return out


def expand_target(record: dict, sources: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    ref = record.get("source_table_ref")
    if ref:
        sid = ref["source_table_id"] if isinstance(ref, dict) else str(ref)
        if sid not in sources:
            raise KeyError(f"{record['table_id']}: unresolved source_table_ref {sid}")
        source = sources[sid]
        return source["columns"], source["rows"]
    return record["columns"], record["rows"]


def target_parts(record: dict, sources: dict[str, dict]) -> list[str]:
    columns, rows = expand_target(record, sources)
    return serialize_table_parts(
        {"columns": columns, "rows": rows},
        max_rows=TARGET_MAX_ROWS,
        max_cell_chars=TARGET_MAX_CELL_CHARS,
    )


def plan_work(dataset_root: Path, cache_dir: Path, kinds: set[str], shard: int, num_shards: int,
              *, max_objects: int = 0, skip: set[str] | None = None) -> list[dict]:
    covered = {
        e.object_id
        for e in content.read_manifest(cache_dir / "teacher_manifest.jsonl", "teacher_feature_path")
    }
    work: list[dict] = []
    if "text" in kinds or "image" in kinds:
        for shard_path in sorted((dataset_root / "bridge_assets").glob("part-*.jsonl")):
            for row in iter_jsonl(shard_path):
                kind = str(row["asset_type"])
                if kind not in kinds:
                    continue
                oid = str(row["asset_id"])
                if shard_of(oid, num_shards) != shard:
                    continue
                work.append(
                    {
                        "object_id": oid,
                        "object_type": kind,
                        "text": row.get("content") if kind == "text" else None,
                        "image": row.get("local_path") if kind == "image" else None,
                        "instruction": EMBEDDING_INSTRUCTIONS[("evidence", kind)],
                    }
                )
    if "table" in kinds:
        records = [r for r in iter_jsonl(dataset_root / "data_lake_tables" / "part-00000.jsonl")
                   if str(r["table_id"]) not in covered]
        refs = {
            (r["source_table_ref"]["source_table_id"] if isinstance(r.get("source_table_ref"), dict) else str(r["source_table_ref"]))
            for r in records
            if r.get("source_table_ref")
        }
        sources = load_source_tables(dataset_root, refs)
        for row in records:
            oid = str(row["table_id"])
            if shard_of(oid, num_shards) != shard:
                continue
            work.append(
                {
                    "object_id": oid,
                    "object_type": "table",
                    "parts": target_parts(row, sources),
                    "instruction": EMBEDDING_INSTRUCTIONS[("target", "table")],
                }
            )
    if skip:
        work = [w for w in work if w["object_id"] not in skip]
    if max_objects:
        work = work[:max_objects]
    return work


def existing_ids(chunk_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in sorted(Path(chunk_dir).glob("chunk_*.ids.npy")):
        ids.update(str(x) for x in np.load(path, allow_pickle=False))
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--backbone-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--text-batch", type=int, default=8)
    ap.add_argument("--image-batch", type=int, default=4)
    ap.add_argument("--max-objects", type=int, default=0)
    ap.add_argument("--kinds", default="text,image,table")
    ap.add_argument("--object-ids-file", default=None)
    args = ap.parse_args(argv)

    torch.cuda.set_device(torch.device(args.device))
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    skip = existing_ids(chunk_dir)
    work = plan_work(dataset_root, Path(args.cache_dir), set(args.kinds.split(",")),
                     args.shard, args.num_shards, max_objects=args.max_objects, skip=skip)
    if args.object_ids_file:
        wanted = {line.strip() for line in Path(args.object_ids_file).read_text().split() if line.strip()}
        work = [w for w in work if w["object_id"] in wanted]
    text_queue = [w for w in work if w["object_type"] == "text"]
    image_queue = [w for w in work if w["object_type"] == "image"]
    table_queue = [w for w in work if w["object_type"] == "table"]
    print(json.dumps({"event": "shard_plan", "shard": args.shard, "text": len(text_queue),
                      "image": len(image_queue), "table": len(table_queue)}), flush=True)
    if not work:
        print(json.dumps({"event": "shard_done", "shard": args.shard, "objects": 0}), flush=True)
        return 0

    embedder = content.load_embedder_class(Path(args.backbone_dir))(
        str(args.backbone_dir), max_length=8192, min_pixels=4096, max_pixels=1843200
    )
    failures = out_dir / f"failures_shard{args.shard}.jsonl"
    state = {"chunk": _next_chunk(chunk_dir), "rows": [], "done": 0, "skipped": len(skip)}

    def flush(force: bool = False) -> None:
        if state["rows"] and (force or len(state["rows"]) >= content.CHUNK_OBJECTS):
            content.write_chunk(chunk_dir, state["chunk"], state["rows"])
            state["chunk"] += 1
            state["rows"] = []

    def fail(object_id: str, reason: str) -> None:
        with failures.open("a") as fh:
            fh.write(json.dumps({"object_id": object_id, "reason": reason}) + "\n")

    def handle(batch: list[dict]) -> None:
        kind = batch[0]["object_type"]
        try:
            pairs = content.encode_hidden(embedder, [
                {"text": "\n".join(b["parts"]) if kind == "table" else b["text"],
                 "image": b["image"] if kind == "image" else None,
                 "instruction": b["instruction"]}
                for b in batch
            ])
        except Exception as error:  # external file/model boundary
            for b in batch:
                if b["object_type"] == "image" and not Path(str(b.get("image") or "")).is_file():
                    fail(b["object_id"], "missing_image_file")
                else:
                    fail(b["object_id"], f"{type(error).__name__}: {error}")
            return
        if len(pairs) != len(batch):
            # The official wrapper collapses a malformed vision batch into fewer
            # outputs; never let that silently drop a real object.
            if len(batch) > 1:
                for b in batch:
                    handle([b])
                return
            fail(batch[0]["object_id"], "wrapper_returned_no_output_for_single_object")
            return
        for b, (hidden, input_ids) in zip(batch, pairs):
            try:
                if kind == "table":
                    item = {"text": "\n".join(b["parts"]), "image": None, "instruction": b["instruction"]}
                    indices, groups = content.table_token_groups(embedder, item, b["parts"], input_ids)
                    tokens = content.pool_table_groups(hidden.index_select(0, indices), groups)
                else:
                    tokens = content.compress_bins(hidden)
            except Exception as error:
                fail(b["object_id"], f"{type(error).__name__}: {error}")
                continue
            state["rows"].append((b["object_id"], b["object_type"], tokens.numpy().astype(np.float16)))
            state["done"] += 1
        flush()
        if state["done"] and state["chunk"] > state.get("last_reported", state["chunk"] - 1):
            state["last_reported"] = state["chunk"]
            print(json.dumps({"event": "progress", "shard": args.shard, "done": state["done"],
                              "next_chunk": state["chunk"]}), flush=True)

    for queue, size, tag in ((text_queue, args.text_batch, "text"),
                             (image_queue, args.image_batch, "image"),
                             (table_queue, 1, "table")):
        for start in range(0, len(queue), size):
            handle(queue[start : start + size])
        flush(force=True)
        print(json.dumps({"event": "queue_done", "shard": args.shard, "kind": tag,
                          "next_chunk": state["chunk"]}), flush=True)

    print(json.dumps({"event": "shard_done", "shard": args.shard, "objects": state["done"]}), flush=True)
    return 2 if failures.exists() else 0


def _next_chunk(chunk_dir: Path) -> int:
    existing = sorted(Path(chunk_dir).glob("chunk_*.ids.npy"))
    return (int(existing[-1].name.split("_")[1].split(".")[0]) + 1) if existing else 0


if __name__ == "__main__":
    sys.exit(main())
