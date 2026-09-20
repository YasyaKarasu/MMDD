"""Compress the upstream pure token cache into this run's content contract.

Reads only the frozen full-token ``teacher_objects`` tier; writes immutable
chunks.  Tables keep the upstream per schema/row pooled tokens; text/image
token states become at most 64 consecutive bin means (SPEC 4.2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import features


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-objects", type=int, default=0)
    ap.add_argument("--exclude-prefix", action="append", default=[])
    args = ap.parse_args(argv)

    chunk_dir = Path(args.out_dir) / "chunks"
    entries = features.read_manifest(Path(args.cache_dir) / "teacher_manifest.jsonl", "teacher_feature_path")
    if args.exclude_prefix:
        entries = [e for e in entries if not e.object_id.startswith(tuple(args.exclude_prefix))]
    if args.max_objects:
        entries = entries[: args.max_objects]
    rows: list[tuple[str, str, "object"]] = []
    chunk = 0
    total_tokens = 0
    done = 0
    for entry in entries:
        import torch

        payload = torch.load(Path(args.cache_dir) / entry.path, map_location="cpu", weights_only=False)
        hidden = payload["hidden_states"].float()
        tokens = features.table_tokens(hidden) if entry.object_type == "table" else features.compress_bins(hidden)
        rows.append((entry.object_id, entry.object_type, tokens.numpy().astype("float16")))
        total_tokens += len(tokens)
        done += 1
        if len(rows) >= features.CHUNK_OBJECTS:
            features.write_chunk(chunk_dir, chunk, rows)
            chunk += 1
            rows = []
            print(json.dumps({"event": "chunk", "chunk": chunk, "done": done}), flush=True)
    if rows:
        features.write_chunk(chunk_dir, chunk, rows)
        chunk += 1
    print(json.dumps({"event": "extract_done", "objects": done, "tokens": total_tokens, "chunks": chunk}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
