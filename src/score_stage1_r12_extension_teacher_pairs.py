#!/usr/bin/env python
"""Score only the new Teacher pairs required by the R12 extension."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_teacher
from mmdd_stage1.features import FeatureStore
from score_stage1_r12_teacher_pairs import (
    _completed_parts,
    _save_part,
    _score_batch,
)


def _shard_count(start: int, end: int, shard_index: int, num_shards: int) -> int:
    first = start + ((shard_index - start) % num_shards)
    return 0 if first >= end else 1 + (end - 1 - first) // num_shards


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    started = time.monotonic()
    candidate_dir = (
        args.output_root
        / "taskC_training/candidates_seed13_extension_steps357_1318"
    )
    candidate_manifest_path = candidate_dir / "manifest.json"
    candidate_manifest = json.loads(
        candidate_manifest_path.read_text(encoding="utf-8")
    )
    pair_path = candidate_dir / "new_teacher_pairs.jsonl.gz"
    if candidate_manifest["new_pair_manifest_sha256"] != checkpoint_fingerprint(
        pair_path
    ):
        raise ValueError("Extension Teacher pair manifest fingerprint mismatch")
    budget = candidate_manifest["teacher_pair_budget"]
    pair_start = int(budget["global_pair_id_start"])
    pair_end = int(budget["global_pair_id_end_exclusive"])
    expected_total = _shard_count(
        pair_start, pair_end, args.shard_index, args.num_shards
    )

    output_dir = (
        args.output_root
        / "taskC_training/teacher_extension_pair_scores"
        / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print(json.dumps(manifest, indent=2))
            return manifest

    teacher_checkpoint = (
        args.root
        / "work/stage1_optimization_r11_20260908/taskC_clean/teacher/teacher_edge.pt"
    )
    base_features = (
        args.root
        / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
    )
    teacher_paths = [
        args.output_root / "taskC_training/teacher_extra",
        args.output_root / "taskC_training/teacher_extension_extra",
    ]
    store = FeatureStore.from_path(
        base_features,
        cache_size=args.feature_cache_size,
        teacher_paths=teacher_paths,
    )
    device = torch.device(args.device)
    teacher = load_teacher(teacher_checkpoint, device).eval()
    teacher.set_compute_dtype(None)
    compressed = {}
    parts, last_pair_id = _completed_parts(output_dir)
    next_part = len(parts)
    part_pair_ids = []
    part_scores = []
    batch = []
    scored_this_invocation = 0

    def flush_batch() -> None:
        nonlocal scored_this_invocation
        if not batch:
            return
        scores = _score_batch(batch, teacher, store, compressed, device)
        if scores.shape != (len(batch),) or not bool(torch.isfinite(scores).all()):
            raise ValueError("Teacher returned invalid extension pair scores")
        part_pair_ids.extend(int(row["pair_id"]) for row in batch)
        part_scores.append(scores)
        scored_this_invocation += len(batch)
        batch.clear()

    def flush_part() -> None:
        nonlocal next_part, part_pair_ids, part_scores
        if not part_pair_ids:
            return
        path = output_dir / f"part_{next_part:05d}.pt"
        _save_part(path, part_pair_ids, part_scores)
        parts.append(
            {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(path),
                "count": len(part_pair_ids),
                "first_pair_id": part_pair_ids[0],
                "last_pair_id": part_pair_ids[-1],
            }
        )
        next_part += 1
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "new_pairs_saved": sum(int(row["count"]) for row in parts),
                    "compressed_objects": len(compressed),
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
        part_pair_ids = []
        part_scores = []

    with gzip.open(pair_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pair_id = int(row["pair_id"])
            if not pair_start <= pair_id < pair_end:
                raise ValueError("Extension pair ID is outside the frozen suffix")
            if pair_id % args.num_shards != args.shard_index or pair_id <= last_pair_id:
                continue
            batch.append(row)
            if len(batch) == args.batch_size:
                flush_batch()
            if len(part_pair_ids) >= args.part_size:
                flush_part()
    flush_batch()
    flush_part()
    total_saved = sum(int(row["count"]) for row in parts)
    complete = total_saved == expected_total
    payload = {
        "format_version": 1,
        "status": "complete" if complete else "partial",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "global_pair_id_start": pair_start,
        "global_pair_id_end_exclusive": pair_end,
        "expected_pairs": expected_total,
        "saved_pairs": total_saved,
        "pair_manifest": str(pair_path.resolve()),
        "pair_manifest_sha256": checkpoint_fingerprint(pair_path),
        "candidate_manifest_sha256": checkpoint_fingerprint(
            candidate_manifest_path
        ),
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": checkpoint_fingerprint(teacher_checkpoint),
        "teacher_score_space": "raw_logit",
        "teacher_compute_dtype": "float32",
        "teacher_feature_manifests": {
            str(path.resolve()): checkpoint_fingerprint(
                path / "teacher_manifest.jsonl"
            )
            for path in teacher_paths
        },
        "parts": parts,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "completed_at_utc": (
            datetime.now(timezone.utc).isoformat() if complete else None
        ),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(manifest_path, payload)
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--part-size", type=int, default=50_000)
    parser.add_argument("--feature-cache-size", type=int, default=256)
    args = parser.parse_args()
    if min(args.num_shards, args.batch_size, args.part_size) <= 0:
        parser.error("Shard, batch, and part sizes must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
