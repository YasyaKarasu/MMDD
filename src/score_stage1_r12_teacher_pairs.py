#!/usr/bin/env python
"""Score the frozen R12 Teacher pair manifest in resumable GPU shards."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_teacher
from mmdd_stage1.features import FeatureStore, ObjectFeatures


def _save_part(
    path: Path,
    pair_ids: list[int],
    scores: list[torch.Tensor],
) -> None:
    payload = {
        "format_version": 1,
        "pair_ids": torch.tensor(pair_ids, dtype=torch.int64),
        "scores": torch.cat(scores).float(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _completed_parts(output_dir: Path) -> tuple[list[dict[str, Any]], int]:
    parts = []
    last_pair_id = -1
    for path in sorted(output_dir.glob("part_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        pair_ids = payload.get("pair_ids") if isinstance(payload, dict) else None
        scores = payload.get("scores") if isinstance(payload, dict) else None
        if (
            not isinstance(pair_ids, torch.Tensor)
            or not isinstance(scores, torch.Tensor)
            or pair_ids.ndim != 1
            or scores.ndim != 1
            or pair_ids.shape != scores.shape
            or pair_ids.numel() == 0
            or not bool(torch.isfinite(scores).all())
        ):
            raise ValueError(f"Invalid Teacher score part: {path}")
        first = int(pair_ids[0])
        last = int(pair_ids[-1])
        if first <= last_pair_id or not bool(torch.all(pair_ids[1:] > pair_ids[:-1])):
            raise ValueError(f"Teacher score parts are not strictly ordered: {path}")
        parts.append(
            {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(path),
                "count": int(pair_ids.numel()),
                "first_pair_id": first,
                "last_pair_id": last,
            }
        )
        last_pair_id = last
    return parts, last_pair_id


def _feature(
    object_id: str,
    object_type: str,
    store: FeatureStore,
    compressed: dict[str, torch.Tensor],
    batch_cache: dict[str, ObjectFeatures],
    device: torch.device,
    hidden_dtype: torch.dtype,
) -> ObjectFeatures:
    if object_id in batch_cache:
        return batch_cache[object_id]
    if object_id in compressed:
        value = ObjectFeatures(
            object_id=object_id,
            object_type=object_type,
            embedding=torch.empty(0, device=device),
        )
    else:
        value = store.get(object_id, include_hidden=True).for_scoring(
            device,
            include_hidden=True,
            hidden_dtype=hidden_dtype,
        )
    batch_cache[object_id] = value
    return value


@torch.inference_mode()
def _score_batch(
    records: list[dict[str, Any]],
    teacher: torch.nn.Module,
    store: FeatureStore,
    compressed: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    hidden_dtype = next(teacher.parameters()).dtype
    batch_cache: dict[str, ObjectFeatures] = {}
    sources = [
        _feature(
            str(row["source_id"]),
            str(row["source_type"]),
            store,
            compressed,
            batch_cache,
            device,
            hidden_dtype,
        )
        for row in records
    ]
    destinations = [
        _feature(
            str(row["destination_id"]),
            str(row["destination_type"]),
            store,
            compressed,
            batch_cache,
            device,
            hidden_dtype,
        )
        for row in records
    ]
    scores = teacher.score_pairs(
        sources,
        destinations,
        compression_cache=compressed,
    )
    return teacher.transform_pair_scores(
        scores,
        [str(row["source_type"]) for row in records],
        [str(row["destination_type"]) for row in records],
        "raw_logit",
    ).detach().cpu()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    started = time.monotonic()
    candidate_dir = args.output_root / "taskC_training/candidates_seed13_steps356"
    pair_path = candidate_dir / "teacher_pairs.jsonl.gz"
    candidate_manifest_path = candidate_dir / "manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("pair_manifest_sha256") != checkpoint_fingerprint(pair_path):
        raise ValueError("Teacher pair manifest fingerprint mismatch")

    output_dir = (
        args.output_root
        / "taskC_training/teacher_pair_scores"
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
        args.root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
    )
    extra_features = args.output_root / "taskC_training/teacher_extra"
    store = FeatureStore.from_path(
        base_features,
        cache_size=args.feature_cache_size,
        teacher_paths=[extra_features],
    )
    device = torch.device(args.device)
    teacher = load_teacher(teacher_checkpoint, device).eval()
    teacher.set_compute_dtype(None)
    compressed: dict[str, torch.Tensor] = {}

    parts, last_pair_id = _completed_parts(output_dir)
    next_part = len(parts)
    part_pair_ids: list[int] = []
    part_scores: list[torch.Tensor] = []
    batch: list[dict[str, Any]] = []
    scored_this_invocation = 0
    selected_seen = sum(int(part["count"]) for part in parts)

    def flush_batch() -> None:
        nonlocal scored_this_invocation
        if not batch:
            return
        scores = _score_batch(batch, teacher, store, compressed, device)
        if scores.shape != (len(batch),) or not bool(torch.isfinite(scores).all()):
            raise ValueError("Teacher returned invalid R12 pair scores")
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
        row = {
            "path": str(path.resolve()),
            "sha256": checkpoint_fingerprint(path),
            "count": len(part_pair_ids),
            "first_pair_id": part_pair_ids[0],
            "last_pair_id": part_pair_ids[-1],
        }
        parts.append(row)
        next_part += 1
        selected_total = selected_seen + scored_this_invocation
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "selected_pairs_saved": selected_total,
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
            if pair_id % args.num_shards != args.shard_index or pair_id <= last_pair_id:
                continue
            batch.append(row)
            if len(batch) == args.batch_size:
                flush_batch()
            if len(part_pair_ids) >= args.part_size:
                flush_part()
            if args.limit is not None and scored_this_invocation >= args.limit:
                break
    flush_batch()
    flush_part()

    expected_total = math.ceil(
        (int(candidate_manifest["teacher_pair_budget"]["unique_shared_pairs"]) - args.shard_index)
        / args.num_shards
    )
    total_saved = sum(int(part["count"]) for part in parts)
    complete = args.limit is None and total_saved == expected_total
    manifest = {
        "format_version": 1,
        "status": "complete" if complete else "partial",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "expected_pairs": expected_total,
        "saved_pairs": total_saved,
        "pair_manifest": str(pair_path.resolve()),
        "pair_manifest_sha256": checkpoint_fingerprint(pair_path),
        "candidate_manifest_sha256": checkpoint_fingerprint(candidate_manifest_path),
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": checkpoint_fingerprint(teacher_checkpoint),
        "teacher_score_space": "raw_logit",
        "teacher_compute_dtype": "float32",
        "base_features": str(base_features.resolve()),
        "extra_teacher_features": str(extra_features.resolve()),
        "extra_teacher_manifest_sha256": checkpoint_fingerprint(
            extra_features / "teacher_manifest.jsonl"
        ),
        "parts": parts,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat() if complete else None,
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


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
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if min(args.num_shards, args.batch_size, args.part_size) <= 0:
        parser.error("shard, batch, and part sizes must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
