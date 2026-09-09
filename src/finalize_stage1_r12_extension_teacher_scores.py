#!/usr/bin/env python
"""Validate and merge the R12 extension Teacher score suffix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    candidate_dir = (
        args.output_root
        / "taskC_training/candidates_seed13_extension_steps357_1318"
    )
    candidate_manifest_path = candidate_dir / "manifest.json"
    candidate_manifest = json.loads(
        candidate_manifest_path.read_text(encoding="utf-8")
    )
    budget = candidate_manifest["teacher_pair_budget"]
    pair_start = int(budget["global_pair_id_start"])
    pair_end = int(budget["global_pair_id_end_exclusive"])
    suffix_count = pair_end - pair_start
    pair_path = candidate_dir / "new_teacher_pairs.jsonl.gz"
    pair_sha256 = checkpoint_fingerprint(pair_path)
    if pair_sha256 != candidate_manifest["new_pair_manifest_sha256"]:
        raise ValueError("Extension candidate and pair manifests disagree")

    score_root = args.output_root / "taskC_training/teacher_extension_pair_scores"
    shard_manifests = []
    for path in sorted(score_root.glob("shard_*_of_*/manifest.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "complete":
            raise ValueError(f"Teacher score shard is incomplete: {path}")
        if row.get("pair_manifest_sha256") != pair_sha256:
            raise ValueError(f"Teacher score shard used another pair manifest: {path}")
        shard_manifests.append(row)
    if not shard_manifests:
        raise FileNotFoundError("No complete extension Teacher score shards found")
    num_shards = int(shard_manifests[0]["num_shards"])
    if len(shard_manifests) != num_shards or {
        int(row["shard_index"]) for row in shard_manifests
    } != set(range(num_shards)):
        raise ValueError("Extension Teacher score shard set is incomplete")

    scores = torch.full((suffix_count,), float("nan"), dtype=torch.float32)
    seen = torch.zeros(suffix_count, dtype=torch.bool)
    parts = []
    teacher_sha256 = None
    total_gpu_seconds = 0.0
    for manifest in shard_manifests:
        current_teacher = str(manifest["teacher_checkpoint_sha256"])
        if teacher_sha256 is None:
            teacher_sha256 = current_teacher
        elif teacher_sha256 != current_teacher:
            raise ValueError("Extension shards used different Teachers")
        total_gpu_seconds += float(manifest["elapsed_seconds_this_invocation"])
        for row in manifest["parts"]:
            path = Path(row["path"])
            if checkpoint_fingerprint(path) != row["sha256"]:
                raise ValueError(f"Teacher score part fingerprint mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            pair_ids = payload["pair_ids"].long()
            values = payload["scores"].float()
            offsets = pair_ids - pair_start
            if (
                pair_ids.shape != values.shape
                or pair_ids.ndim != 1
                or bool(torch.any(offsets < 0))
                or bool(torch.any(offsets >= suffix_count))
                or bool(seen[offsets].any())
                or not bool(torch.isfinite(values).all())
            ):
                raise ValueError(f"Teacher score part failed validation: {path}")
            scores[offsets] = values
            seen[offsets] = True
            parts.append(
                {
                    "path": str(path.resolve()),
                    "sha256": row["sha256"],
                    "count": int(pair_ids.numel()),
                }
            )
    if int(seen.sum()) != suffix_count or not bool(torch.isfinite(scores).all()):
        raise ValueError("Extension Teacher scores contain gaps or non-finite values")
    output_path = score_root / "scores.pt"
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": 1,
            "global_pair_id_start": pair_start,
            "global_pair_id_end_exclusive": pair_end,
            "pair_manifest_sha256": pair_sha256,
            "teacher_checkpoint_sha256": teacher_sha256,
            "teacher_score_space": "raw_logit",
            "scores": scores,
        },
        temporary,
    )
    temporary.replace(output_path)
    payload = {
        "format_version": 1,
        "status": "complete",
        "global_pair_id_start": pair_start,
        "global_pair_id_end_exclusive": pair_end,
        "pair_count": suffix_count,
        "coverage_count": int(seen.sum()),
        "duplicate_pair_ids": 0,
        "non_finite_scores": int((~torch.isfinite(scores)).sum()),
        "pair_manifest": str(pair_path.resolve()),
        "pair_manifest_sha256": pair_sha256,
        "candidate_manifest_sha256": checkpoint_fingerprint(
            candidate_manifest_path
        ),
        "teacher_checkpoint_sha256": teacher_sha256,
        "teacher_score_space": "raw_logit",
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "scores": str(output_path.resolve()),
        "scores_sha256": checkpoint_fingerprint(output_path),
        "screen_scores_sha256": candidate_manifest["screen_teacher_scores_sha256"],
        "source_parts": parts,
        "total_gpu_seconds_across_shards": total_gpu_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(score_root / "manifest.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())
