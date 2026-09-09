#!/usr/bin/env python
"""Validate and merge the complete R12 Teacher pair-score shards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    candidate_dir = args.output_root / "taskC_training/candidates_seed13_steps356"
    candidate_manifest_path = candidate_dir / "manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    pair_count = int(candidate_manifest["teacher_pair_budget"]["unique_shared_pairs"])
    pair_path = candidate_dir / "teacher_pairs.jsonl.gz"
    pair_sha256 = checkpoint_fingerprint(pair_path)
    if candidate_manifest.get("pair_manifest_sha256") != pair_sha256:
        raise ValueError("Candidate and Teacher pair manifests disagree")

    score_root = args.output_root / "taskC_training/teacher_pair_scores"
    shard_manifests = []
    for path in sorted(score_root.glob("shard_*_of_*/manifest.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "complete":
            raise ValueError(f"Teacher score shard is incomplete: {path}")
        if row.get("pair_manifest_sha256") != pair_sha256:
            raise ValueError(f"Teacher score shard used another pair manifest: {path}")
        shard_manifests.append((path, row))
    if not shard_manifests:
        raise FileNotFoundError("No complete Teacher score shards found")
    num_shards = int(shard_manifests[0][1]["num_shards"])
    if len(shard_manifests) != num_shards or {
        int(row["shard_index"]) for _path, row in shard_manifests
    } != set(range(num_shards)):
        raise ValueError("Teacher score shard set is incomplete")

    scores = torch.full((pair_count,), float("nan"), dtype=torch.float32)
    seen = torch.zeros(pair_count, dtype=torch.bool)
    parts = []
    teacher_sha256 = None
    total_gpu_seconds = 0.0
    for manifest_path, manifest in shard_manifests:
        current_teacher = str(manifest["teacher_checkpoint_sha256"])
        if teacher_sha256 is None:
            teacher_sha256 = current_teacher
        elif teacher_sha256 != current_teacher:
            raise ValueError("Teacher score shards used different checkpoints")
        total_gpu_seconds += float(manifest["elapsed_seconds_this_invocation"])
        for row in manifest["parts"]:
            path = Path(row["path"])
            if checkpoint_fingerprint(path) != row["sha256"]:
                raise ValueError(f"Teacher score part fingerprint mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            pair_ids = payload["pair_ids"].long()
            values = payload["scores"].float()
            if (
                pair_ids.shape != values.shape
                or pair_ids.ndim != 1
                or bool(torch.any(pair_ids < 0))
                or bool(torch.any(pair_ids >= pair_count))
                or bool(seen[pair_ids].any())
                or not bool(torch.isfinite(values).all())
            ):
                raise ValueError(f"Teacher score part failed coverage validation: {path}")
            scores[pair_ids] = values
            seen[pair_ids] = True
            parts.append(
                {
                    "path": str(path.resolve()),
                    "sha256": row["sha256"],
                    "count": int(pair_ids.numel()),
                }
            )
    if int(seen.sum()) != pair_count or not bool(torch.isfinite(scores).all()):
        raise ValueError("Teacher pair scores contain gaps or non-finite values")

    output_path = score_root / "scores.pt"
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": 1,
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
        "pair_count": pair_count,
        "coverage_count": int(seen.sum()),
        "duplicate_pair_ids": 0,
        "non_finite_scores": int((~torch.isfinite(scores)).sum()),
        "pair_manifest": str(pair_path.resolve()),
        "pair_manifest_sha256": pair_sha256,
        "candidate_manifest_sha256": checkpoint_fingerprint(candidate_manifest_path),
        "teacher_checkpoint_sha256": teacher_sha256,
        "teacher_score_space": "raw_logit",
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "scores": str(output_path.resolve()),
        "scores_sha256": checkpoint_fingerprint(output_path),
        "source_parts": parts,
        "total_gpu_seconds_across_shards": total_gpu_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(score_root / "manifest.json", payload)
    with (args.output_root / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "C1 offline Teacher pair scoring",
                    "status": "pass",
                    "command": payload["command"],
                    "output": str((score_root / "manifest.json").resolve()),
                    "pair_count": pair_count,
                    "total_gpu_seconds_across_shards": total_gpu_seconds,
                    "elapsed_seconds": payload["elapsed_seconds"],
                }
            )
            + "\n"
        )
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())
