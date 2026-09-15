"""Held-out natural-pool quality audit for the frozen R22 Teachers."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.features import FeatureStore
from run_stage1_r19 import _score_id_pairs, load_r19_checkpoint
from run_stage1_r21 import paths as r21_paths
from run_stage1_r23 import SEEDS, out, read_rows, r23_paths, teacher_feature_paths

ROOT = Path(__file__).resolve().parents[1]
TEACHERS = ("T0", "T1-A", "T1-B")


def run(root: Path, teacher_name: str, seed: int, device_name: str) -> dict[str, Any]:
    if teacher_name not in TEACHERS or seed not in SEEDS:
        raise ValueError("teacher must be T0/T1-A/T1-B and seed 13/29")
    checkpoint = root / "work/stage1_optimization_r22_20260911" / "fresh_lineage" / teacher_name / f"seed{seed}" / "checkpoints/step_010536.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    destination = out(root) / "teacher_quality" / teacher_name / f"seed{seed}"
    metrics_path = destination / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    device = torch.device(device_name)
    _arm, saved_seed, _step, teacher, _payload = load_r19_checkpoint(checkpoint, device)
    if saved_seed != seed:
        raise RuntimeError(f"Teacher seed mismatch: expected {seed}, got {saved_seed}")
    teacher.eval()
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=30000, teacher_paths=teacher_feature_paths(root))
    cache = teacher.new_compression_cache()
    rows = []
    missing = 0; total = 0
    for pool in read_rows(r23_paths(root)["candidate_pools"]):
        candidates = list(dict.fromkeys(map(str, pool["natural_candidate_ids"])))
        available = []
        for candidate in candidates:
            total += 1
            feature = store.get(candidate, include_hidden=True)
            if feature.hidden_states is None:
                missing += 1
            else:
                available.append(candidate)
        with torch.inference_mode():
            values = _score_id_pairs(
                teacher,
                [(str(pool["query_id"]), candidate) for candidate in available],
                store,
                device,
                batch_size=128,
                cache=cache,
            ) if available else []
        ranking = [candidate for candidate, _value in sorted(zip(available, values), key=lambda pair: (-pair[1], pair[0]))]
        positives = set(map(str, pool["positive_target_ids"]))
        rows.append({"query_id": str(pool["query_id"]), "query_kind": pool["query_kind"], "positive_target_ids": sorted(positives), "candidate_count": len(candidates), "scored_count": len(available), "ranking": ranking,
                     "raw_recall": len(positives & set(candidates)) / len(positives) if positives else 0.0,
                     **{f"recall@{k}": len(positives & set(ranking[:k])) / len(positives) if positives else 0.0 for k in (10, 20, 50)}})
    aggregate = lambda values: {key: statistics.fmean(float(row[key]) for row in values) for key in ("raw_recall", "recall@10", "recall@20", "recall@50")}
    destination.mkdir(parents=True, exist_ok=True)
    from run_stage1_r23 import write_rows
    ranking_path = destination / "rankings.jsonl.gz"; write_rows(ranking_path, rows)
    result = {"format_version": 1, "status": "complete" if missing == 0 else "partial", "teacher": teacher_name, "seed": seed, "checkpoint_sha256": checkpoint_fingerprint(checkpoint), "queries": len(rows), "natural_pool": aggregate(rows), "by_query_kind": {kind: aggregate([row for row in rows if row["query_kind"] == kind]) for kind in ("implicit", "explicit")}, "teacher_candidate_coverage": 1.0 - missing / total if total else 0.0, "missing_candidate_count": missing, "total_candidate_count": total, "rankings": str(ranking_path.resolve()), "note": "Missing hidden-state candidates are ranked after scored candidates; metrics are lower-bound partial." if missing else None}
    write_json(metrics_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=ROOT); parser.add_argument("--teacher", required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); print(json.dumps(run(args.root.resolve(), args.teacher, args.seed, args.device), indent=2))
