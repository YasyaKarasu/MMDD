"""Score bridge checkpoints on the same frozen B0 D/E/U candidate pools.

This deliberately bypasses ANN and evidence re-retrieval.  It isolates Student
scoring geometry from candidate-pool changes; own-pool evaluation remains in
``evaluate_stage1_bridge.py``.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from prepare_stage1_r27 import ROOT, read_json, write_json
from run_stage1_bridge import OUT, sha256


B0_RANKINGS = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation/rankings/H-C2-step000178/rankings.jsonl.gz"
FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recall(positives: set[str], ranking: list[str], k: int | None = None) -> float:
    values = ranking if k is None else ranking[:k]
    return len(positives & set(values)) / len(positives) if positives else 0.0


def evaluate(generator: str, checkpoint: Path, *, device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    model = load_student(checkpoint, device).eval()
    store = FeatureStore.from_path(FEATURES, cache_size=60_000)
    source_rows = read_rows(B0_RANKINGS)
    output_rows = []
    metric_values = {pool: {f"R{k}": [] for k in (10, 20, 50)} | {"RawRecall": []}
                     for pool in ("D", "E", "U")}
    with torch.inference_mode():
        for row in source_rows:
            pools = {
                "D": list(dict.fromkeys(row["D100_EXACT"])),
                "E": list(dict.fromkeys(row["rankings"]["E_ONLY"])),
                "U": list(dict.fromkeys(row["U"])),
            }
            query = store.embedding_features(row["query_id"]).embedding.unsqueeze(0).to(device)
            query_vector = model.relation_query(query, "table", "table")
            rankings = {}
            scores = {}
            positives = set(row["positive_target_ids"])
            for pool, target_ids in pools.items():
                target_vectors = torch.stack([store.embedding_features(target).embedding for target in target_ids]).to(device)
                values = (query_vector @ model.index_vector(target_vectors, "table").T).squeeze(0).cpu().tolist()
                order = sorted(range(len(target_ids)), key=lambda i: (-values[i], target_ids[i]))
                rankings[pool] = [target_ids[i] for i in order]
                scores[pool] = {target_ids[i]: float(values[i]) for i in range(len(target_ids))}
                for k in (10, 20, 50):
                    metric_values[pool][f"R{k}"].append(recall(positives, rankings[pool], k))
                metric_values[pool]["RawRecall"].append(recall(positives, rankings[pool]))
            output_rows.append({"query_id": row["query_id"], "query_kind": row["query_kind"],
                                "source_table_id": row["source_table_id"],
                                "positive_target_ids": row["positive_target_ids"],
                                "candidate_pool_id": row["candidate_pool_id"],
                                "generator_id": generator, "rankings": rankings, "scores": scores})
    destination = OUT / "evaluation" / "fixed_pools" / generator
    destination.mkdir(parents=True, exist_ok=True)
    rankings_path = destination / "rankings.jsonl.gz"
    with gzip.open(rankings_path, "wt", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = {pool: {metric: sum(values) / len(values) for metric, values in pool_values.items()}
               for pool, pool_values in metric_values.items()}
    receipt = {"status": "completed", "generator": generator, "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
               "reference_pools": {"source": str(B0_RANKINGS.resolve()), "sha256": sha256(B0_RANKINGS),
                                   "semantics": "B0 historical D100_EXACT, E_ONLY, and U IDs; model rescored on frozen IDs"},
               "device": device_name, "queries": len(output_rows), "metrics": metrics,
               "rankings": {"path": str(rankings_path.resolve()), "sha256": sha256(rankings_path)}}
    write_json(destination / "METRICS.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.generator, args.checkpoint, device_name=args.device), ensure_ascii=False, indent=2))
