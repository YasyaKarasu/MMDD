"""R22 F0 evaluation: fixed-pool + full-lake for fresh-lineage arms.

Handles F0-SUP and F0-D2 checkpoints stored under fresh_lineage/.
"""
from __future__ import annotations

import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    build_indices,
    load_corpus_ids,
    retrieve_zero_one_hop_detailed_many,
)
from run_stage1_r21 import (
    paths as r21_paths,
    out as r21_out,
    read_rows,
    write_rows,
    _score_target_ids,
)
from run_stage1_r22 import paths as r22_paths, out as r22_out, SEEDS

ROOT = Path(__file__).resolve().parents[1]
FINAL_STEP = 1318

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _f0_checkpoint(root: Path, arm: str, seed: int, step: int) -> Path:
    return r22_out(root) / "fresh_lineage" / arm / f"seed{seed}" / "edge" / "checkpoints" / f"step_{step:06d}.pt"

def _load_f0(root: Path, arm: str, seed: int, step: int, device: torch.device):
    path = _f0_checkpoint(root, arm, seed, step)
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_student(path, device).eval(), path

@torch.inference_mode()
def evaluate_fixed_pool(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    ck_path = _f0_checkpoint(root, arm, seed, step)
    model = load_student(ck_path, device).eval()
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=40000)
    records = []
    for row in read_rows(r22_paths(root)["candidate_pools"]):
        q = str(row["query_id"])
        pos = set(map(str, row["positive_target_ids"]))
        pools = {
            "natural": list(map(str, row["natural_candidate_ids"])),
            "direct100": list(map(str, row["ann_direct100_ids"])),
            "matched": list(map(str, row["matched_direct_candidate_ids"])),
        }
        ids = list(dict.fromkeys(x for v in pools.values() for x in v))
        sm = {}
        for st in range(0, len(ids), 2048):
            part = ids[st:st+2048]
            vals = model.score_pairs_in_space(
                [store.embedding_features(q)] * len(part),
                [store.embedding_features(x) for x in part],
                "raw_logit"
            ).detach().cpu().tolist()
            sm.update(zip(part, map(float, vals)))
        rec = {"query_id": q, "query_kind": row.get("query_kind"), "pools": {}}
        for name, cand in pools.items():
            rank = sorted(dict.fromkeys(cand), key=lambda x: (-sm[x], x))
            rec["pools"][name] = {
                "recall@10": len(pos & set(rank[:10])) / len(pos) if pos else 0.0,
                "recall@20": len(pos & set(rank[:20])) / len(pos) if pos else 0.0,
                "recall@50": len(pos & set(rank[:50])) / len(pos) if pos else 0.0,
                "raw_recall": len(pos & set(cand)) / len(pos) if pos else 0.0,
            }
        records.append(rec)
    mean = lambda n, k: statistics.fmean(r["pools"][n][k] for r in records)
    dest = r22_out(root) / "fresh_lineage" / arm / f"seed{seed}" / "evaluations" / f"step_{step:06d}"
    dest.mkdir(parents=True, exist_ok=True)
    write_rows(dest / "fixed_pool_rankings.jsonl.gz", records)
    result = {
        "format_version": 1, "status": "complete", "arm": arm, "seed": seed, "step": step,
        "queries": len(records),
        "natural": {k: mean("natural", k) for k in ("raw_recall", "recall@10", "recall@20", "recall@50")},
        "direct100": {k: mean("direct100", k) for k in ("raw_recall", "recall@10")},
        "matched": {k: mean("matched", k) for k in ("raw_recall", "recall@10")},
        "checkpoint_sha256": checkpoint_fingerprint(ck_path),
        "completed_at_utc": now(),
    }
    write_json(dest / "fixed_pool_metrics.json", result)
    return result

def build_f0_index(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    ps = r21_paths(root)
    store = FeatureStore.from_path(ps["features"], cache_size=100000)
    ids_by_type = load_corpus_ids(ps["corpus"], store)
    corpus_sha = checkpoint_fingerprint(ps["corpus"])
    model, checkpoint = _load_f0(root, arm, seed, step, device)
    directory = r22_out(root) / "fresh_lineage" / arm / f"seed{seed}" / "indexes" / f"step_{step:06d}"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        build_indices(model, store, ids_by_type, directory, device=device,
                      checkpoint_sha256=checkpoint_fingerprint(checkpoint),
                      corpus_sha256=corpus_sha, batch_size=4096)
    return json.loads(manifest.read_text(encoding="utf-8"))

@torch.inference_mode()
def evaluate_full_lake(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    ps = r21_paths(root)
    store = FeatureStore.from_path(ps["features"], cache_size=120000)
    pool_rows = list(read_rows(r22_paths(root)["candidate_pools"]))
    query_ids = [str(row["query_id"]) for row in pool_rows]
    corpus_sha = checkpoint_fingerprint(ps["corpus"])
    model, checkpoint = _load_f0(root, arm, seed, step, device)
    index_dir = r22_out(root) / "fresh_lineage" / arm / f"seed{seed}" / "indexes" / f"step_{step:06d}"
    if not (index_dir / "manifest.json").is_file():
        build_f0_index(root, arm, seed, step, device_name)
    indices = StudentANNIndices(model, store, index_dir, device=device,
                                checkpoint_sha256=checkpoint_fingerprint(checkpoint),
                                corpus_sha256=corpus_sha)
    destination = r22_out(root) / "fresh_lineage" / arm / f"seed{seed}" / "full_lake" / f"step_{step:06d}"
    metrics_path = destination / "metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    detailed = retrieve_zero_one_hop_detailed_many(
        query_ids, indices, k=100, direct_k=100, evidence_k=20,
        targets_per_evidence=20, query_batch_size=16)
    ranking_rows = []
    for pool, retrieved in zip(pool_rows, detailed):
        positive = set(map(str, pool["positive_target_ids"]))
        direct_ids = [str(item["target_id"]) for item in retrieved["direct"]]
        evidence_ids = [str(item["target_id"]) for item in retrieved["evidence"]]
        union = list(dict.fromkeys([*direct_ids, *evidence_ids]))
        score_map = _score_target_ids(model, store, str(pool["query_id"]), union, device)
        ranking = sorted(union, key=lambda t: (-score_map[t], t))
        ranking_rows.append({
            "query_id": str(pool["query_id"]), "query_kind": pool["query_kind"],
            "positive_target_ids": sorted(positive),
            "direct_ann": direct_ids, "evidence_ann": evidence_ids, "U": union,
            "u_exact_ranking": ranking,
            "u_size": len(union), "direct_size": len(direct_ids),
            "direct_raw_recall@100": len(positive & set(direct_ids)) / len(positive),
            "u_raw_recall": len(positive & set(union)) / len(positive),
            "u_recall@10": len(positive & set(ranking[:10])) / len(positive),
            "u_recall@20": len(positive & set(ranking[:20])) / len(positive),
            "u_recall@50": len(positive & set(ranking[:50])) / len(positive),
        })
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "rankings.jsonl.gz", ranking_rows)
    def mean(name: str) -> float:
        return statistics.fmean(float(r[name]) for r in ranking_rows)
    metrics = {
        "format_version": 1, "status": "complete", "retriever": arm,
        "seed": seed, "step": step, "queries": len(ranking_rows),
        "direct_raw@100": mean("direct_raw_recall@100"),
        "u_raw_recall": mean("u_raw_recall"), "u_r10": mean("u_recall@10"),
        "u_r20": mean("u_recall@20"), "u_cr50": mean("u_recall@50"),
        "u_size_mean": statistics.fmean(r["u_size"] for r in ranking_rows),
        "checkpoint_sha256": checkpoint_fingerprint(_f0_checkpoint(root, arm, seed, step)),
        "completed_at_utc": now(),
    }
    write_json(metrics_path, metrics)
    return metrics

def run_all(arm: str, seed: int, device_name: str, step: int = FINAL_STEP) -> None:
    root = ROOT
    print(json.dumps({"status": "starting", "arm": arm, "seed": seed, "device": device_name, "time": now()}), flush=True)
    evaluate_fixed_pool(root, arm, seed, step, device_name)
    print(json.dumps({"status": "fixed_done", "arm": arm, "seed": seed, "time": now()}), flush=True)
    build_f0_index(root, arm, seed, step, device_name)
    print(json.dumps({"status": "index_done", "arm": arm, "seed": seed, "time": now()}), flush=True)
    evaluate_full_lake(root, arm, seed, step, device_name)
    print(json.dumps({"status": "fulllake_done", "arm": arm, "seed": seed, "time": now()}), flush=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="Fresh Student arm name")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--step", type=int, default=FINAL_STEP,
                    help="Checkpoint step to evaluate (default: 1318)")
    args = ap.parse_args()
    run_all(args.arm, args.seed, args.device, args.step)

if __name__ == "__main__":
    main()
