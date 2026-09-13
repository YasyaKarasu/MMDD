"""R22 full-lake extension: build indexes, ANN retrieval, exact-direct eval.

Runs E1/E2/E3 × seed13/29 in parallel across available GPUs.
Reuses R21 Qwen-Raw index corpus table_ids (same features + corpus).
"""
from __future__ import annotations

import argparse, json, statistics, subprocess, sys, time
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
    _metric_rows,
    FINAL_STEP,
)
from run_stage1_r22 import paths as r22_paths, out as r22_out, ARMS, SEEDS

ROOT = Path(__file__).resolve().parents[1]
FINAL_STEP_R22 = 1318


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_checkpoint(root: Path, arm: str, seed: int, step: int, device: torch.device):
    path = r22_out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_student(path, device).eval(), path


def build_index(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for index building")
    device = torch.device(device_name)
    ps = r21_paths(root)
    store = FeatureStore.from_path(ps["features"], cache_size=100000)
    ids_by_type = load_corpus_ids(ps["corpus"], store)
    corpus_sha = checkpoint_fingerprint(ps["corpus"])
    model, checkpoint = _load_checkpoint(root, arm, seed, step, device)
    directory = r22_out(root) / "indexes" / arm / f"seed{seed}" / f"step_{step:06d}"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        build_indices(model, store, ids_by_type, directory, device=device,
                      checkpoint_sha256=checkpoint_fingerprint(checkpoint),
                      corpus_sha256=corpus_sha, batch_size=4096)
    return json.loads(manifest.read_text(encoding="utf-8"))


@torch.inference_mode()
def evaluate_full_lake(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device(device_name)
    ps = r21_paths(root)
    store = FeatureStore.from_path(ps["features"], cache_size=120000)
    pool_rows = list(read_rows(ps["candidate_pools"]))
    query_ids = [str(row["query_id"]) for row in pool_rows]
    corpus_sha = checkpoint_fingerprint(ps["corpus"])
    model, checkpoint = _load_checkpoint(root, arm, seed, step, device)
    index_dir = r22_out(root) / "indexes" / arm / f"seed{seed}" / f"step_{step:06d}"
    if not (index_dir / "manifest.json").is_file():
        build_index(root, arm, seed, step, device_name)
    indices = StudentANNIndices(model, store, index_dir, device=device,
                                checkpoint_sha256=checkpoint_fingerprint(checkpoint),
                                corpus_sha256=corpus_sha)
    destination = r22_out(root) / "full_lake" / arm / f"seed{seed}_step{step:06d}"
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
        direct_exact = sorted(direct_ids, key=lambda t: (-score_map[t], t))
        ranking_rows.append({
            "query_id": str(pool["query_id"]), "query_kind": pool["query_kind"],
            "positive_target_ids": sorted(positive),
            "direct_ann": direct_ids, "evidence_ann": evidence_ids, "U": union,
            "direct_exact_ranking": direct_exact, "u_exact_ranking": ranking,
            "u_size": len(union), "direct_size": len(direct_ids),
            "direct_raw_recall@100": len(positive & set(direct_ids)) / len(positive),
            "u_raw_recall": len(positive & set(union)) / len(positive),
            "u_recall@10": len(positive & set(ranking[:10])) / len(positive),
            "u_recall@20": len(positive & set(ranking[:20])) / len(positive),
            "u_recall@50": len(positive & set(ranking[:50])) / len(positive),
            "direct_exact_recall@10": len(positive & set(direct_exact[:10])) / len(positive),
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
        "direct_exact_r10": mean("direct_exact_recall@10"),
        "u_size_mean": statistics.fmean(r["u_size"] for r in ranking_rows),
        "by_query_kind": {
            kind: {n: statistics.fmean(float(r[n]) for r in ranking_rows if r["query_kind"] == kind)
                   for n in ("direct_raw_recall@100", "u_raw_recall", "u_recall@10",
                              "u_recall@20", "u_recall@50")}
            for kind in ("implicit", "explicit")
        },
        "rankings": str((destination / "rankings.jsonl.gz").resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "completed_at_utc": now(),
    }
    write_json(metrics_path, metrics)
    return metrics


@torch.inference_mode()
def evaluate_exact_direct(root: Path, arm: str, seed: int, step: int, device_name: str,
                          query_batch_size: int = 32) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device(device_name)
    ps = r21_paths(root)
    source_dir = r22_out(root) / "full_lake" / arm / f"seed{seed}_step{step:06d}"
    source_rankings = source_dir / "rankings.jsonl.gz"
    if not source_rankings.is_file():
        raise FileNotFoundError(source_rankings)
    existing = list(read_rows(source_rankings))
    store = FeatureStore.from_path(ps["features"], cache_size=50000)
    # Reuse the table_ids built by R21's Qwen-Raw index (same corpus)
    r21_index_dir = r21_out(root) / "indexes" / "Qwen-Raw"
    table_ids = json.loads((r21_index_dir / "table_ids.json").read_text(encoding="utf-8"))
    target_embeddings = torch.stack(
        [store.embedding_features(str(v)).embedding for v in table_ids]
    ).to(device=device, dtype=torch.float32)
    model, checkpoint = _load_checkpoint(root, arm, seed, step, device)
    target_vectors = model.project(target_embeddings, "table", role="target")
    output_rows = []
    for start in range(0, len(existing), query_batch_size):
        batch = existing[start:start + query_batch_size]
        query_embeddings = torch.stack(
            [store.embedding_features(str(row["query_id"])).embedding for row in batch]
        ).to(device=device, dtype=torch.float32)
        query_vectors = model.project(query_embeddings, "table", role="query") @ \
            model.relations[model.relation_key("table", "table")]
        scores = query_vectors @ target_vectors.T
        values, indices = scores.topk(k=100, dim=1)
        for row, row_indices, row_values in zip(batch, indices.cpu(), values.cpu()):
            exact_ids = [str(table_ids[int(idx)]) for idx in row_indices]
            ann_ids = [str(v) for v in row["direct_ann"][:100]]
            positives = set(map(str, row["positive_target_ids"]))
            rec = {
                "query_id": str(row["query_id"]), "query_kind": str(row["query_kind"]),
                "positive_target_ids": sorted(positives),
                "exact_ids": exact_ids,
                "exact_scores": [float(v) for v in row_values],
                "ann_ids": ann_ids,
                "neighbor_recall@100": len(set(exact_ids) & set(ann_ids)) / 100,
            }
            for k in (10, 20, 50, 100):
                rec[f"exact_recall@{k}"] = len(positives & set(exact_ids[:k])) / len(positives)
                rec[f"ann_recall@{k}"] = len(positives & set(ann_ids[:k])) / len(positives)
            output_rows.append(rec)
    exact_path = source_dir / "direct_exact_full_lake.jsonl.gz"
    write_rows(exact_path, output_rows)
    def mean(k: str) -> float:
        return statistics.fmean(row[k] for row in output_rows)
    result = {
        "format_version": 1, "status": "complete", "retriever": arm, "seed": seed, "step": step,
        "corpus_tables": len(table_ids), "queries": len(output_rows),
        "exact": {f"recall@{k}": mean(f"exact_recall@{k}") for k in (10, 20, 50, 100)},
        "ann": {f"recall@{k}": mean(f"ann_recall@{k}") for k in (10, 20, 50, 100)},
        "ann_neighbor_recall@100": mean("neighbor_recall@100"),
        "by_query_kind": {
            kind: {
                f"exact_recall@{k}": statistics.fmean(
                    row[f"exact_recall@{k}"] for row in output_rows if row["query_kind"] == kind)
                for k in (10, 20, 50, 100)
            } for kind in ("implicit", "explicit")
        },
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "rankings": str(exact_path.resolve()),
        "completed_at_utc": now(),
    }
    write_json(source_dir / "EXACT_ANN.json", result)
    # Merge into the full-lake metrics
    metrics_path = source_dir / "metrics.json"
    if metrics_path.is_file():
        m = json.loads(metrics_path.read_text())
        m["direct_exact_full_lake"] = result
        write_json(metrics_path, m)
    return result


def run_one(arm: str, seed: int, device_name: str) -> None:
    root = ROOT
    step = FINAL_STEP_R22
    print(json.dumps({"status": "starting", "arm": arm, "seed": seed, "device": device_name,
                      "time": now()}), flush=True)
    build_index(root, arm, seed, step, device_name)
    print(json.dumps({"status": "index_done", "arm": arm, "seed": seed, "time": now()}), flush=True)
    evaluate_full_lake(root, arm, seed, step, device_name)
    print(json.dumps({"status": "fulllake_done", "arm": arm, "seed": seed, "time": now()}), flush=True)
    evaluate_exact_direct(root, arm, seed, step, device_name)
    print(json.dumps({"status": "exact_done", "arm": arm, "seed": seed, "time": now()}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--step", type=int, default=FINAL_STEP_R22)
    args = ap.parse_args()
    if args.arm == "all":
        # Caller is responsible for GPU assignment; just run the given arm/seed
        raise SystemExit("Use --arm <E1|E2|E3> --seed <13|29> --device <cuda:N> for single jobs")
    seed = args.seed if args.seed is not None else 13
    run_one(args.arm, seed, args.device)


if __name__ == "__main__":
    main()
