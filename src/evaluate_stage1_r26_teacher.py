"""Score independently retrieved own U/M, Equal BT100, and D100 with fixed T0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.r26_metrics import population_metrics
from mmdd_stage1.r26_teacher import TeacherPairCache, rerank_pools
from mmdd_stage1.teacher_rerank import _teacher_scores
from prepare_stage1_r26 import ROOT, OUT, file_record, stable_sha
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r21 import paths, read_rows, write_rows
from run_stage1_r25 import _json, _r25_teacher_feature_paths, sha256


def metrics(rows: list[dict]) -> dict:
    result = {}
    for kind in ("overall", "implicit", "explicit"):
        subset = [r for r in rows if kind == "overall" or r["query_kind"] == kind]
        qrels = {r["query_id"]: r["positive_target_ids"] for r in subset}
        result[kind] = {name: population_metrics({r["query_id"]: r["rankings"][name] for r in subset}, qrels, (10, 20, 50))
                        for name in rows[0]["rankings"]}
        result[kind]["queries"] = len(subset)
    return result


@torch.inference_mode()
def run(generators: list[str], device_name: str, benchmark_queries: int, *, cache_name: str = "T0_pairs.sqlite") -> dict:
    torch.set_num_threads(4)
    device = torch.device(device_name)
    checkpoint = ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"
    started = time.monotonic()
    _, _, _, teacher, _ = load_r19_checkpoint(checkpoint, device)
    teacher.eval()
    store = FeatureStore.from_path(paths(ROOT)["features"], cache_size=30000, cache_bytes=8*1024**3,
                                   teacher_paths=_r25_teacher_feature_paths(ROOT))
    directory = OUT / "teacher"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"teacher": file_record(checkpoint), "score_semantics": "R19GlobalTeacher.score_pairs raw QT scalar; no path input",
                "code": {name: sha256(ROOT / "src" / name) for name in (
                    "mmdd_stage1/r26_teacher.py", "mmdd_stage1/teacher_rerank.py", "run_stage1_r19.py")}}
    namespace = stable_sha(identity)
    _json(directory / "CACHE_IDENTITY.json", {**identity, "namespace": namespace,
          "pair_key": ["namespace", "query_id", "target_id", "actual_query_feature_sha", "actual_target_feature_sha"]})
    cache = TeacherPairCache(directory / cache_name, namespace, teacher, store, device)
    initialization_seconds = time.monotonic()-started
    completed = []
    for generator in generators:
        source = OUT / "rankings" / generator
        receipt = source / "RETRIEVAL_RECEIPT.json"
        if not receipt.exists():
            print(json.dumps({"event": "pending_own_retrieval", "generator": generator}), flush=True)
            continue
        destination = directory / generator
        signature = {"generator": generator, "own_receipt": file_record(receipt),
                     "own_rankings": file_record(source / "rankings.jsonl.gz"), "namespace": namespace,
                     "runner_sha": sha256(Path(__file__)), "benchmark_queries": benchmark_queries, "device": device_name}
        finished = destination / "TEACHER_RECEIPT.json"
        if finished.exists():
            previous = json.loads(finished.read_text())
            if previous["signature"] != signature:
                raise ValueError(f"Teacher identity changed: {generator}")
            if previous["rankings"]["sha256"] != sha256(destination / "rankings.jsonl.gz"):
                raise ValueError("Teacher rankings changed")
            completed.append(generator)
            continue
        rows, costs, benchmarks = [], [], []
        started = time.monotonic()
        for index, row in enumerate(read_rows(source / "rankings.jsonl.gz")):
            targets = sorted(set(row["U"]) | set(row["M_exact"]))
            scores, cost = cache.score(row["query_id"], targets)
            costs.append(cost)
            rankings = rerank_pools(row, scores)
            truth = set(row["positive_target_ids"])
            rows.append({key: row[key] for key in ("query_id", "query_kind", "source_table_id", "positive_target_ids", "candidate_pool_id")})
            rows[-1].update({"generator_id": generator, "teacher_namespace": namespace, "teacher_scores": scores,
                "rankings": rankings, "cost": cost, "U_raw_recall": len(truth & set(row["U"]))/len(truth),
                "BT100_pretruncate_recall_loss": len(truth & (set(row["U"])-set(rankings["BT100_NO_T0"])))/len(truth)})
            # Actual pair computations, bypassing all query-result/score caches.
            # Cold means cold Teacher compression; frozen features are already local.
            if index < benchmark_queries:
                for pool_name in ("BT100_NO_T0", "D100_NO_T0"):
                    compression = teacher.new_compression_cache()
                    for mode in ("cold_compression", "warm_compression"):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        timer = time.monotonic()
                        values = _teacher_scores(teacher, row["query_id"], rankings[pool_name], store, device,
                                                 batch_size=64, compression_cache=compression)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        benchmarks.append({"query_id": row["query_id"], "pool": pool_name, "mode": mode,
                            "pairs": len(values), "seconds": time.monotonic()-timer,
                            "max_score_difference_from_cached": max((abs(v-scores[t]) for t,v in zip(rankings[pool_name],values)), default=0)})
            if (index+1) % 100 == 0:
                print(json.dumps({"event": "teacher_queries", "generator": generator, "queries": index+1,
                                  "new_pairs": sum(c["new_pairs"] for c in costs)}), flush=True)
        destination.mkdir(parents=True, exist_ok=True)
        write_rows(destination / "rankings.jsonl.gz", rows)
        write_rows(destination / "latency_benchmarks.jsonl", benchmarks)
        summary = metrics(rows)
        _json(destination / "metrics.json", summary)
        latency = {}
        for pool_name in ("BT100_NO_T0", "D100_NO_T0"):
            for mode in ("cold_compression", "warm_compression"):
                values = [r["seconds"] for r in benchmarks if r["pool"] == pool_name and r["mode"] == mode]
                if values:
                    latency[pool_name+"/"+mode] = {"n": len(values), "p50": float(np.quantile(values,.5)), "p95": float(np.quantile(values,.95))}
        _json(finished, {"signature": signature, "execution_status": "ran", "scientific_validity": "valid",
            "rankings": file_record(destination / "rankings.jsonl.gz"), "metrics": summary,
            "cost": {"initialization_seconds": initialization_seconds, "elapsed_seconds": time.monotonic()-started,
                **{key: sum(c[key] for c in costs) for key in costs[0]}, "online_T0_latency_seconds": latency,
                "latency_scope": "T0 only, local frozen features; cold/warm compression, no pair-score cache; excludes Student/feature extraction",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}})
        completed.append(generator)
        print(json.dumps({"event": "teacher_complete", "generator": generator}), flush=True)
    cache.db.close()
    return {"completed": completed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark-queries", type=int, default=32)
    args = parser.parse_args()
    generators = args.generator or [r["generator_id"] for r in json.loads((OUT / "MODEL_INVENTORY.json").read_text())]
    print(json.dumps(run(generators, args.device, args.benchmark_queries)))
