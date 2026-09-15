"""Measure fixed T0 across frozen B13 C50/C100/C150/C200/Full-U pools."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.teacher_rerank import _teacher_scores
from replay_r26_evidence_order_abc import read_rows, read_json, write_json, record
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r25 import _r25_teacher_feature_paths

BUDGETS = (50, 100, 150, 200, None)


def other_gpu_processes(device: str) -> list[dict]:
    """Keep this small benchmark off a GPU used by another experiment."""
    uuid = subprocess.check_output(["nvidia-smi", "-i", device.split(":")[-1],
                                   "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    lines = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                     "--format=csv,noheader,nounits"], text=True).splitlines()
    result = []
    for line in lines:
        gpu, pid, memory = [v.strip() for v in line.split(",")]
        if gpu == uuid and int(pid) != os.getpid():
            result.append({"pid": int(pid), "memory_mib": memory})
    return result


@torch.inference_mode()
def run(root: Path, output: Path, device_name: str) -> None:
    torch.set_num_threads(2)
    if other_gpu_processes(device_name):
        raise RuntimeError("Benchmark GPU occupied; do not interfere with existing experiment")
    dest = output / "latency"
    dest.mkdir(parents=True, exist_ok=False)
    source = root / "work/stage1_optimization_r26_20260914"
    rp, tp = (source / name / "B13/rankings.jsonl.gz" for name in ("rankings", "teacher"))
    receipt = read_json(tp.with_name("TEACHER_RECEIPT.json"))
    teacher_path = root / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"
    population_path = source / "common/dev_queries.jsonl"
    qids = sorted([r["query_id"] for r in read_rows(population_path)],
                  key=lambda q: (hashlib.sha256(q.encode()).hexdigest(), q))[:32]
    selected = {r["query_id"]: r for r in read_rows(rp) if r["query_id"] in qids}
    reference = {r["query_id"]: r["teacher_scores"] for r in read_rows(tp) if r["query_id"] in qids}
    assert len(selected) == len(reference) == 32
    inputs = {"retrieval": record(rp), "teacher_scores": record(tp), "checkpoint": record(teacher_path),
              "population": record(population_path)}
    assert inputs["teacher_scores"]["sha256"] == receipt["rankings"]["sha256"]
    assert inputs["retrieval"]["sha256"] == receipt["signature"]["own_rankings"]["sha256"]
    write_json(dest / "PROTOCOL.json", {"inputs": inputs, "queries": qids, "budgets": [50,100,150,200,"Full-U"],
        "device": device_name, "sampling": "first32 SHA256(query_id), label blind",
        "measurement": "three repetitions per query/budget, rotated budget order; fresh compression cache then warm cache; no pair-score cache; synchronized GPU time including CPU dispatch/transfers; preloaded frozen local features",
        "exclusions": "No backbone, ANN, retention, Stage2 or model-init cost in latency; feature loading and model-init reported separately",
        "score_parity_tolerance": .001})
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    started = time.monotonic()
    _, _, _, teacher, _ = load_r19_checkpoint(teacher_path, device)
    teacher.eval()
    init_seconds = time.monotonic() - started
    store = FeatureStore.from_path(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
        cache_size=30000, cache_bytes=3*1024**3, teacher_paths=_r25_teacher_feature_paths(root))
    # Warm up kernels before any recorded query.
    _teacher_scores(teacher, qids[0], selected[qids[0]]["rankings"]["Equal"][:64], store, device, batch_size=64)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    with (dest / "measurements.jsonl").open("w") as handle:
        for qi, q in enumerate(qids):
            if other_gpu_processes(device_name):
                raise RuntimeError("Another process started on benchmark GPU; stopping to avoid contention")
            started = time.monotonic()
            for tid in [q, *selected[q]["rankings"]["Equal"]]:
                store.get(tid, include_hidden=True)
            feature_seconds = time.monotonic() - started
            for rep in range(3):
                offset = (qi + rep) % len(BUDGETS)
                order = BUDGETS[offset:] + BUDGETS[:offset]
                for budget in order:
                    targets = selected[q]["rankings"]["Equal"][:budget]
                    cache = teacher.new_compression_cache()
                    previous = None
                    for mode in ("cold_compression", "warm_compression"):
                        torch.cuda.synchronize(device)
                        started = time.monotonic()
                        values = _teacher_scores(teacher, q, targets, store, device, batch_size=64, compression_cache=cache)
                        torch.cuda.synchronize(device)
                        elapsed = time.monotonic() - started
                        error = max(abs(v - reference[q][t]) for v, t in zip(values, targets))
                        repeat_error = max(abs(a-b) for a,b in zip(previous, values)) if previous is not None else 0.0
                        if max(error, repeat_error) > .001:
                            raise ValueError("Measured T0 differs from frozen reference beyond .001")
                        item = {"query_id": q, "budget": "Full-U" if budget is None else f"C{budget}",
                                "mode": mode, "repeat": rep, "pairs": len(targets), "seconds": elapsed,
                                "feature_load_full_U_seconds": feature_seconds, "max_score_error": error,
                                "compression_repeat_error": repeat_error}
                        rows.append(item)
                        handle.write(json.dumps(item) + "\n")
                        handle.flush()
                        previous = values
                    del cache
            gc.collect()
            print(json.dumps({"latency_queries": qi+1, "total": len(qids), "device": device_name}), flush=True)
    summary = {}
    for budget in ("C50", "C100", "C150", "C200", "Full-U"):
        summary[budget] = {}
        for mode in ("cold_compression", "warm_compression"):
            values = [float(np.median([r["seconds"] for r in rows if r["query_id"] == q and r["budget"] == budget and r["mode"] == mode])) for q in qids]
            summary[budget][mode] = {"n_queries": 32, "repeats_per_query": 3,
                                    "median_seconds": float(np.median(values)), "p95_seconds": float(np.quantile(values,.95)),
                                    "mean_seconds": float(np.mean(values))}
    write_json(dest / "RESULTS.json", {"latency": summary, "teacher_init_seconds": init_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "device": torch.cuda.get_device_name(device),
        "max_score_error": max(r["max_score_error"] for r in rows), "measurements": len(rows),
        "measurements_record": record(dest / "measurements.jsonl")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    run(args.root, args.output, args.device)
