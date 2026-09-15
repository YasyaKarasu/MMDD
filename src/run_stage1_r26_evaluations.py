"""Bounded parallel evaluation queue, deduplicating numerically identical endpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import torch

from mmdd_stage1.checkpoints import load_student
from prepare_stage1_r26 import ROOT, OUT, file_record, parameter_sha
from run_stage1_r25 import _json, sha256


def run(workers: int, device: str) -> dict:
    torch.set_num_threads(2)
    path = OUT / "MODEL_INVENTORY.json"
    inventory = json.loads(path.read_text())
    existing_ids = {r["generator_id"] for r in inventory}
    for arm in ("O-NATIVE", "O-SUP", "E-GRAPH"):
        for seed in (13, 29):
            job = OUT / "training" / arm / f"seed{seed}"
            receipt = job / "C2_COMPLETION_RECEIPT.json"
            if not receipt.exists():
                raise FileNotFoundError(f"Core C2 unfinished: {receipt}")
            for step in (0, 89, 178):
                name = f"R26-{arm}/seed{seed}/step{step}"
                if name not in existing_ids:
                    checkpoint = job / "checkpoints" / f"step_{step:06d}.pt"
                    inventory.append({"generator_id": name, "seed": seed, "step": step, "checkpoint": str(checkpoint), **file_record(checkpoint)})
    _json(path, inventory)
    canonical = {}
    aliases, tasks, missing = [], [], []
    for row in inventory:
        name = row["generator_id"]
        checkpoint = Path(row["checkpoint"]) if row["checkpoint"] else None
        if checkpoint and not checkpoint.exists():
            missing.append({"generator_id": name, "checkpoint": str(checkpoint), "execution_status": "blocked_missing_checkpoint", "scientific_validity": "unassessable"})
            continue
        fingerprint = parameter_sha(load_student(checkpoint, torch.device("cpu"))) if checkpoint else "raw"
        row["parameter_sha256"] = fingerprint
        if fingerprint in canonical:
            aliases.append({"generator_id": name, "canonical_generator": canonical[fingerprint], "parameter_sha256": fingerprint,
                            "checkpoint": file_record(checkpoint), "reason": "identical actual trainable P/R tensors"})
        else:
            canonical[fingerprint] = name
            tasks.append(row)
    _json(OUT / "EVALUATION_QUEUE.json", {"tasks": tasks, "aliases": aliases, "missing": missing, "workers": workers, "device": device})
    _json(path, inventory)

    def evaluate(row: dict) -> dict:
        name = row["generator_id"]
        log_path = OUT / "logs" / ("eval_" + name.replace("/", "_") + ".log")
        print(json.dumps({"event": "start", "generator": name}), flush=True)
        command = [sys.executable, str(ROOT / "src/evaluate_stage1_r26.py"), "--generator", name, "--device", device]
        with log_path.open("w") as handle:
            completed = subprocess.run(command, cwd="/tmp", stdout=handle, stderr=subprocess.STDOUT)
        result = {"generator_id": name, "exit_code": completed.returncode, "log": str(log_path)}
        _json(OUT / "rankings" / name / "PROCESS_RESULT.json", result)
        print(json.dumps({"event": "exit", **result}), flush=True)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(evaluate, row) for row in tasks]):
            results.append(future.result())
    for alias in aliases:
        source = OUT / "rankings" / alias["canonical_generator"]
        receipt = source / "RETRIEVAL_RECEIPT.json"
        if receipt.exists():
            canonical_receipt = json.loads(receipt.read_text())
            if canonical_receipt["signature"]["parameter_sha256"] != alias["parameter_sha256"]:
                raise ValueError("Alias P/R fingerprint differs from executed generator")
            _json(OUT / "rankings" / alias["generator_id"] / "ALIAS_RECEIPT.json",
                  {**alias, "canonical_receipt": file_record(receipt), "rankings": file_record(source / "rankings.jsonl.gz"),
                   "execution_status": "verified_parameter_identical_alias", "scientific_validity": "valid"})
    summary = {"results": results, "aliases": aliases, "missing": missing}
    _json(OUT / "EVALUATION_QUEUE_RESULT.json", summary)
    return {"tasks": len(results), "failures": sum(r["exit_code"] != 0 for r in results), "aliases": len(aliases), "missing": missing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    print(json.dumps(run(args.workers, args.device)))
