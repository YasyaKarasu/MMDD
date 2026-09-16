"""Build historical T_core logits for the frozen B2/B3 bridge schedules."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint
from mmdd_stage1.checkpoints import load_teacher
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.scoring import score_edge_batch
from run_stage1_bridge import OUT, R25, ROOT, SEEDS, sha256, write_json
from run_stage1_r25 import _edge_examples_from_schedule, _r25_teacher_feature_paths, c1_teacher_path


def build(seed: int, *, schedule: str = "closure_full", device_name: str = "cuda:0", microbatch: int = 8) -> dict:
    if seed not in SEEDS:
        raise ValueError(seed)
    schedule_path = OUT / f"schedules/seed{seed}_steps659/{schedule}.jsonl.gz"
    if not schedule_path.is_file():
        raise FileNotFoundError(schedule_path)
    destination = OUT / "teacher" / f"edge_historical_seed{seed}_{schedule}.jsonl.gz"
    receipt_path = destination.with_suffix(".json")
    if destination.is_file() and receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    examples = _edge_examples_from_schedule(schedule_path)
    teacher_path = c1_teacher_path(ROOT)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    teacher = load_teacher(teacher_path, device).eval()
    store = FeatureStore.from_path(ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
                                   cache_size=24000, teacher_paths=_r25_teacher_feature_paths(ROOT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    started = time.monotonic()
    rows = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(examples), microbatch):
            batch = examples[start:start + microbatch]
            scores = score_edge_batch(teacher, batch, store, device, student_score_space="raw_logit")
            for index, example in enumerate(batch):
                n = len(example.candidate_ids)
                handle.write(json.dumps({"query_id": str(example.query_id),
                                         "relation": f"{example.source_type}->{example.destination_type}",
                                         "candidate_ids": list(example.candidate_ids),
                                         "scores": [float(x) for x in scores.logits[index, :n].detach().cpu()]}) + "\n")
                rows += 1
            if rows and rows % (microbatch * 100) == 0:
                print(json.dumps({"seed": seed, "schedule": schedule, "rows": rows, "total": len(examples)}), flush=True)
    temporary.replace(destination)
    payload = {"status": "complete", "seed": seed, "schedule": schedule,
               "rows": rows, "microbatch": microbatch, "device": device_name,
               "elapsed_seconds": time.monotonic() - started,
               "schedule_sha256": sha256(schedule_path), "teacher_checkpoint": str(teacher_path),
               "teacher_checkpoint_sha256": sha256(teacher_path), "score_space": "raw_logit",
               "implementation": "historical T_core checkpoint and score_edge_batch on bridge lists",
               "path": str(destination.resolve()), "sha256": sha256(destination),
               "code_sha256": checkpoint_fingerprint(Path(__file__))}
    write_json(receipt_path, payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, choices=SEEDS)
    parser.add_argument("--schedule", default="closure_full", choices=("base_full", "closure_full"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", type=int, default=8)
    args = parser.parse_args()
    for seed in args.seed or list(SEEDS):
        print(json.dumps(build(seed, schedule=args.schedule, device_name=args.device, microbatch=args.microbatch), ensure_ascii=False))
