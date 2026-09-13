"""R22 A1: temperature=4 conditional experiment.

Same as E3 (full-natural TT, R_TT-only, SUP+QT-KD) but with τ=4 instead of τ=1.
Triggered because both lineages have median exp(H) < 3 on the TT soft targets.
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, random, statistics, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.scoring import ListScores, score_edge_batch
from mmdd_stage1.training import _student_edge_losses, student_gradient_norms
from run_stage1_r21 import _load_teacher, _score_id_pairs, _teacher_cache_key, _feature_paths, paths as r21_paths
from run_stage1_r22 import paths as r22_paths, out as r22_out, read_rows, write_rows, _load_examples, _freeze_tt, _teacher_map, SEEDS, BATCH, EPOCHS, TRAIN_LISTS, FINAL_STEP, SEED_HASH, LR, WD, TEACHER_SHA

ROOT = Path(__file__).resolve().parents[1]
TEMPERATURE = 4.0

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def train_a1(root: Path, seed: int, device_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest = r22_out(root) / "manifests" / "full_natural.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Need E2/E3 manifests first: {manifest}")
    job = r22_out(root) / "A1" / f"seed{seed}"
    final = job / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
    if final.exists() and (job / "config.json").exists():
        return json.loads((job / "config.json").read_text())
    device = torch.device(device_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    examples = _load_examples(manifest)
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=24000)
    model = load_student(r22_paths(root)["b13"], device).train()
    params = _freeze_tt(model)
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    # Load teacher soft scores
    cp = r22_out(root) / "teacher_soft_scores" / f"lineage{seed}.jsonl.gz"
    if not cp.exists():
        raise FileNotFoundError(f"Teacher cache missing: {cp}")
    teacher = _teacher_map(cp)
    job.mkdir(parents=True, exist_ok=True)
    (job / "checkpoints").mkdir(exist_ok=True)
    history = []
    rng = random.Random(SEED_HASH + seed)
    order = list(range(len(examples)))
    step = 0
    started = time.monotonic()
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        epoch_losses = []
        for start in range(0, len(order), BATCH):
            batch = [examples[i] for i in order[start:start + BATCH]]
            optimizer.zero_grad(set_to_none=True)
            scores = score_edge_batch(model, batch, store, device)
            tt = [i for i, e in enumerate(batch) if e.source_type == "table" and e.destination_type == "table"]
            if not tt:
                continue
            selected = [batch[i] for i in tt]
            ss = scores.select(torch.tensor([i in tt for i in range(len(batch))], device=device))
            # Build teacher scores
            rec = [teacher[_teacher_cache_key(e.query_id, "table->table", e.candidate_ids)] for e in selected]
            width = max(len(e.candidate_ids) for e in selected)
            logits = torch.zeros((len(rec), width), device=device)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            pos = torch.zeros_like(mask)
            for j, (e, r) in enumerate(zip(selected, rec)):
                n = len(r["candidate_ids"])
                logits[j, :n] = torch.tensor(r["scores"], device=device)
                mask[j, :n] = True
                pos[j, :n] = torch.tensor([str(x) in set(e.positive_ids) for x in r["candidate_ids"]], device=device)
            ts = ListScores(logits, mask, pos.float().argmax(1), pos)
            terms = _student_edge_losses(
                model, selected, ss, ts, ss, None,
                ranking_weight=len(tt) / len(batch),
                temperature=TEMPERATURE,
                distillation_weight=len(tt) / len(batch),
                edge_bce_weight=0.0,
                anchor_weight=0.0,
                anchor_weight_evidence=0.0,
            )
            terms["loss"].backward()
            optimizer.step()
            step += 1
            epoch_losses.append(float(terms["loss"].detach()))
            if step % 100 == 0:
                print(json.dumps({"arm": "A1", "seed": seed, "step": step, "loss": statistics.fmean(epoch_losses[-100:]), "elapsed": time.monotonic() - started}), flush=True)
        ck = {
            "format_version": 1, "model_kind": "student", "completed_stage": "r22-A1",
            "arm": "A1", "seed": seed, "step": step, "temperature": TEMPERATURE,
            "config": {**model.config(), "freeze_projections": True},
            "trainable_parameters": ["relations.table_to_table"],
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(ck, job / "checkpoints" / f"step_{step:06d}.pt")
        history.append({
            "epoch": epoch + 1, "step": step,
            "loss": statistics.fmean(epoch_losses) if epoch_losses else None,
            "checkpoint_sha256": checkpoint_fingerprint(job / "checkpoints" / f"step_{step:06d}.pt"),
        })
        write_rows(job / "train_history.jsonl", [json.loads(json.dumps(h)) for h in history])
    cfg = {
        "format_version": 1, "status": "pass", "arm": "A1", "seed": seed,
        "temperature": TEMPERATURE, "manifest_sha256": checkpoint_fingerprint(manifest),
        "initialization_sha256": checkpoint_fingerprint(r22_paths(root)["b13"]),
        "trainable_parameters": ["relations.table_to_table"],
        "updates": step, "device": device_name, "history": history,
        "completed_at_utc": now(),
    }
    write_json(job / "config.json", cfg)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    print(json.dumps({"status": "starting", "arm": "A1", "seed": args.seed, "temperature": TEMPERATURE, "device": args.device, "time": now()}), flush=True)
    result = train_a1(ROOT, args.seed, args.device)
    print(json.dumps({"status": "train_done", "arm": "A1", "seed": args.seed, "time": now()}), flush=True)

if __name__ == "__main__":
    main()
