"""R22 F0: Fresh Student baseline and D2 transfer.

F0 starts from PCA-1024 + identity-R (not B13), trains all P and R.

Two arms:
  F0-SUP: supervised only (no Teacher KD)
  F0-D2:  supervised + D2 final QT-only KD

Both use the same R20 D0 manifest (42143 lists, 5 relations) and the same
training budget as the E main line (batch=64, 2 epochs = 1318 updates).

Edge stage (this file): trains on edge_lists with all 5 relations.
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
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.scoring import ListScores, score_edge_batch
from mmdd_stage1.training import _student_edge_losses, student_gradient_norms
from run_stage1_r21 import _load_teacher, _score_id_pairs, _teacher_cache_key, _feature_paths, paths as r21_paths
from run_stage1_r22 import paths as r22_paths, out as r22_out, read_rows, write_rows, _teacher_map, SEEDS, BATCH, EPOCHS, TRAIN_LISTS, FINAL_STEP, SEED_HASH, TEACHER_SHA

ROOT = Path(__file__).resolve().parents[1]
# B13 recipe: relation_lr=1e-5, projection_lr=1e-6
RELATION_LR = 1e-5
PROJECTION_LR = 1e-6
WD = 0.01
KD_WEIGHT = 0.3
KD_TEMPERATURE = 1.0
ANCHOR_WEIGHT = 0.1
ANCHOR_WEIGHT_EVIDENCE = 0.1

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _build_fresh_student(root: Path, device: torch.device) -> StudentJoinabilityModel:
    """Build a fresh Student from PCA-1024 + identity-R (no B13 parameters)."""
    pca_path = root / "work/stage1_pca_dimension_ceiling_20260828/pca_spectrum.pt"
    pca_data = torch.load(pca_path, map_location="cpu", weights_only=False)
    pca_basis = pca_data["projection"][:1024].clone()  # [1024, 4096]
    # Verify orthonormality
    gram = pca_basis @ pca_basis.T
    assert torch.allclose(gram, torch.eye(1024), atol=1e-4), "PCA basis not orthonormal"
    model = StudentJoinabilityModel(
        input_dim=4096,
        student_dim=1024,
        initialization="pca",
        initialization_basis=pca_basis,
        initialization_noise_std=0.01,
        freeze_projections=False,
        relation_param="full",
        projection_mode="shared",
    )
    return model.to(device)

def _optimizer(model: StudentJoinabilityModel) -> torch.optim.AdamW:
    """B13-style optimizer: relation_lr=1e-5, projection_lr=1e-6."""
    return torch.optim.AdamW(
        [
            {"params": model.relation_parameters(), "lr": RELATION_LR},
            {"params": model.projections.parameters(), "lr": PROJECTION_LR},
        ],
        weight_decay=WD,
    )

def train_f0_edge(root: Path, arm: str, seed: int, device_name: str,
                  manifest_path: Path | None = None) -> dict[str, Any]:
    """Train a fresh Student edge arm.

    ``manifest_path`` is used by the fresh-lineage common-pool controls; the
    default remains the R22 full-natural manifest used by F0.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest = manifest_path or (r22_out(root) / "manifests" / "full_natural.jsonl")
    if not manifest.exists():
        raise FileNotFoundError(f"Need manifests first: {manifest}")
    job = r22_out(root) / "fresh_lineage" / arm / f"seed{seed}" / "edge"
    final = job / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
    if final.exists() and (job / "config.json").exists():
        return json.loads((job / "config.json").read_text())
    device = torch.device(device_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    examples = load_edge_examples(manifest, split="train")
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=24000)
    model = _build_fresh_student(root, device).train()
    optimizer = _optimizer(model)
    # Load teacher for F0-D2
    teacher = None
    if arm == "F0-D2":
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
            # For F0, all relations are trainable, so use full batch
            ts = None
            if teacher is not None:
                # Only TT rows have teacher soft targets
                tt_indices = [i for i, e in enumerate(batch) if e.source_type == "table" and e.destination_type == "table"]
                if tt_indices:
                    tt_examples = [batch[i] for i in tt_indices]
                    rec = [teacher[_teacher_cache_key(e.query_id, "table->table", e.candidate_ids)] for e in tt_examples]
                    width = max(len(e.candidate_ids) for e in tt_examples)
                    logits = torch.zeros((len(batch), max(len(e.candidate_ids) for e in batch)), device=device)
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    pos = torch.zeros_like(mask)
                    # Fill only TT rows with teacher scores
                    for j, (bi, e, r) in enumerate(zip(tt_indices, tt_examples, rec)):
                        n = len(r["candidate_ids"])
                        logits[bi, :n] = torch.tensor(r["scores"], device=device)
                        mask[bi, :n] = True
                        pos[bi, :n] = torch.tensor([str(x) in set(e.positive_ids) for x in r["candidate_ids"]], device=device)
                    ts = ListScores(logits, mask, pos.float().argmax(1), pos)
            kd_weight = KD_WEIGHT if teacher is not None else 0.0
            terms = _student_edge_losses(
                model, batch, scores, ts, scores, None,
                ranking_weight=1.0,
                temperature=KD_TEMPERATURE,
                distillation_weight=kd_weight,
                edge_bce_weight=0.0,
                anchor_weight=ANCHOR_WEIGHT,
                anchor_weight_evidence=ANCHOR_WEIGHT_EVIDENCE,
            )
            terms["loss"].backward()
            optimizer.step()
            step += 1
            epoch_losses.append(float(terms["loss"].detach()))
            if step % 100 == 0:
                print(json.dumps({"arm": arm, "seed": seed, "step": step, "loss": statistics.fmean(epoch_losses[-100:]), "elapsed": time.monotonic() - started}), flush=True)
        ck = {
            "format_version": 1, "model_kind": "student", "completed_stage": f"r22-{arm}-edge",
            "arm": arm, "seed": seed, "step": step,
            "config": model.config(),
            "trainable_parameters": [n for n, p in model.named_parameters() if p.requires_grad],
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
        "format_version": 1, "status": "pass", "arm": arm, "seed": seed,
        "stage": "edge", "initialization": "fresh_pca_1024",
        "pca_source": "work/stage1_pca_dimension_ceiling_20260828/pca_spectrum.pt",
        "manifest_sha256": checkpoint_fingerprint(manifest),
        "trainable_parameters": [n for n, p in model.named_parameters() if p.requires_grad],
        "updates": step, "device": device_name,
        "optimizer": {"relation_lr": RELATION_LR, "projection_lr": PROJECTION_LR, "weight_decay": WD},
        "kd": {"weight": kd_weight, "temperature": KD_TEMPERATURE, "relations": ["table_to_table"] if teacher else []},
        "anchor": {"weight": ANCHOR_WEIGHT, "evidence": ANCHOR_WEIGHT_EVIDENCE},
        "history": history, "completed_at_utc": now(),
    }
    write_json(job / "config.json", cfg)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="Fresh Student arm name")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Optional edge manifest (for a fresh common-pool control)")
    args = ap.parse_args()
    print(json.dumps({"status": "starting", "arm": args.arm, "seed": args.seed, "device": args.device, "time": now()}), flush=True)
    result = train_f0_edge(ROOT, args.arm, args.seed, args.device, args.manifest)
    print(json.dumps({"status": "edge_done", "arm": args.arm, "seed": args.seed, "time": now()}), flush=True)

if __name__ == "__main__":
    main()
