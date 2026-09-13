"""R22 F1: Fresh Teacher lineage rebuild.

Stages:
  T0: Train Teacher from scratch on initial candidates (R20 D0 manifest)
  Hard mining: Use T0 to find hard negatives in train natural pool
  T1: Retrain Teacher with original + hard candidates
  Fresh Student (F1-T0, F1-T1, F1-SUP): KD from T0/T1 or SUP-only on same pool

All stages use 2 epochs to be consistent with F0 and E main line.
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, random, statistics, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import EdgeExample, load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.scoring import score_edge_batch
from run_stage1_r19 import R19GlobalResidualTeacher, backward_logical_batch, _edge_loss
from run_stage1_r22 import paths as r22_paths, out as r22_out, read_rows, write_rows, SEEDS, SEED_HASH, TEACHER_SHA
from run_stage1_r21 import _feature_paths, paths as r21_paths

ROOT = Path(__file__).resolve().parents[1]
# Teacher config (same as D2)
TEACHER_CONFIG = dict(
    input_dim=4096, model_dim=512, num_heads=8, num_layers=3,
    text_latents=16, image_latents=24, table_tokens_per_group=1,
    dropout=0.1, global_dim=512, global_hidden_dim=512,
)
TEACHER_BATCH = 8
TEACHER_LR = 5e-5
TEACHER_WD = 0.01
TEACHER_EPOCHS = 2
TEACHER_MICROBATCH = 2

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _fresh_teacher(device: torch.device) -> R19GlobalResidualTeacher:
    """Create a fresh Teacher from scratch (random init)."""
    torch.manual_seed(42)  # deterministic init
    model = R19GlobalResidualTeacher(**TEACHER_CONFIG)
    model.cache_identity = f"fresh_teacher:{hashlib.sha256(json.dumps(TEACHER_CONFIG, sort_keys=True).encode()).hexdigest()}"
    return model.to(device)

def train_teacher(root: Path, stage: str, seed: int, device_name: str,
                  hard_examples: list[EdgeExample] | None = None) -> dict[str, Any]:
    """Train Teacher T0 (no hard examples) or T1 (with hard examples)."""
    if stage not in ("T0", "T1"):
        raise ValueError("stage must be T0 or T1")
    if stage == "T1" and not hard_examples:
        raise ValueError("T1 requires hard_examples")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    job = r22_out(root) / "fresh_lineage" / stage / f"seed{seed}"
    final_ckpt_dir = job / "checkpoints"
    # Count expected updates
    manifest_path = r22_out(root) / "manifests" / "full_natural.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    examples = load_edge_examples(manifest_path, split="train")
    n_lists = len(examples)
    updates_per_epoch = math.ceil(n_lists / TEACHER_BATCH)
    total_updates = TEACHER_EPOCHS * updates_per_epoch
    final_step = total_updates
    final_path = final_ckpt_dir / f"step_{final_step:06d}.pt"
    if final_path.exists() and (job / "config.json").exists():
        return json.loads((job / "config.json").read_text())
    device = torch.device(device_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _fresh_teacher(device).train()
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=4096,
                                   teacher_paths=_feature_paths(r21_paths(root)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=TEACHER_LR, weight_decay=TEACHER_WD)
    job.mkdir(parents=True, exist_ok=True)
    final_ckpt_dir.mkdir(exist_ok=True)
    history = []
    rng = random.Random(SEED_HASH + seed)
    order = list(range(n_lists))
    step = 0
    started = time.monotonic()
    for epoch in range(TEACHER_EPOCHS):
        rng.shuffle(order)
        epoch_losses = []
        for local_start in range(0, len(order), TEACHER_BATCH):
            batch_indices = order[local_start:local_start + TEACHER_BATCH]
            batch = [examples[i] for i in batch_indices]
            optimizer.zero_grad()
            loss, slots = backward_logical_batch(model, batch, store, device, microbatch_lists=TEACHER_MICROBATCH)
            optimizer.step()
            step += 1
            epoch_losses.append(loss)
            if step % 500 == 0:
                print(json.dumps({"stage": stage, "seed": seed, "step": step,
                                  "loss": statistics.fmean(epoch_losses[-500:]),
                                  "elapsed": time.monotonic() - started}), flush=True)
        ck_payload = {
            "format_version": 1, "model_kind": "teacher_r19_global",
            "completed_stage": f"r22-{stage}", "r19_arm": "fresh",
            "continuation_seed": seed, "optimizer_step": step,
            "config": model.config(),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
        }
        ckpt_path = final_ckpt_dir / f"step_{step:06d}.pt"
        torch.save(ck_payload, ckpt_path)
        history.append({
            "epoch": epoch + 1, "step": step,
            "loss": statistics.fmean(epoch_losses) if epoch_losses else None,
            "checkpoint_sha256": checkpoint_fingerprint(ckpt_path),
        })
        write_rows(job / "train_history.jsonl", [json.loads(json.dumps(h)) for h in history])
    cfg = {
        "format_version": 1, "status": "pass", "stage": stage, "seed": seed,
        "initialization": "random_fresh", "teacher_config": TEACHER_CONFIG,
        "manifest_sha256": checkpoint_fingerprint(manifest_path),
        "n_lists": n_lists, "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "optimizer": {"lr": TEACHER_LR, "weight_decay": TEACHER_WD},
        "batch_size": TEACHER_BATCH, "microbatch": TEACHER_MICROBATCH,
        "epochs": TEACHER_EPOCHS,
        "hard_examples_count": len(hard_examples) if hard_examples else 0,
        "history": history, "device": device_name,
        "completed_at_utc": now(),
    }
    write_json(job / "config.json", cfg)
    return cfg


def _load_t0(root: Path, seed: int, device: torch.device) -> R19GlobalResidualTeacher:
    """Load a trained T0 checkpoint."""
    job = r22_out(root) / "fresh_lineage" / "T0" / f"seed{seed}"
    # Find the final checkpoint
    ckpts = sorted((job / "checkpoints").glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No T0 checkpoints for seed {seed}")
    path = ckpts[-1]
    from mmdd_stage1.checkpoints import load_checkpoint
    payload = load_checkpoint(path)
    model = R19GlobalResidualTeacher(**payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.cache_identity = f"T0_seed{seed}:{checkpoint_fingerprint(path)}"
    return model.to(device).eval()


@torch.inference_mode()
def mine_teacher_hard(root: Path, seed: int, device_name: str, top_k: int = 32) -> dict[str, Any]:
    """Branch A: T0 scores natural reservoir candidates, takes top-K non-positive as hard negatives."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    job = r22_out(root) / "fresh_lineage" / "T0_mining" / f"seed{seed}"
    result_path = job / "hard_negatives.jsonl.gz"
    manifest_path = job / "mining_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    device = torch.device(device_name)
    model = _load_t0(root, seed, device)
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=4096,
                                   teacher_paths=_feature_paths(r21_paths(root)))
    # Load full natural manifest and known positives
    manifest = r22_out(root) / "manifests" / "full_natural.jsonl"
    positives_by_query: dict[str, set[str]] = {}
    candidates_by_query: dict[str, list[str]] = {}
    for row in read_rows(manifest):
        if row.get("source_type") != "table" or row.get("destination_type") != "table":
            continue
        q = str(row["query_id"])
        pos = set()
        if row.get("positive_ids"):
            pos.update(str(x) for x in row["positive_ids"])
        if row.get("positive_id"):
            pos.add(str(row["positive_id"]))
        labels = row.get("confirmed_labels")
        cids = [str(x) for x in row.get("candidate_ids", [])]
        if labels:
            for cid, lbl in zip(cids, labels):
                if lbl and lbl > 0:
                    pos.add(cid)
        if q not in positives_by_query:
            positives_by_query[q] = set()
            candidates_by_query[q] = []
        positives_by_query[q].update(pos)
        # Use the longest candidate list for this query
        if len(cids) > len(candidates_by_query[q]):
            candidates_by_query[q] = cids
    job.mkdir(parents=True, exist_ok=True)
    hard_rows = []
    started = time.monotonic()
    cache = model.new_compression_cache() if hasattr(model, 'new_compression_cache') else {}
    queries = sorted(candidates_by_query.keys())
    for i, q in enumerate(queries):
        cids = candidates_by_query[q]
        pos = positives_by_query[q]
        if len(cids) < 2:
            continue
        # Score all candidates
        from run_stage1_r21 import _score_id_pairs
        scores = _score_id_pairs(model, [(q, c) for c in cids], store, device, batch_size=512, cache=cache)
        scored = sorted(zip(cids, scores), key=lambda x: -x[1])
        # Take top-K non-positive
        hard = [(cid, s) for cid, s in scored if cid not in pos][:top_k]
        if hard:
            hard_rows.append({
                "query_id": q,
                "hard_candidate_ids": [h[0] for h in hard],
                "hard_scores": [h[1] for h in hard],
                "n_candidates": len(cids),
                "n_positives": len(pos),
                "n_hard": len(hard),
            })
        if (i + 1) % 2000 == 0:
            print(json.dumps({"stage": "T0_mining", "seed": seed, "queries": i + 1,
                              "total": len(queries), "hard_rows": len(hard_rows),
                              "elapsed": time.monotonic() - started}), flush=True)
    write_rows(result_path, hard_rows)
    result = {
        "format_version": 1, "status": "pass", "seed": seed, "branch": "A",
        "top_k": top_k, "queries_mined": len(hard_rows),
        "total_hard_candidates": sum(r["n_hard"] for r in hard_rows),
        "mining_manifest_sha256": checkpoint_fingerprint(result_path),
        "completed_at_utc": now(),
    }
    write_json(manifest_path, result)
    return result


def train_s0(root: Path, seed: int, device_name: str) -> dict[str, Any]:
    """Branch B: Train S0 = fresh Student with T0 KD (edge, 2 epochs)."""
    from run_stage1_r22_f0 import _build_fresh_student, _optimizer, RELATION_LR, PROJECTION_LR, WD as STUDENT_WD, KD_WEIGHT, KD_TEMPERATURE, ANCHOR_WEIGHT, ANCHOR_WEIGHT_EVIDENCE
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    job = r22_out(root) / "fresh_lineage" / "S0" / f"seed{seed}" / "edge"
    final = job / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
    if final.exists() and (job / "config.json").exists():
        return json.loads((job / "config.json").read_text())
    # First cache T0 soft scores on the full natural manifest
    t0_cache_path = r22_out(root) / "fresh_lineage" / "T0" / f"seed{seed}" / "teacher_soft_scores.jsonl.gz"
    if not t0_cache_path.exists():
        _cache_teacher_scores(root, "T0", seed, device_name, t0_cache_path)
    device = torch.device(device_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    manifest = r22_out(root) / "manifests" / "full_natural.jsonl"
    examples = load_edge_examples(manifest, split="train")
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=24000)
    model = _build_fresh_student(root, device).train()
    optimizer = _optimizer(model)
    from run_stage1_r22 import _teacher_map
    from run_stage1_r21 import _teacher_cache_key
    teacher = _teacher_map(t0_cache_path)
    job.mkdir(parents=True, exist_ok=True)
    (job / "checkpoints").mkdir(exist_ok=True)
    history = []
    rng = random.Random(SEED_HASH + seed)
    order = list(range(len(examples)))
    step = 0
    started = time.monotonic()
    n_lists = len(examples)
    from mmdd_stage1.scoring import ListScores
    for epoch in range(TEACHER_EPOCHS):
        rng.shuffle(order)
        epoch_losses = []
        BATCH = 64
        for start in range(0, len(order), BATCH):
            batch = [examples[i] for i in order[start:start + BATCH]]
            optimizer.zero_grad(set_to_none=True)
            scores = score_edge_batch(model, batch, store, device)
            # Build teacher targets for TT rows only
            tt_indices = [i for i, e in enumerate(batch) if e.source_type == "table" and e.destination_type == "table"]
            ts = None
            if tt_indices:
                tt_examples = [batch[i] for i in tt_indices]
                rec_list = []
                for e in tt_examples:
                    key = _teacher_cache_key(e.query_id, "table->table", e.candidate_ids)
                    rec_list.append(teacher.get(key))
                if all(r is not None for r in rec_list):
                    width = max(len(e.candidate_ids) for e in batch)
                    logits = torch.zeros((len(batch), width), device=device)
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    pos = torch.zeros_like(mask)
                    for bi, e, r in zip(tt_indices, tt_examples, rec_list):
                        n = len(r["candidate_ids"])
                        logits[bi, :n] = torch.tensor(r["scores"], device=device)
                        mask[bi, :n] = True
                        pos[bi, :n] = torch.tensor([str(x) in set(e.positive_ids) for x in r["candidate_ids"]], device=device)
                    ts = ListScores(logits, mask, pos.float().argmax(1), pos)
            from mmdd_stage1.training import _student_edge_losses, student_gradient_norms
            kd_weight = KD_WEIGHT if ts is not None else 0.0
            terms = _student_edge_losses(
                model, batch, scores, ts, scores, None,
                ranking_weight=1.0, temperature=KD_TEMPERATURE,
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
                print(json.dumps({"arm": "S0", "seed": seed, "step": step,
                                  "loss": statistics.fmean(epoch_losses[-100:]),
                                  "elapsed": time.monotonic() - started}), flush=True)
        ck = {
            "format_version": 1, "model_kind": "student",
            "completed_stage": "r22-S0-edge", "arm": "S0", "seed": seed, "step": step,
            "config": model.config(),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(ck, job / "checkpoints" / f"step_{step:06d}.pt")
        history.append({"epoch": epoch + 1, "step": step,
                        "loss": statistics.fmean(epoch_losses) if epoch_losses else None,
                        "checkpoint_sha256": checkpoint_fingerprint(job / "checkpoints" / f"step_{step:06d}.pt")})
        write_rows(job / "train_history.jsonl", [json.loads(json.dumps(h)) for h in history])
    cfg = {
        "format_version": 1, "status": "pass", "arm": "S0", "seed": seed,
        "stage": "edge", "initialization": "fresh_pca_1024",
        "teacher": f"T0_seed{seed}", "kd_weight": KD_WEIGHT,
        "updates": step, "device": device_name,
        "history": history, "completed_at_utc": now(),
    }
    write_json(job / "config.json", cfg)
    return cfg


def _cache_teacher_scores(root: Path, teacher_stage: str, seed: int, device_name: str,
                          output_path: Path, manifest_path: Path | None = None) -> None:
    """Cache Teacher soft scores on TT lists for Student KD."""
    device = torch.device(device_name)
    if teacher_stage == "T0":
        model = _load_t0(root, seed, device)
    else:
        # Load T1-A or T1-B
        job = r22_out(root) / "fresh_lineage" / teacher_stage / f"seed{seed}"
        ckpts = sorted((job / "checkpoints").glob("step_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No {teacher_stage} checkpoints for seed {seed}")
        from mmdd_stage1.checkpoints import load_checkpoint
        payload = load_checkpoint(ckpts[-1])
        model = R19GlobalResidualTeacher(**payload["config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model.cache_identity = f"{teacher_stage}_seed{seed}:{checkpoint_fingerprint(ckpts[-1])}"
        model = model.to(device).eval()
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=4096,
                                   teacher_paths=_feature_paths(r21_paths(root)))
    manifest = manifest_path or (r22_out(root) / "manifests" / "full_natural.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    from run_stage1_r21 import _score_id_pairs
    started = time.monotonic()
    cache = model.new_compression_cache() if hasattr(model, 'new_compression_cache') else {}
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for i, row in enumerate(read_rows(manifest), 1):
            if row.get("source_type") != "table" or row.get("destination_type") != "table":
                continue
            ids = [str(x) for x in row["candidate_ids"]]
            scores = _score_id_pairs(model, [(str(row["query_id"]), x) for x in ids], store, device, batch_size=512, cache=cache)
            f.write(json.dumps({"query_id": str(row["query_id"]), "relation": "table->table",
                                "candidate_ids": ids, "scores": scores}) + "\n")
            if i % 2000 == 0:
                print(json.dumps({"caching": teacher_stage, "seed": seed, "lists": i,
                                  "elapsed": time.monotonic() - started}), flush=True)
    tmp.replace(output_path)


@torch.inference_mode()
def mine_student_ann(root: Path, student_arm: str, seed: int, device_name: str, top_k: int = 32) -> dict[str, Any]:
    """Branch B / F2: Student ANN mining of hard negatives from full lake."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    job = r22_out(root) / "fresh_lineage" / f"{student_arm}_mining" / f"seed{seed}"
    manifest_path = job / "mining_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    device = torch.device(device_name)
    # Load Student checkpoint
    student_dir = r22_out(root) / "fresh_lineage" / student_arm / f"seed{seed}" / "edge"
    ckpts = sorted((student_dir / "checkpoints").glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No {student_arm} checkpoints for seed {seed}")
    student_path = ckpts[-1]
    model = load_student(student_path, device).eval()
    ps = r21_paths(root)
    store = FeatureStore.from_path(ps["features"], cache_size=100000)
    # Build ANN index for table type
    from mmdd_stage1.retrieval import build_indices, load_corpus_ids, StudentANNIndices
    ids_by_type = load_corpus_ids(ps["corpus"], store)
    corpus_sha = checkpoint_fingerprint(ps["corpus"])
    index_dir = job / "index"
    if not (index_dir / "manifest.json").is_file():
        build_indices(model, store, ids_by_type, index_dir, device=device,
                      checkpoint_sha256=checkpoint_fingerprint(student_path),
                      corpus_sha256=corpus_sha, batch_size=4096)
    indices = StudentANNIndices(model, store, index_dir, device=device,
                                checkpoint_sha256=checkpoint_fingerprint(student_path),
                                corpus_sha256=corpus_sha)
    # Load train TT queries and their known positives
    train_manifest = r22_out(root) / "manifests" / "full_natural.jsonl"
    positives_by_query: dict[str, set[str]] = {}
    for row in read_rows(train_manifest):
        if row.get("source_type") != "table" or row.get("destination_type") != "table":
            continue
        q = str(row["query_id"])
        pos = set()
        if row.get("positive_ids"):
            pos.update(str(x) for x in row["positive_ids"])
        if row.get("positive_id"):
            pos.add(str(row["positive_id"]))
        labels = row.get("confirmed_labels")
        cids = [str(x) for x in row.get("candidate_ids", [])]
        if labels:
            for cid, lbl in zip(cids, labels):
                if lbl and lbl > 0:
                    pos.add(cid)
        if q not in positives_by_query:
            positives_by_query[q] = set()
        positives_by_query[q].update(pos)
    # ANN search for each train TT query
    query_ids = sorted(positives_by_query.keys())
    job.mkdir(parents=True, exist_ok=True)
    hard_rows = []
    started = time.monotonic()
    BATCH_Q = 64
    for start in range(0, len(query_ids), BATCH_Q):
        batch_qids = query_ids[start:start + BATCH_Q]
        # Direct ANN search: Q → T top-100
        for qid in batch_qids:
            q_feat = store.embedding_features(qid)
            query_vec = model.relation_query(
                q_feat.embedding.to(device), q_feat.object_type, "table",
                source_role="query"
            ).unsqueeze(0)
            # Search table index
            table_index = indices.table_index if hasattr(indices, 'table_index') else None
            if table_index is not None:
                scores_arr, idx_arr = table_index.search(query_vec.cpu().numpy(), top_k + 50)
                table_ids_list = json.loads((index_dir / "table_ids.json").read_text())
                retrieved = [str(table_ids_list[int(i)]) for i in idx_arr[0] if i >= 0]
            else:
                # Fallback: use indices.query_direct
                from mmdd_stage1.retrieval import retrieve_zero_one_hop_detailed_many
                detail = retrieve_zero_one_hop_detailed_many([qid], indices, k=top_k + 50, direct_k=top_k + 50, evidence_k=0, targets_per_evidence=0, query_batch_size=1)
                retrieved = [str(item["target_id"]) for item in detail[0]["direct"]]
            pos = positives_by_query[qid]
            hard = [cid for cid in retrieved if cid not in pos][:top_k]
            if hard:
                hard_rows.append({
                    "query_id": qid,
                    "hard_candidate_ids": hard,
                    "n_retrieved": len(retrieved),
                    "n_positives": len(pos),
                    "n_hard": len(hard),
                    "source": "student_ann",
                })
        if (start + BATCH_Q) % 2000 < BATCH_Q:
            print(json.dumps({"stage": f"{student_arm}_mining", "seed": seed,
                              "queries": min(start + BATCH_Q, len(query_ids)),
                              "total": len(query_ids), "hard_rows": len(hard_rows),
                              "elapsed": time.monotonic() - started}), flush=True)
    result_path = job / "hard_negatives.jsonl.gz"
    write_rows(result_path, hard_rows)
    result = {
        "format_version": 1, "status": "pass", "seed": seed,
        "student_arm": student_arm, "mining_method": "student_ann",
        "top_k": top_k, "queries_mined": len(hard_rows),
        "total_hard_candidates": sum(r["n_hard"] for r in hard_rows),
        "student_checkpoint_sha256": checkpoint_fingerprint(student_path),
        "mining_manifest_sha256": checkpoint_fingerprint(result_path),
        "completed_at_utc": now(),
    }
    write_json(manifest_path, result)
    return result


def train_teacher_with_hard(root: Path, teacher_name: str, seed: int, device_name: str,
                            hard_source: str) -> dict[str, Any]:
    """Train T1-A or T1-B: fresh Teacher + original lists augmented with mined hard negatives."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    job = r22_out(root) / "fresh_lineage" / teacher_name / f"seed{seed}"
    final_ckpt_dir = job / "checkpoints"
    # Load hard negatives
    hard_dir = r22_out(root) / "fresh_lineage" / hard_source / f"seed{seed}"
    hard_path = hard_dir / "hard_negatives.jsonl.gz"
    if not hard_path.exists():
        raise FileNotFoundError(f"Hard negatives missing: {hard_path}")
    hard_by_query: dict[str, list[str]] = {}
    for row in read_rows(hard_path):
        hard_by_query[str(row["query_id"])] = [str(x) for x in row["hard_candidate_ids"]]
    # Load base manifest and augment TT lists with hard negatives
    manifest = r22_out(root) / "manifests" / "full_natural.jsonl"
    examples = load_edge_examples(manifest, split="train")
    n_augmented = 0
    for ex in examples:
        if ex.source_type == "table" and ex.destination_type == "table":
            hard = hard_by_query.get(ex.query_id, [])
            if hard:
                new_ids = list(dict.fromkeys([*ex.candidate_ids, *hard]))
                if len(new_ids) > len(ex.candidate_ids):
                    # Extend the candidate list
                    old_labels = list(ex.confirmed_labels) if ex.confirmed_labels else [None] * len(ex.candidate_ids)
                    old_labels.extend([None] * (len(new_ids) - len(ex.candidate_ids)))
                    object.__setattr__(ex, 'candidate_ids', new_ids)
                    object.__setattr__(ex, 'confirmed_labels', old_labels)
                    n_augmented += 1
    n_lists = len(examples)
    updates_per_epoch = math.ceil(n_lists / TEACHER_BATCH)
    total_updates = TEACHER_EPOCHS * updates_per_epoch
    final_step = total_updates
    final_path = final_ckpt_dir / f"step_{final_step:06d}.pt"
    if final_path.exists() and (job / "config.json").exists():
        return json.loads((job / "config.json").read_text())
    device = torch.device(device_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _fresh_teacher(device).train()
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=4096,
                                   teacher_paths=_feature_paths(r21_paths(root)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=TEACHER_LR, weight_decay=TEACHER_WD)
    job.mkdir(parents=True, exist_ok=True)
    final_ckpt_dir.mkdir(exist_ok=True)
    history = []
    rng = random.Random(SEED_HASH + seed + 1000)  # Different shuffle from T0
    order = list(range(n_lists))
    step = 0
    started = time.monotonic()
    for epoch in range(TEACHER_EPOCHS):
        rng.shuffle(order)
        epoch_losses = []
        for local_start in range(0, len(order), TEACHER_BATCH):
            batch_indices = order[local_start:local_start + TEACHER_BATCH]
            batch = [examples[i] for i in batch_indices]
            optimizer.zero_grad()
            loss, slots = backward_logical_batch(model, batch, store, device, microbatch_lists=TEACHER_MICROBATCH)
            optimizer.step()
            step += 1
            epoch_losses.append(loss)
            if step % 500 == 0:
                print(json.dumps({"stage": teacher_name, "seed": seed, "step": step,
                                  "loss": statistics.fmean(epoch_losses[-500:]),
                                  "elapsed": time.monotonic() - started}), flush=True)
        ck_payload = {
            "format_version": 1, "model_kind": "teacher_r19_global",
            "completed_stage": f"r22-{teacher_name}", "r19_arm": "fresh",
            "continuation_seed": seed, "optimizer_step": step,
            "config": model.config(),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
        }
        ckpt_path = final_ckpt_dir / f"step_{step:06d}.pt"
        torch.save(ck_payload, ckpt_path)
        history.append({"epoch": epoch + 1, "step": step,
                        "loss": statistics.fmean(epoch_losses) if epoch_losses else None,
                        "checkpoint_sha256": checkpoint_fingerprint(ckpt_path)})
        write_rows(job / "train_history.jsonl", [json.loads(json.dumps(h)) for h in history])
    cfg = {
        "format_version": 1, "status": "pass", "stage": teacher_name, "seed": seed,
        "initialization": "random_fresh", "hard_source": hard_source,
        "n_augmented_lists": n_augmented,
        "total_hard_candidates": sum(len(v) for v in hard_by_query.values()),
        "n_lists": n_lists, "total_updates": total_updates,
        "history": history, "device": device_name, "completed_at_utc": now(),
    }
    write_json(job / "config.json", cfg)
    return cfg


def train_fresh_student_kd(root: Path, arm: str, seed: int, device_name: str,
                           teacher_stage: str, manifest_path: Path | None = None,
                           teacher_cache_path: Path | None = None) -> dict[str, Any]:
    """Train a fresh Student with KD from a specified Teacher (for F1-A-T0, F1-A-T1, S1, S2, etc)."""
    from run_stage1_r22_f0 import _build_fresh_student, _optimizer, KD_WEIGHT, KD_TEMPERATURE, ANCHOR_WEIGHT, ANCHOR_WEIGHT_EVIDENCE
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    job = r22_out(root) / "fresh_lineage" / arm / f"seed{seed}" / "edge"
    final = job / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
    if final.exists() and (job / "config.json").exists():
        return json.loads((job / "config.json").read_text())
    # Ensure Teacher scores are cached
    cache_path = teacher_cache_path or (r22_out(root) / "fresh_lineage" / teacher_stage / f"seed{seed}" / "teacher_soft_scores.jsonl.gz")
    if not cache_path.exists():
        _cache_teacher_scores(root, teacher_stage, seed, device_name, cache_path, manifest_path)
    device = torch.device(device_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    manifest = manifest_path or (r22_out(root) / "manifests" / "full_natural.jsonl")
    examples = load_edge_examples(manifest, split="train")
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=24000)
    model = _build_fresh_student(root, device).train()
    optimizer = _optimizer(model)
    from run_stage1_r22 import _teacher_map
    from run_stage1_r21 import _teacher_cache_key
    teacher = _teacher_map(cache_path)
    job.mkdir(parents=True, exist_ok=True)
    (job / "checkpoints").mkdir(exist_ok=True)
    history = []
    rng = random.Random(SEED_HASH + seed)
    order = list(range(len(examples)))
    step = 0
    started = time.monotonic()
    from mmdd_stage1.scoring import ListScores
    from mmdd_stage1.training import _student_edge_losses
    BATCH = 64
    for epoch in range(TEACHER_EPOCHS):
        rng.shuffle(order)
        epoch_losses = []
        for start in range(0, len(order), BATCH):
            batch = [examples[i] for i in order[start:start + BATCH]]
            optimizer.zero_grad(set_to_none=True)
            scores = score_edge_batch(model, batch, store, device)
            tt_indices = [i for i, e in enumerate(batch) if e.source_type == "table" and e.destination_type == "table"]
            ts = None
            if tt_indices:
                tt_examples = [batch[i] for i in tt_indices]
                rec_list = [teacher.get(_teacher_cache_key(e.query_id, "table->table", e.candidate_ids)) for e in tt_examples]
                if all(r is not None for r in rec_list):
                    width = max(len(e.candidate_ids) for e in batch)
                    logits = torch.zeros((len(batch), width), device=device)
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    pos = torch.zeros_like(mask)
                    for bi, e, r in zip(tt_indices, tt_examples, rec_list):
                        n = len(r["candidate_ids"])
                        logits[bi, :n] = torch.tensor(r["scores"], device=device)
                        mask[bi, :n] = True
                        pos[bi, :n] = torch.tensor([str(x) in set(e.positive_ids) for x in r["candidate_ids"]], device=device)
                    ts = ListScores(logits, mask, pos.float().argmax(1), pos)
            kd_weight = KD_WEIGHT if ts is not None else 0.0
            terms = _student_edge_losses(
                model, batch, scores, ts, scores, None,
                ranking_weight=1.0, temperature=KD_TEMPERATURE,
                distillation_weight=kd_weight,
                edge_bce_weight=0.0,
                anchor_weight=ANCHOR_WEIGHT, anchor_weight_evidence=ANCHOR_WEIGHT_EVIDENCE,
            )
            terms["loss"].backward()
            optimizer.step()
            step += 1
            epoch_losses.append(float(terms["loss"].detach()))
            if step % 100 == 0:
                print(json.dumps({"arm": arm, "seed": seed, "step": step,
                                  "loss": statistics.fmean(epoch_losses[-100:]),
                                  "elapsed": time.monotonic() - started}), flush=True)
        ck = {
            "format_version": 1, "model_kind": "student",
            "completed_stage": f"r22-{arm}-edge", "arm": arm, "seed": seed, "step": step,
            "config": model.config(),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(ck, job / "checkpoints" / f"step_{step:06d}.pt")
        history.append({"epoch": epoch + 1, "step": step,
                        "loss": statistics.fmean(epoch_losses) if epoch_losses else None,
                        "checkpoint_sha256": checkpoint_fingerprint(job / "checkpoints" / f"step_{step:06d}.pt")})
        write_rows(job / "train_history.jsonl", [json.loads(json.dumps(h)) for h in history])
    cfg = {
        "format_version": 1, "status": "pass", "arm": arm, "seed": seed,
        "teacher": teacher_stage, "kd_weight": KD_WEIGHT,
        "updates": step, "device": device_name,
        "history": history, "completed_at_utc": now(),
    }
    write_json(job / "config.json", cfg)
    return cfg


FINAL_STEP = 1318  # Student final step (2 epochs, batch=64, 42143 lists)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=[
        "T0", "T1",
        "mine-teacher", "mine-student",
        "train-s0",
        "train-teacher-hard",
        "train-student-kd",
        "cache-teacher",
    ])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--teacher-stage", default="T0", help="Teacher stage for cache/KD")
    ap.add_argument("--teacher-name", default="T1-A", help="Name for teacher-with-hard (T1-A, T1-B)")
    ap.add_argument("--hard-source", default="T0_mining", help="Hard negative source dir name")
    ap.add_argument("--student-arm", default="S0", help="Student arm for mining")
    ap.add_argument("--arm", default="F1-A-T0", help="Student arm name for train-student-kd")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--teacher-cache", type=Path, default=None)
    args = ap.parse_args()
    print(json.dumps({"status": "starting", "stage": args.stage, "seed": args.seed,
                      "device": args.device, "time": now()}), flush=True)
    if args.stage in ("T0", "T1"):
        result = train_teacher(ROOT, args.stage, args.seed, args.device)
    elif args.stage == "mine-teacher":
        result = mine_teacher_hard(ROOT, args.seed, args.device)
    elif args.stage == "mine-student":
        result = mine_student_ann(ROOT, args.student_arm, args.seed, args.device)
    elif args.stage == "train-s0":
        result = train_s0(ROOT, args.seed, args.device)
    elif args.stage == "train-teacher-hard":
        result = train_teacher_with_hard(ROOT, args.teacher_name, args.seed, args.device, args.hard_source)
    elif args.stage == "train-student-kd":
        result = train_fresh_student_kd(ROOT, args.arm, args.seed, args.device, args.teacher_stage,
                                        args.manifest, args.teacher_cache)
    elif args.stage == "cache-teacher":
        out_path = args.teacher_cache or (r22_out(ROOT) / "fresh_lineage" / args.teacher_stage / f"seed{args.seed}" / "teacher_soft_scores.jsonl.gz")
        _cache_teacher_scores(ROOT, args.teacher_stage, args.seed, args.device, out_path, args.manifest)
        result = {"status": "pass", "path": str(out_path)}
    else:
        raise ValueError(f"Unknown stage: {args.stage}")
    print(json.dumps({"status": "done", "stage": args.stage, "seed": args.seed,
                      "time": now()}), flush=True)


if __name__ == "__main__":
    main()
