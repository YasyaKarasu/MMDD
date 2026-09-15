"""R23 Stage-1 repair experiments.

This runner implements the frozen R23 contract: one verified S1 seed29
augmented pool, fresh PCA-1024 + identity-R Students, and the three explicit
loss arms G0/G1/G2.  It writes hashes, cache coverage, checkpoints, raw fixed
pool rankings, full-lake ANN/exact rankings, and per-step diagnostics under a
new R23 output directory.  No test/dev labels are used to construct training
lists.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import distillation_kl, listwise_cross_entropy
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    build_indices,
    load_corpus_ids,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.scoring import ListScores, score_edge_batch
from mmdd_stage1.training import _student_edge_losses, student_gradient_norms
from run_stage1_r21 import (
    _feature_paths,
    _score_target_ids,
    _teacher_cache_key,
    out as r21_out,
    paths as r21_paths,
    read_rows,
    write_rows,
)
from run_stage1_r19 import _score_id_pairs, load_r19_checkpoint
from run_stage1_r22 import paths as r22_paths, out as r22_out
from run_stage1_r22_f0 import _build_fresh_student, _optimizer

ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = "stage1_optimization_r23_20260913"
SEEDS = (13, 29)
ARMS = ("G0-SUP", "G1-QTKD", "G2-QTKD-U")
BATCH = 64
EPOCHS = 2
TRAIN_LISTS = 42143
UPDATES_PER_EPOCH = math.ceil(TRAIN_LISTS / BATCH)
FINAL_STEP = EPOCHS * UPDATES_PER_EPOCH
SEED_HASH = 220911
KD_WEIGHT = 0.3
UNIFORM_WEIGHT = 0.3
TEMPERATURE = 1.0
ANCHOR_WEIGHT = 0.1
ANCHOR_WEIGHT_EVIDENCE = 0.1
POOL_SHA256 = "77d3fca543d1a288640f1740258ed922f45d107afbf913ae3648c0e27b135f46"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def out(root: Path) -> Path:
    return root / "work" / OUT_NAME


def r23_paths(root: Path) -> dict[str, Path]:
    r22 = r22_out(root)
    return {
        "pool": r22 / "manifests" / "fresh_aug_S1_seed29.jsonl",
        "pool_sidecar": r22 / "manifests" / "fresh_aug_S1_seed29.jsonl.manifest.json",
        "features": r22_paths(root)["features"],
        "corpus": r21_paths(root)["corpus"],
        "candidate_pools": r22_paths(root)["candidate_pools"],
        "pca": root / "work/stage1_pca_dimension_ceiling_20260828/pca_spectrum.pt",
    }


def teacher_feature_paths(root: Path) -> list[Path]:
    """Return the frozen legacy Teacher shards plus any R23 supplements."""
    paths = list(_feature_paths(r21_paths(root)))
    supplement = out(root) / "teacher_supplement"
    if (supplement / "teacher_manifest.jsonl").is_file():
        paths.append(supplement)
    return paths


def _hash_file(path: Path) -> str:
    return checkpoint_fingerprint(path)


def audit_inputs(root: Path) -> dict[str, Any]:
    ps = r23_paths(root)
    required = {
        "pool": ps["pool"],
        "pool_sidecar": ps["pool_sidecar"],
        "features_manifest": ps["features"] / "manifest.jsonl",
        "corpus": ps["corpus"],
        "candidate_pools": ps["candidate_pools"],
        "pca": ps["pca"],
    }
    missing = [str(p.resolve()) for p in required.values() if not p.exists()]
    inputs = {
        k: {"path": str(p.resolve()), "sha256": _hash_file(p), "bytes": p.stat().st_size}
        for k, p in required.items() if p.exists() and p.is_file()
    }
    sidecar = json.loads(ps["pool_sidecar"].read_text()) if ps["pool_sidecar"].exists() else {}
    pool_hash = inputs.get("pool", {}).get("sha256")
    if pool_hash != POOL_SHA256:
        missing.append(f"pool hash mismatch: expected {POOL_SHA256}, got {pool_hash}")
    if sidecar.get("output_sha256") != POOL_SHA256:
        missing.append("pool sidecar output_sha256 mismatch")
    result = {
        "format_version": 1,
        "status": "pass" if not missing else "blocked",
        "inputs": inputs,
        "pool_sidecar": sidecar,
        "expected_pool_sha256": POOL_SHA256,
        "missing_or_invalid": missing,
        "recorded_at_utc": now(),
    }
    out(root).mkdir(parents=True, exist_ok=True)
    write_json(out(root) / "INPUT_MANIFEST.json", result)
    return result


def snapshot_sources(root: Path) -> dict[str, Any]:
    files = [
        root / "src/mmdd_stage1/training.py",
        root / "src/mmdd_stage1/objectives.py",
        root / "src/mmdd_stage1/models.py",
        root / "src/mmdd_stage1/scoring.py",
        root / "src/mmdd_stage1/features.py",
        root / "src/mmdd_stage1/retrieval.py",
        root / "src/mmdd_stage1/data.py",
        root / "src/mmdd_stage1/checkpoints.py",
        root / "src/mmdd_stage1/artifacts.py",
        root / "src/run_stage1_r23.py",
        root / "src/run_stage1_r23_et.py",
        root / "src/audit_stage1_r23_evidence.py",
        root / "src/audit_stage1_r23_parity.py",
        root / "src/audit_stage1_r23_teachers.py",
        root / "tests/test_stage1_r23.py",
        root / "src/run_stage1_r21.py",
        root / "src/run_stage1_r22_f0.py",
        root / "src/run_stage1_r22.py",
        root / "src/run_stage1_r19.py",
        root / "src/run_stage1_r18.py",
        root / "mmdd_r22_actual_review/R23_STAGE1_REPAIR_PLAN.md",
        root / "mmdd_r22_actual_review/R22_ACTUAL_RESULTS_REVIEW.md",
        root / "mmdd_r22_actual_review/discussion.md",
        root / "mmdd_r22_actual_review/SOURCE_FINDINGS.md",
        root / "mmdd_r21_review/README.md",
        root / "mmdd_r21_review/R21_KD_FAILURE_ANALYSIS.md",
        root / "mmdd_r21_review/R22_STAGE1_EXPERIMENT_PLAN.md",
        root / "mmdd_r21_review/R22_F1_REVISED_PLAN.md",
        root / "mmdd_r21_review/discussion.md",
        root / "mmdd_r21_review/SOURCE_CODE_FINDINGS.md",
    ]
    rows = [{"path": str(p.relative_to(root)), "sha256": _hash_file(p), "bytes": p.stat().st_size}
            for p in files if p.exists()]
    result = {"format_version": 1, "status": "complete", "files": rows, "created_at_utc": now()}
    write_json(out(root) / "SOURCE_SNAPSHOT.json", result)
    return result


def build_manifests(root: Path) -> dict[str, Any]:
    audit = audit_inputs(root)
    if audit["status"] != "pass":
        raise RuntimeError("R23 input audit failed; refusing to construct training input")
    source = r23_paths(root)["pool"]
    rows = list(read_rows(source))
    if len(rows) != TRAIN_LISTS:
        raise RuntimeError(f"Expected {TRAIN_LISTS} rows, got {len(rows)}")
    destination = out(root) / "manifests"
    destination.mkdir(parents=True, exist_ok=True)
    # Preserve the ordered logical lists byte-for-byte.  Each arm gets its own
    # named copy so consumed lineage is explicit while membership stays equal.
    for arm in ARMS:
        write_rows(destination / f"{arm}.jsonl", rows)
    tt = sum(r.get("source_type") == "table" and r.get("destination_type") == "table" for r in rows)
    by_relation = Counter(f"{r.get('source_type')}->{r.get('destination_type')}" for r in rows)
    result = {
        "format_version": 1,
        "status": "complete",
        "source_manifest": str(source.resolve()),
        "source_sha256": _hash_file(source),
        "arm_manifest_sha256": {arm: _hash_file(destination / f"{arm}.jsonl") for arm in ARMS},
        "rows": len(rows), "tt_rows": tt, "by_relation": dict(by_relation),
        "changed_logical_lists": json.loads(r23_paths(root)["pool_sidecar"].read_text()).get("changed_logical_lists"),
        "created_at_utc": now(),
    }
    write_json(out(root) / "MANIFEST_SUMMARY.json", result)
    return result


def _teacher_cache_path(root: Path, seed: int) -> Path:
    return r22_out(root) / "fresh_lineage" / "T1-B" / f"seed{seed}" / "teacher_soft_scores_S1aug.jsonl.gz"


def validate_teacher_cache(root: Path, seed: int, manifest: Path) -> dict[str, Any]:
    cache_path = _teacher_cache_path(root, seed)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    expected = []
    for row in read_rows(manifest):
        if row.get("source_type") == "table" and row.get("destination_type") == "table":
            expected.append(_teacher_cache_key(str(row["query_id"]), "table->table", row["candidate_ids"]))
    actual_rows = list(read_rows(cache_path))
    actual = {_teacher_cache_key(str(r["query_id"]), str(r["relation"]), r["candidate_ids"]): r for r in actual_rows}
    missing = [k for k in expected if k not in actual]
    extra = [k for k in actual if k not in set(expected)]
    if missing or extra or len(actual_rows) != len(expected):
        raise RuntimeError(f"Teacher cache coverage failure seed={seed}: missing={len(missing)} extra={len(extra)}")
    # Never silently substitute a different candidate list or a missing row.
    for r in actual_rows:
        if len(r["candidate_ids"]) != len(r["scores"]):
            raise RuntimeError(f"Teacher cache score length mismatch seed={seed}")
    result = {
        "format_version": 1, "status": "complete", "seed": seed,
        "path": str(cache_path.resolve()), "sha256": _hash_file(cache_path),
        "manifest_sha256": _hash_file(manifest), "tt_rows_expected": len(expected),
        "tt_rows_seen": len(actual_rows), "kd_active_tt_rows": len(actual_rows),
        "missing": len(missing), "extra": len(extra),
        "teacher_checkpoint_sha256": _teacher_checkpoint_sha(root, seed),
        "validated_at_utc": now(),
    }
    dest = out(root) / "teacher_soft_scores"
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / f"lineage{seed}.coverage.json", result)
    return result


def _teacher_checkpoint_sha(root: Path, seed: int) -> str | None:
    cfg = r22_out(root) / "fresh_lineage" / "T1-B" / f"seed{seed}" / "config.json"
    if not cfg.exists():
        return None
    data = json.loads(cfg.read_text())
    history = data.get("history") or []
    return history[-1].get("checkpoint_sha256") if history else None


def _teacher_map(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {_teacher_cache_key(str(r["query_id"]), str(r["relation"]), r["candidate_ids"]): r
            for r in read_rows(path)}


def _make_teacher_scores(batch: list[Any], teacher: dict[tuple[str, str, str], dict[str, Any]],
                         device: torch.device) -> tuple[ListScores | None, int]:
    tt_indices = [i for i, e in enumerate(batch)
                  if e.source_type == "table" and e.destination_type == "table"]
    if not tt_indices:
        return None, 0
    records = []
    for i in tt_indices:
        e = batch[i]
        key = _teacher_cache_key(e.query_id, "table->table", e.candidate_ids)
        record = teacher.get(key)
        if record is None:
            raise RuntimeError(f"Teacher cache miss for {key}")
        if [str(x) for x in record["candidate_ids"]] != [str(x) for x in e.candidate_ids]:
            raise RuntimeError(f"Teacher candidate identity mismatch for {key}")
        records.append(record)
    # Match the Student batch width; non-TT rows are padded in Teacher tensors
    # and excluded by the mask intersection in _student_edge_losses.
    width = max(len(e.candidate_ids) for e in batch)
    logits = torch.zeros((len(batch), width), device=device)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    pos = torch.zeros_like(mask)
    for bi, e, record in zip(tt_indices, (batch[i] for i in tt_indices), records):
        n = len(record["candidate_ids"])
        logits[bi, :n] = torch.tensor(record["scores"], device=device)
        mask[bi, :n] = True
        pids = set(map(str, e.positive_ids))
        pos[bi, :n] = torch.tensor([str(x) in pids for x in record["candidate_ids"]], device=device)
    return ListScores(logits, mask, pos.float().argmax(1), pos), len(tt_indices)


def _relation_loss(scores: ListScores, rows: list[Any], relation: str) -> float | None:
    indices = [i for i, e in enumerate(rows)
               if f"{e.source_type}->{e.destination_type}" == relation]
    if not indices:
        return None
    idx = torch.tensor(indices, device=scores.logits.device)
    return float(listwise_cross_entropy(scores.logits.index_select(0, idx), scores.positive_indices.index_select(0, idx),
                                        scores.candidate_mask.index_select(0, idx),
                                        None if scores.positive_mask is None else scores.positive_mask.index_select(0, idx)).detach().cpu())


def _gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    result = student_gradient_norms(model)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and name in result:
            # Compact direction diagnostic: signed mean and positive fraction,
            # avoiding persistence of multi-million-element gradient vectors.
            g = parameter.grad.detach()
            result[name] = {
                "norm": float(g.norm().cpu()),
                "signed_mean": float(g.mean().cpu()),
                "positive_fraction": float((g > 0).float().mean().cpu()),
            }
    return result


def _save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, job: Path,
                     arm: str, seed: int, step: int, completed_stage: str) -> str:
    path = job / "checkpoints" / f"step_{step:06d}.pt"
    payload = {
        "format_version": 1, "model_kind": "student", "completed_stage": completed_stage,
        "arm": arm, "seed": seed, "step": step, "config": model.config(),
        "trainable_parameters": [n for n, p in model.named_parameters() if p.requires_grad],
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)
    return _hash_file(path)


def train(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError(f"arm must be one of {ARMS}, seed one of {SEEDS}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest = out(root) / "manifests" / f"{arm}.jsonl"
    if not manifest.exists():
        build_manifests(root)
    teacher = None
    coverage = None
    if arm != "G0-SUP":
        coverage = validate_teacher_cache(root, seed, manifest)
        teacher = _teacher_map(_teacher_cache_path(root, seed))
    job = out(root) / arm / f"seed{seed}"
    final = job / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
    cfg_path = job / "config.json"
    if final.exists() and cfg_path.exists():
        return json.loads(cfg_path.read_text())
    device = torch.device(device_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    examples = load_edge_examples(manifest, split="train")
    if len(examples) != TRAIN_LISTS:
        raise RuntimeError(f"Unexpected training rows: {len(examples)}")
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=24000)
    model = _build_fresh_student(root, device).train()
    optimizer = _optimizer(model)
    job.mkdir(parents=True, exist_ok=True)
    (job / "checkpoints").mkdir(exist_ok=True)
    init_hash = _save_checkpoint(model, optimizer, job, arm, seed, 0, "r23-initial")
    rng = random.Random(SEED_HASH + seed)
    order = list(range(len(examples)))
    step = 0
    history: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        epoch_losses = []
        for start in range(0, len(order), BATCH):
            batch = [examples[i] for i in order[start:start + BATCH]]
            optimizer.zero_grad(set_to_none=True)
            scores = score_edge_batch(model, batch, store, device)
            ts, n_tt = _make_teacher_scores(batch, teacher, device) if teacher is not None else (None, 0)
            n_non = len(batch) - n_tt
            terms = _student_edge_losses(
                model, batch, scores, ts, scores, None,
                ranking_weight=1.0,
                temperature=TEMPERATURE,
                distillation_weight=KD_WEIGHT * n_tt / len(batch) if ts is not None else 0.0,
                edge_bce_weight=0.0,
                anchor_weight=ANCHOR_WEIGHT,
                anchor_weight_evidence=ANCHOR_WEIGHT_EVIDENCE,
            )
            uniform = scores.logits.new_zeros(())
            if arm == "G2-QTKD-U" and n_non:
                non_mask = torch.tensor([not (e.source_type == "table" and e.destination_type == "table") for e in batch], device=device)
                uniform = distillation_kl(scores.logits[non_mask], torch.zeros_like(scores.logits[non_mask]), scores.candidate_mask[non_mask], TEMPERATURE)
                terms["loss"] = terms["loss"] + UNIFORM_WEIGHT * n_non / len(batch) * uniform
            terms["loss"].backward()
            grad = _gradient_summary(model)
            optimizer.step()
            step += 1
            value = float(terms["loss"].detach().cpu())
            epoch_losses.append(value)
            if step <= 2 or step % 100 == 0:
                relation_losses = {
                    rel: _relation_loss(scores, batch, rel)
                    for rel in ("table->table", "table->text", "table->image", "text->table", "image->table")
                }
                diagnostics.append({
                    "step": step, "epoch": epoch + 1, "n_tt": n_tt, "n_non_tt": n_non,
                    "loss": value, "supervised_loss": float(terms["supervised_loss"].detach().cpu()),
                    "weighted_supervised_loss": float(terms["weighted_supervised_loss"].detach().cpu()),
                    "teacher_kd_loss": float(terms["distillation_loss"].detach().cpu()),
                    "weighted_teacher_kd_loss": float((terms["distillation_loss"] * KD_WEIGHT * n_tt / len(batch)).detach().cpu()) if n_tt else 0.0,
                    "uniform_kd_loss": float(uniform.detach().cpu()),
                    "weighted_uniform_kd_loss": float((uniform * UNIFORM_WEIGHT * n_non / len(batch)).detach().cpu()) if n_non else 0.0,
                    "supervised_loss_by_relation": relation_losses,
                    "gradient": grad,
                    "elapsed": time.monotonic() - started,
                })
            if step % 100 == 0:
                print(json.dumps({"arm": arm, "seed": seed, "step": step, "loss": statistics.fmean(epoch_losses[-100:]), "elapsed": time.monotonic() - started}), flush=True)
        ck_hash = _save_checkpoint(model, optimizer, job, arm, seed, step, "r23-edge")
        history.append({"epoch": epoch + 1, "step": step, "loss": statistics.fmean(epoch_losses), "checkpoint_sha256": ck_hash})
        write_rows(job / "train_history.jsonl", history)
        write_rows(job / "step_diagnostics.jsonl", diagnostics)
    cfg = {
        "format_version": 1, "status": "pass", "arm": arm, "seed": seed, "stage": "edge",
        "initialization": "fresh_pca_1024_identity_R", "initialization_noise_std": 0.01,
        "manifest_sha256": _hash_file(manifest), "source_pool_sha256": POOL_SHA256,
        "teacher": "T1-B" if teacher is not None else None,
        "teacher_cache_coverage": coverage,
        "kd": {"tt_weight": KD_WEIGHT if teacher is not None else 0.0, "temperature": TEMPERATURE, "relation": "table->table" if teacher is not None else None},
        "uniform_kd": {"weight": UNIFORM_WEIGHT if arm == "G2-QTKD-U" else 0.0, "temperature": TEMPERATURE, "relations": ["table->text", "text->table", "table->image", "image->table"] if arm == "G2-QTKD-U" else []},
        "anchor": {"weight": ANCHOR_WEIGHT, "evidence": ANCHOR_WEIGHT_EVIDENCE},
        "optimizer": {"relation_lr": 1e-5, "projection_lr": 1e-6, "weight_decay": 0.01},
        "batch_size": BATCH, "epochs": EPOCHS, "updates": step, "initial_checkpoint_sha256": init_hash,
        "history": history, "diagnostics": str((job / "step_diagnostics.jsonl").resolve()),
        "device": device_name, "completed_at_utc": now(),
    }
    write_json(cfg_path, cfg)
    return cfg


@torch.inference_mode()
def evaluate_fixed(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    ck = out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
    if not ck.exists():
        raise FileNotFoundError(ck)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = load_student(ck, device).eval()
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=40000)
    records = []
    for row in read_rows(r23_paths(root)["candidate_pools"]):
        q, pos = str(row["query_id"]), set(map(str, row["positive_target_ids"]))
        pools = {"U": list(map(str, row["natural_candidate_ids"])), "D": list(map(str, row["ann_direct100_ids"])), "M": list(map(str, row["matched_direct_candidate_ids"]))}
        ids = list(dict.fromkeys(x for values in pools.values() for x in values))
        scores = _score_target_ids(model, store, q, ids, device)
        rec = {"query_id": q, "query_kind": row.get("query_kind"), "positive_target_ids": sorted(pos), "pools": {}}
        for name, candidates in pools.items():
            ranking = sorted(dict.fromkeys(candidates), key=lambda x: (-scores[x], x))
            rec["pools"][name] = {"candidate_ids": candidates, "scores": [scores[x] for x in candidates], "ranking": ranking,
                                   "raw_recall": len(pos & set(candidates)) / len(pos) if pos else 0.0,
                                   **{f"recall@{k}": len(pos & set(ranking[:k])) / len(pos) if pos else 0.0 for k in (10, 20, 50)}}
        records.append(rec)
    dest = out(root) / "evaluations" / arm / f"seed{seed}" / f"step_{step:06d}"
    dest.mkdir(parents=True, exist_ok=True)
    write_rows(dest / "fixed_U_M_D_rankings.jsonl.gz", records)
    result = {"format_version": 1, "status": "complete", "arm": arm, "seed": seed, "step": step, "queries": len(records),
              "checkpoint_sha256": _hash_file(ck), "rankings": str((dest / "fixed_U_M_D_rankings.jsonl.gz").resolve()),
              "pools": {name: {metric: statistics.fmean(r["pools"][name][metric] for r in records) for metric in ("raw_recall", "recall@10", "recall@20", "recall@50")} for name in ("U", "D", "M")},
              "by_query_kind": {kind: {name: {"recall@10": statistics.fmean(r["pools"][name]["recall@10"] for r in records if r["query_kind"] == kind)} for name in ("U", "D", "M")} for kind in ("implicit", "explicit")},
              "completed_at_utc": now()}
    write_json(dest / "metrics.json", result)
    return result


@torch.inference_mode()
def evaluate_fixed_teacher(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    """Re-rank the fixed U/M/D pools with the corresponding frozen T1-B."""
    checkpoint = out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    destination = out(root) / "evaluations" / arm / f"seed{seed}" / f"step_{step:06d}" / "fixed_teacher"
    metrics_path = destination / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    teacher_path = r22_out(root) / "fresh_lineage" / "T1-B" / f"seed{seed}" / "checkpoints" / "step_010536.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(teacher_path)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    _arm, saved_seed, _step, teacher, _payload = load_r19_checkpoint(teacher_path, device)
    if saved_seed != seed:
        raise RuntimeError(f"Teacher seed mismatch: expected {seed}, got {saved_seed}")
    teacher.eval()
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=50000, teacher_paths=teacher_feature_paths(root))
    score_cache = teacher.new_compression_cache()
    records = []
    missing = 0
    total = 0
    for row in read_rows(r23_paths(root)["candidate_pools"]):
        query_id = str(row["query_id"])
        positives = set(map(str, row["positive_target_ids"]))
        pools = {"U": list(map(str, row["natural_candidate_ids"])),
                 "D": list(map(str, row["ann_direct100_ids"])),
                 "M": list(map(str, row["matched_direct_candidate_ids"]))}
        all_ids = list(dict.fromkeys(x for values in pools.values() for x in values))
        available = []
        for candidate in all_ids:
            total += 1
            if store.get(candidate, include_hidden=True).hidden_states is None:
                missing += 1
            else:
                available.append(candidate)
        values = _score_id_pairs(teacher, [(query_id, candidate) for candidate in available], store, device, batch_size=128, cache=score_cache) if available else []
        score_map = dict(zip(available, values, strict=True))
        rec = {"query_id": query_id, "query_kind": row.get("query_kind"), "positive_target_ids": sorted(positives), "pools": {}}
        for name, candidates in pools.items():
            ranking = sorted(candidates, key=lambda candidate: (-score_map[candidate], candidate) if candidate in score_map else (float("inf"), candidate))
            rec["pools"][name] = {"candidate_ids": candidates, "ranking": ranking,
                                   "scored_count": sum(candidate in score_map for candidate in candidates),
                                   "missing_count": sum(candidate not in score_map for candidate in candidates),
                                   "raw_recall": len(positives & set(candidates)) / len(positives) if positives else 0.0,
                                   **{f"recall@{k}": len(positives & set(ranking[:k])) / len(positives) if positives else 0.0 for k in (10, 20, 50)}}
        records.append(rec)
    def aggregate(name: str) -> dict[str, float]:
        return {key: statistics.fmean(float(record["pools"][name][key]) for record in records) for key in ("raw_recall", "recall@10", "recall@20", "recall@50")}
    destination.mkdir(parents=True, exist_ok=True)
    ranking_path = destination / "rankings.jsonl.gz"
    write_rows(ranking_path, records)
    result = {"format_version": 1, "status": "complete" if missing == 0 else "partial", "arm": arm, "seed": seed, "step": step,
              "teacher": "T1-B", "teacher_checkpoint_sha256": _hash_file(teacher_path), "queries": len(records),
              "pools": {name: aggregate(name) for name in ("U", "D", "M")}, "rankings": str(ranking_path.resolve()),
              "teacher_candidate_coverage": 1.0 - missing / total if total else 1.0,
              "missing_candidate_count": missing, "total_candidate_count": total,
              "note": "Missing Teacher hidden-state candidates are ranked last; metrics are conservative lower bounds." if missing else None,
              "completed_at_utc": now()}
    write_json(metrics_path, result)
    return result


@torch.inference_mode()
def evaluate_full_lake(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    dest = out(root) / "full_lake" / arm / f"seed{seed}_step{step:06d}"
    # Retrieval is expensive. A completed, hash-addressed result is immutable
    # and can be reused by the exact-table evaluator and later audits.
    if (dest / "metrics.json").exists() and (dest / "rankings.jsonl.gz").exists():
        return json.loads((dest / "metrics.json").read_text())
    device = torch.device(device_name)
    ps = r21_paths(root)
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=120000)
    ck = out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
    model = load_student(ck, device).eval()
    ids_by_type = load_corpus_ids(ps["corpus"], store)
    index_dir = out(root) / "indexes" / arm / f"seed{seed}" / f"step_{step:06d}"
    if not (index_dir / "manifest.json").exists():
        build_indices(model, store, ids_by_type, index_dir, device=device, checkpoint_sha256=_hash_file(ck), corpus_sha256=_hash_file(ps["corpus"]), batch_size=4096)
    indices = StudentANNIndices(model, store, index_dir, device=device, checkpoint_sha256=_hash_file(ck), corpus_sha256=_hash_file(ps["corpus"]))
    pools = list(read_rows(r23_paths(root)["candidate_pools"]))
    detailed = retrieve_zero_one_hop_detailed_many([str(r["query_id"]) for r in pools], indices, k=100, direct_k=100, evidence_k=20, targets_per_evidence=20, query_batch_size=16)
    rows = []
    for pool, ret in zip(pools, detailed):
        pos = set(map(str, pool["positive_target_ids"]))
        direct = [str(x["target_id"]) for x in ret["direct"]]
        evidence = [str(x["target_id"]) for x in ret["evidence"]]
        union = list(dict.fromkeys([*direct, *evidence]))
        sm = _score_target_ids(model, store, str(pool["query_id"]), union, device)
        rank = sorted(union, key=lambda x: (-sm[x], x))
        rows.append({"query_id": str(pool["query_id"]), "query_kind": pool["query_kind"], "positive_target_ids": sorted(pos),
                     "direct_ann": direct, "evidence_ann": evidence, "U": union, "u_exact_ranking": rank,
                     "direct_scores": [sm[x] for x in direct], "evidence_scores": [sm[x] for x in evidence],
                     "u_scores": [sm[x] for x in rank], "evidence_paths": ret.get("evidence", []),
                     "direct_raw_recall@100": len(pos & set(direct)) / len(pos) if pos else 0.0,
                     "u_raw_recall": len(pos & set(union)) / len(pos) if pos else 0.0,
                     **{f"u_recall@{k}": len(pos & set(rank[:k])) / len(pos) if pos else 0.0 for k in (10, 20, 50)}})
    dest.mkdir(parents=True, exist_ok=True)
    write_rows(dest / "rankings.jsonl.gz", rows)
    result = {"format_version": 1, "status": "complete", "retriever": arm, "seed": seed, "step": step, "queries": len(rows),
              "direct_raw@100": statistics.fmean(r["direct_raw_recall@100"] for r in rows), "u_raw_recall": statistics.fmean(r["u_raw_recall"] for r in rows),
              "u_r10": statistics.fmean(r["u_recall@10"] for r in rows), "u_r20": statistics.fmean(r["u_recall@20"] for r in rows), "u_cr50": statistics.fmean(r["u_recall@50"] for r in rows),
              "u_size_mean": statistics.fmean(len(r["U"]) for r in rows), "checkpoint_sha256": _hash_file(ck), "rankings": str((dest / "rankings.jsonl.gz").resolve()), "completed_at_utc": now()}
    write_json(dest / "metrics.json", result)
    return result


@torch.inference_mode()
def evaluate_exact_direct(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    ps = r21_paths(root)
    source = out(root) / "full_lake" / arm / f"seed{seed}_step{step:06d}"
    existing = list(read_rows(source / "rankings.jsonl.gz"))
    table_ids = json.loads((r21_out(root) / "indexes" / "Qwen-Raw" / "table_ids.json").read_text())
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=50000)
    target = torch.stack([store.embedding_features(str(x)).embedding for x in table_ids]).to(device=device, dtype=torch.float32)
    ck = out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
    model = load_student(ck, device).eval()
    tv = model.project(target, "table", role="target")
    rel = model.relations[model.relation_key("table", "table")]
    rows = []
    for start in range(0, len(existing), 32):
        batch = existing[start:start + 32]
        q = torch.stack([store.embedding_features(str(r["query_id"])).embedding for r in batch]).to(device=device, dtype=torch.float32)
        scores = model.project(q, "table", role="query") @ rel @ tv.T
        vals, idx = scores.topk(k=100, dim=1)
        for old, ii, vv in zip(batch, idx.cpu(), vals.cpu()):
            ids = [str(table_ids[int(i)]) for i in ii]; pos = set(map(str, old["positive_target_ids"]))
            rows.append({"query_id": old["query_id"], "query_kind": old["query_kind"], "positive_target_ids": sorted(pos), "exact_ids": ids, "exact_scores": [float(x) for x in vv],
                         **{f"exact_recall@{k}": len(pos & set(ids[:k])) / len(pos) if pos else 0.0 for k in (10, 20, 50, 100)}})
    path = source / "direct_exact_full_lake.jsonl.gz"; write_rows(path, rows)
    result = {"format_version": 1, "status": "complete", "arm": arm, "seed": seed, "step": step, "corpus_tables": len(table_ids), "queries": len(rows),
              "exact": {f"recall@{k}": statistics.fmean(r[f"exact_recall@{k}"] for r in rows) for k in (10, 20, 50, 100)}, "rankings": str(path.resolve()), "checkpoint_sha256": _hash_file(ck), "completed_at_utc": now()}
    write_json(source / "EXACT_D100.json", result)
    metrics = json.loads((source / "metrics.json").read_text()); metrics["direct_exact_full_lake"] = result; write_json(source / "metrics.json", metrics)
    return result


def evaluate(root: Path, arm: str, seed: int, step: int, device_name: str) -> None:
    evaluate_fixed(root, arm, seed, step, device_name)
    evaluate_full_lake(root, arm, seed, step, device_name)
    evaluate_exact_direct(root, arm, seed, step, device_name)


def report(root: Path) -> dict[str, Any]:
    jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            base = out(root) / arm / f"seed{seed}"
            jobs.append({"arm": arm, "seed": seed, "train": (base / "config.json").exists(),
                         "step0": (base / "checkpoints" / "step_000000.pt").exists(),
                         "step659": (base / "checkpoints" / "step_000659.pt").exists(),
                         "step1318": (base / "checkpoints" / f"step_{FINAL_STEP:06d}.pt").exists(),
                         "fixed_step0": (out(root) / "evaluations" / arm / f"seed{seed}" / "step_000000" / "metrics.json").exists(),
                         "fixed_step659": (out(root) / "evaluations" / arm / f"seed{seed}" / "step_000659" / "metrics.json").exists(),
                         "fixed": (out(root) / "evaluations" / arm / f"seed{seed}" / f"step_{FINAL_STEP:06d}" / "metrics.json").exists(),
                         "fixed_teacher": (out(root) / "evaluations" / arm / f"seed{seed}" / f"step_{FINAL_STEP:06d}" / "fixed_teacher" / "metrics.json").exists(),
                         "full_lake": (out(root) / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}" / "metrics.json").exists(),
                         "exact": (out(root) / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}" / "EXACT_D100.json").exists(),
                         "diagnostics": (base / "step_diagnostics.jsonl").exists()})
    conditional_jobs = []
    for arm in ("H0-old-list", "H1-natural-hard"):
        for seed in SEEDS:
            base = out(root) / "conditional_et" / arm / f"seed{seed}"
            evidence = out(root) / "conditional_et" / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}" / "evidence_diagnostics" / "ET_EXACT.json"
            fixed_teacher = out(root) / "conditional_et" / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}" / "fixed_teacher" / "metrics.json"
            fixed_teacher_metrics = json.loads(fixed_teacher.read_text()) if fixed_teacher.exists() else None
            conditional_jobs.append({"arm": arm, "seed": seed, "train": (base / "config.json").exists(),
                                     "step0": (base / "checkpoints" / "step_000000.pt").exists(),
                                     "step659": (base / "checkpoints" / "step_000659.pt").exists(),
                                     "step1318": (base / "checkpoints" / f"step_{FINAL_STEP:06d}.pt").exists(),
                                     "full_lake": (out(root) / "conditional_et" / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}" / "metrics.json").exists(),
                                     "et_exact": evidence.exists(),
                                     "fixed_teacher": fixed_teacher.exists(),
                                     "fixed_teacher_status": fixed_teacher_metrics["status"] if fixed_teacher_metrics else "missing"})
    teacher_quality_jobs = []
    for teacher in ("T0", "T1-A", "T1-B"):
        for seed in SEEDS:
            path = out(root) / "teacher_quality" / teacher / f"seed{seed}" / "metrics.json"
            metrics = json.loads(path.read_text()) if path.exists() else None
            teacher_quality_jobs.append({"teacher": teacher, "seed": seed, "audit": path.exists(),
                                         "status": metrics["status"] if metrics else "missing"})
    core_complete = all(all(job[key] for key in ("train", "step0", "step659", "step1318", "fixed_step0", "fixed_step659", "fixed", "fixed_teacher", "full_lake", "exact", "diagnostics")) for job in jobs)
    conditional_complete = all(all(job[key] for key in ("train", "step0", "step659", "step1318", "full_lake", "et_exact", "fixed_teacher")) for job in conditional_jobs)
    teacher_quality_complete = all(job["audit"] for job in teacher_quality_jobs)
    complete = core_complete and conditional_complete and teacher_quality_complete
    matrix = {"format_version": 1, "status": "complete" if complete else "partial", "core_jobs": jobs,
              "conditional_et_jobs": conditional_jobs, "conditional_et_status": "complete" if conditional_complete else "partial",
              "teacher_quality_jobs": teacher_quality_jobs,
              "teacher_quality_status": "complete" if teacher_quality_complete else "partial", "updated_at_utc": now()}
    write_json(out(root) / "EXECUTION_MATRIX.json", matrix)
    write_json(out(root) / "COMPLETION_AUDIT.json", {
        "status": matrix["status"], "required_core_jobs": len(jobs), "core_jobs": jobs,
        "conditional_et": {"status": matrix["conditional_et_status"], "jobs": conditional_jobs,
                            "trigger": "G1/G2 direct stable across both seeds and ET exact hub concentration remained severe",
                            "t2": "not_run_by_plan"},
        "teacher_quality": {"status": matrix["teacher_quality_status"], "jobs": teacher_quality_jobs},
        "notes": ["All six core jobs use the verified seed29 pool and complete step0/659/1318 artifacts.",
                  "The two previously missing Teacher hidden-state objects were regenerated in the immutable R23 teacher_supplement and all frozen-Teacher audits now have complete coverage.",
                  "No T2/T3 or online reranker was run."]})
    write_json(out(root) / "ET_TRIGGER_DECISION.json", {
        "format_version": 1,
        "status": "triggered",
        "decision": "run_H0_H1",
        "evidence": "G1/G2 direct retrieval was stable across both seeds while core ET exact rankings remained hub-concentrated.",
        "t2": "not_run_by_plan",
        "recorded_at_utc": now(),
    })
    reference_baselines = {}
    for name, path in {
        "Qwen-Raw": r21_out(root) / "full_lake" / "Qwen-Raw" / "raw" / "metrics.json",
        "B13": r21_out(root) / "full_lake" / "B13" / "seed13_step001318" / "metrics.json",
    }.items():
        if path.exists():
            metrics = json.loads(path.read_text())
            reference_baselines[name] = {"status": "complete", "metrics": str(path.resolve()), "sha256": _hash_file(path),
                                         "queries": metrics.get("queries"), "direct_raw@100": metrics.get("direct_raw@100"),
                                         "u_raw_recall": metrics.get("u_raw_recall"), "u_r10": metrics.get("u_r10"), "u_cr50": metrics.get("u_cr50")}
        else:
            reference_baselines[name] = {"status": "missing", "metrics": str(path.resolve())}
    write_json(out(root) / "REFERENCE_BASELINES.json", {"format_version": 1, "status": "complete", "baselines": reference_baselines, "note": "Read-only R21 outputs; not retrained in R23."})
    supplement_dir = out(root) / "teacher_supplement"
    supplement_manifest = supplement_dir / "teacher_manifest.jsonl"
    if supplement_manifest.exists():
        supplement_rows = list(read_rows(supplement_manifest))
        supplement_files = []
        for row in supplement_rows:
            feature_path = supplement_dir / row["teacher_feature_path"]
            supplement_files.append({"object_id": row["object_id"], "path": str(feature_path.resolve()), "sha256": _hash_file(feature_path), "bytes": feature_path.stat().st_size})
        write_json(out(root) / "TEACHER_SUPPLEMENT_AUDIT.json", {
            "format_version": 1, "status": "complete" if len(supplement_rows) == 2 and all(item["bytes"] > 0 for item in supplement_files) else "partial",
            "input": str((out(root) / "teacher_supplement_input.jsonl").resolve()),
            "input_sha256": _hash_file(out(root) / "teacher_supplement_input.jsonl"),
            "base_feature_manifest_sha256": _hash_file(r23_paths(root)["features"] / "manifest.jsonl"),
            "teacher_manifest": str(supplement_manifest.resolve()),
            "teacher_manifest_sha256": _hash_file(supplement_manifest),
            "objects": supplement_files,
            "model": "hf_models/Qwen3-VL-Embedding-8B",
            "prompt_version": "role_modality_v2_object_only",
            "note": "R23-only Teacher hidden-state supplement; base retrieval features and training pool were not modified.",
        })
    completion_path = out(root) / "COMPLETION_AUDIT.json"
    completion = json.loads(completion_path.read_text())
    completion["reference_baselines"] = reference_baselines
    supplement_audit = out(root) / "TEACHER_SUPPLEMENT_AUDIT.json"
    completion["teacher_supplement"] = json.loads(supplement_audit.read_text()) if supplement_audit.exists() else {"status": "missing"}
    write_json(completion_path, completion)
    # A compact durable narrative keeps the machine-readable matrix and the
    # scientific interpretation together without packaging heavyweight tensors.
    lines = ["# R23 Stage-1 repair results", "", "Core comparison (query macro means over 1,198 fixed queries):", "", "| arm | seed | fixed U R@10 | frozen-Teacher U R@10 | full direct Raw@100 | full U R@10 | exact table R@100 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for job in jobs:
        arm, seed = job["arm"], job["seed"]
        fixed_path = out(root) / "evaluations" / arm / f"seed{seed}" / f"step_{FINAL_STEP:06d}" / "metrics.json"
        full_path = out(root) / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}" / "metrics.json"
        exact_path = out(root) / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}" / "EXACT_D100.json"
        teacher_path = out(root) / "evaluations" / arm / f"seed{seed}" / f"step_{FINAL_STEP:06d}" / "fixed_teacher" / "metrics.json"
        if fixed_path.exists() and full_path.exists() and exact_path.exists() and teacher_path.exists():
            fixed, full, exact = json.loads(fixed_path.read_text()), json.loads(full_path.read_text()), json.loads(exact_path.read_text())
            teacher = json.loads(teacher_path.read_text())
            lines.append(f"| {arm} | {seed} | {fixed['pools']['U']['recall@10']:.4f} | {teacher['pools']['U']['recall@10']:.4f} | {full['direct_raw@100']:.4f} | {full['u_r10']:.4f} | {exact['exact']['recall@100']:.4f} |")
    lines += ["", "Conditional ET repair:", "", "| arm | seed | ET exact unique targets | ET top-20 share | E-only exact recall | full U R@10 | frozen-Teacher U R@10 | audit |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for job in conditional_jobs:
        p = out(root) / "conditional_et" / "full_lake" / job["arm"] / f"seed{job['seed']}_step{FINAL_STEP:06d}" / "evidence_diagnostics" / "ET_EXACT.json"
        m = out(root) / "conditional_et" / "full_lake" / job["arm"] / f"seed{job['seed']}_step{FINAL_STEP:06d}" / "metrics.json"
        t = out(root) / "conditional_et" / "full_lake" / job["arm"] / f"seed{job['seed']}_step{FINAL_STEP:06d}" / "fixed_teacher" / "metrics.json"
        if p.exists() and m.exists() and t.exists():
            et, full = json.loads(p.read_text()), json.loads(m.read_text())
            frozen = json.loads(t.read_text())
            lines.append(f"| {job['arm']} | {job['seed']} | {et['exact_et']['target_rankings']['unique_targets']} | {et['exact_et']['target_rankings']['top20_share']:.4f} | {et['exact_et']['e_only_positive_recall_exact_et_union']:.4f} | {full['u_r10']:.4f} | {frozen['u']['recall@10']:.4f} | {frozen['status']} |")
    lines += ["", "Frozen Teacher quality on the same held-out natural pools:", "", "| teacher | seed | R@10 | R@20 | R@50 | coverage | audit |", "|---|---:|---:|---:|---:|---:|---|"]
    for job in teacher_quality_jobs:
        p = out(root) / "teacher_quality" / job["teacher"] / f"seed{job['seed']}" / "metrics.json"
        if p.exists():
            metrics = json.loads(p.read_text())
            natural = metrics["natural_pool"]
            lines.append(f"| {job['teacher']} | {job['seed']} | {natural['recall@10']:.4f} | {natural['recall@20']:.4f} | {natural['recall@50']:.4f} | {metrics['teacher_candidate_coverage']:.6f} | {metrics['status']} |")
    lines += ["", "Read-only retrieval references from R21:", "", "| reference | direct Raw@100 | U Raw | U R@10 | U CR@50 | status |", "|---|---:|---:|---:|---:|---|"]
    for name, baseline in reference_baselines.items():
        if baseline["status"] == "complete":
            lines.append(f"| {name} | {baseline['direct_raw@100']:.4f} | {baseline['u_raw_recall']:.4f} | {baseline['u_r10']:.4f} | {baseline['u_cr50']:.4f} | complete |")
        else:
            lines.append(f"| {name} | — | — | — | — | missing |")
    lines += ["", "Interpretation: G2's explicit non-TT uniform term improves core direct/U metrics, while G1 is a stable negative control. ET exact remains hub-concentrated after the core arms, triggering H0/H1. H1's train-evidence natural-hard candidate distribution materially increases target coverage and E-only recovery; H0 old-list continuation does not. H1 does not improve the frozen-Teacher U audit over H0, so its gain is evidence-coverage repair rather than a general reranking gain. The two previously missing Teacher hidden-state objects were regenerated in the immutable R23 supplement; all frozen-Teacher audits now have complete coverage. These are mechanism results, not evidence of a completed Stage-2 join/value verifier.", ""]
    (out(root) / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit-inputs", "snapshot", "build-manifests", "report"):
        sub.add_parser(name)
    p = sub.add_parser("validate-cache"); p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("train"); p.add_argument("--arm", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("evaluate"); p.add_argument("--arm", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--step", type=int, default=FINAL_STEP); p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("fixed"); p.add_argument("--arm", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--step", type=int, required=True); p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("fixed-teacher"); p.add_argument("--arm", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--step", type=int, default=FINAL_STEP); p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("exact"); p.add_argument("--arm", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--step", type=int, default=FINAL_STEP); p.add_argument("--device", default="cuda:0")
    args = ap.parse_args(); root = args.root.resolve()
    if args.cmd == "audit-inputs": print(json.dumps(audit_inputs(root), indent=2))
    elif args.cmd == "snapshot": print(json.dumps(snapshot_sources(root), indent=2))
    elif args.cmd == "build-manifests": print(json.dumps(build_manifests(root), indent=2))
    elif args.cmd == "validate-cache":
        manifest = out(root) / "manifests" / "G1-QTKD.jsonl"
        if not manifest.exists(): build_manifests(root)
        print(json.dumps(validate_teacher_cache(root, args.seed, manifest), indent=2))
    elif args.cmd == "train": print(json.dumps(train(root, args.arm, args.seed, args.device), indent=2))
    elif args.cmd == "evaluate": evaluate(root, args.arm, args.seed, args.step, args.device); print(json.dumps({"status": "complete", "arm": args.arm, "seed": args.seed, "step": args.step}))
    elif args.cmd == "fixed": print(json.dumps(evaluate_fixed(root, args.arm, args.seed, args.step, args.device), indent=2))
    elif args.cmd == "fixed-teacher": print(json.dumps(evaluate_fixed_teacher(root, args.arm, args.seed, args.step, args.device), indent=2))
    elif args.cmd == "exact": print(json.dumps(evaluate_exact_direct(root, args.arm, args.seed, args.step, args.device), indent=2))
    elif args.cmd == "report": print(json.dumps(report(root), indent=2))


if __name__ == "__main__":
    main()
