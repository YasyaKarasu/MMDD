#!/usr/bin/env python3
"""R24 Stage-1 experiments.

This runner is deliberately small and auditable.  It reuses the frozen R23
inputs, trains the missing SUP+Uniform control (N-U), repairs the H1
train-positive closure (H1-PC), and runs the four common-graph target/path
arms.  All generated files live under ``work/stage1_optimization_r24_*`` and
carry input/checkpoint hashes.  The conditional Teacher-feedback and C1/C2
extensions in the R24 contract are not started by this module.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import EdgeExample, TargetExample, load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator, distillation_kl, listwise_cross_entropy
from mmdd_stage1.retrieval import StudentANNIndices, build_indices, load_corpus_ids, retrieve_zero_one_hop_detailed_many
from mmdd_stage1.scoring import ListScores, TargetScores, score_edge_batch, score_target_batch
from mmdd_stage1.training import _student_edge_losses, _student_path_losses, student_gradient_norms

from run_stage1_r19 import _score_id_pairs, load_r19_checkpoint
from run_stage1_r21 import _load_teacher, _score_target_ids, out as r21_out, paths as r21_paths, read_rows, write_rows
from run_stage1_r22 import out as r22_out, paths as r22_paths
from run_stage1_r22_f0 import _build_fresh_student, _optimizer
from run_stage1_r23 import (
    FINAL_STEP as R23_FINAL_STEP,
    _make_teacher_scores,
    _teacher_cache_path,
    _teacher_map,
    r23_paths,
    teacher_feature_paths,
)
from run_stage1_r23_et import _freeze_except_et

ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = "stage1_optimization_r24_20260913"
SEEDS = (13, 29)
BATCH = 64
EDGE_LISTS = 42143
EDGE_UPDATES = math.ceil(EDGE_LISTS / BATCH) * 2
TARGET_BATCHES = math.ceil(11390 / BATCH)
UNIFORM_WEIGHT = 0.3
ANCHOR_WEIGHT = 0.1
KD_WEIGHT = 0.3
TEMPERATURE = 1.0
PATH_ARMS = ("P-Edge", "P-Split-SUP", "P-Split-KD", "P-LSE-KD")
RAM_TEACHER_ROOT = Path("/dev/shm/mmdd_r24_teacher_backfill")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def out(root: Path) -> Path:
    return root / "work" / OUT_NAME


def _sha(path: Path) -> str:
    return checkpoint_fingerprint(path)


def audit_r23(root: Path) -> dict[str, Any]:
    """Verify the immutable R23 inputs before creating any R24 artifact."""
    checks = {
        "g2_seed13": _r23_root(root) / "G2-QTKD-U/seed13/checkpoints/step_001318.pt",
        "g2_seed29": _r23_root(root) / "G2-QTKD-U/seed29/checkpoints/step_001318.pt",
        "g_manifest": _r23_root(root) / "manifests/G0-SUP.jsonl",
        "features_manifest": r23_paths(root)["features"] / "manifest.jsonl",
        "pca": r23_paths(root)["pca"],
        "target_train": root / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/target_lists.train_fit.jsonl",
        "path_hard": root / "work/stage1_optimization_r12_20260908/taskC_training/c2_candidates_seed13/path_hard.jsonl",
    }
    missing = [str(p) for p in checks.values() if not p.exists()]
    payload = {
        "format_version": 1,
        "status": "pass" if not missing else "blocked",
        "inputs": {
            name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}
            for name, path in checks.items() if path.exists()
        },
        "missing": missing,
        "r23_g_manifest_expected_sha256": "77d3fca543d1a288640f1740258ed922f45d107afbf913ae3648c0e27b135f46",
        "recorded_at_utc": now(),
    }
    out(root).mkdir(parents=True, exist_ok=True)
    write_json(out(root) / "P0_INPUT_AUDIT.json", payload)
    if missing:
        raise RuntimeError("R24 P0 input audit failed: " + ", ".join(missing))
    return payload


def _r23_root(root: Path) -> Path:
    return root / "work" / "stage1_optimization_r23_20260913"


def build_positive_registry(root: Path) -> dict[str, Any]:
    """Build train-only (evidence, directed relation) positive closure."""
    source = _r23_root(root) / "conditional_et" / "manifests" / "H1-natural-hard_seed13.jsonl"
    rows = [dict(r) for r in read_rows(source)]
    registry: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        relation = f"{row['source_type']}->{row['destination_type']}"
        registry[(str(row["query_id"]), relation)].update(str(x) for x in row.get("positive_ids", [row["positive_id"]]))
    # Build both seeds from their own train manifest, but require the same
    # closure rule and preserve each candidate list byte-for-byte.
    audits = {}
    for seed in SEEDS:
        source = _r23_root(root) / "conditional_et" / "manifests" / f"H1-natural-hard_seed{seed}.jsonl"
        destination = out(root) / "manifests" / f"H1-PC_seed{seed}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        repaired = []
        affected_lists = affected_positions = 0
        for row in read_rows(source):
            row = dict(row)
            relation = f"{row['source_type']}->{row['destination_type']}"
            key = (str(row["query_id"]), relation)
            known = set(registry[key])
            # Include all source-manifest positives for this seed as well.
            known.update(str(x) for x in row.get("positive_ids", [row["positive_id"]]))
            candidates = [str(x) for x in row["candidate_ids"]]
            original = set(str(x) for x in row.get("positive_ids", [row["positive_id"]]))
            closure = [candidate for candidate in candidates if candidate in known]
            conflicts = set(closure) - original
            affected_lists += int(bool(conflicts))
            affected_positions += len(conflicts)
            if not closure:
                raise RuntimeError(f"No positive remains in H1-PC list {row['query_id']}")
            row["positive_ids"] = closure
            row["positive_id"] = closure[0]
            row["confirmed_labels"] = [1 if c in set(closure) else None for c in candidates]
            row["candidate_source"] = "R23-H1-natural-hard-membership; train-positive-closure"
            repaired.append(row)
        write_rows(destination, repaired)
        audits[str(seed)] = {
            "source": str(source.resolve()), "output": str(destination.resolve()),
            "rows": len(repaired), "affected_lists": affected_lists,
            "known_positive_as_negative_after": 0,
            "candidate_membership_order_preserved": True,
            "sha256": _sha(destination),
        }
    result = {"format_version": 1, "status": "complete", "relation_key": "(source_object_id,directed_relation)", "seeds": audits, "created_at_utc": now()}
    write_json(out(root) / "POSITIVE_REGISTRY.json", result)
    return result


def _save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, job: Path, arm: str, seed: int, step: int, stage: str, *, include_optimizer: bool = True) -> str:
    path = job / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "model_kind": "student", "completed_stage": stage,
        "arm": arm, "seed": seed, "step": step, "config": model.config(),
        "trainable_parameters": [n for n, p in model.named_parameters() if p.requires_grad],
        "projection_references": {"origin": getattr(model, "projection_reference_origin", "r24")},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return _sha(path)


def _edge_training(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    """Train N-U from fresh PCA or H1-PC with only ET relations open."""
    device = torch.device(device_name)
    if arm == "N-U":
        manifest = _r23_root(root) / "manifests" / "G0-SUP.jsonl"
        model = _build_fresh_student(root, device).train()
        optimizer = _optimizer(model)
        teacher = None
    elif arm == "H1-PC":
        manifest = out(root) / "manifests" / f"H1-PC_seed{seed}.jsonl"
        parent = _r23_root(root) / "G2-QTKD-U" / f"seed{seed}" / "checkpoints" / "step_001318.pt"
        model = load_student(parent, device).train()
        _freeze_except_et(model)
        optimizer = torch.optim.AdamW([
            {"params": model.relations["text_to_table"], "lr": 1e-5},
            {"params": model.relations["image_to_table"], "lr": 1e-5},
        ], weight_decay=0.01)
        teacher = None
    else:
        raise ValueError(arm)
    examples = load_edge_examples(manifest, split="train")
    if arm == "N-U" and len(examples) != EDGE_LISTS:
        raise RuntimeError(f"N-U expected {EDGE_LISTS} rows, got {len(examples)}")
    if arm == "H1-PC":
        examples = (examples * math.ceil(EDGE_LISTS / len(examples)))[:EDGE_LISTS]
    job = out(root) / arm / f"seed{seed}"
    final = job / "checkpoints" / f"step_{EDGE_UPDATES:06d}.pt"
    if final.exists() and (job / "config.json").exists():
        return json.loads((job / "config.json").read_text())
    job.mkdir(parents=True, exist_ok=True)
    init_hash = _save_model(model, optimizer, job, arm, seed, 0, "r24-common-initial")
    rng = random.Random(240913 + seed)
    order = list(range(len(examples)))
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=24000)
    history, diagnostics = [], []
    step = 0
    started = time.monotonic()
    for epoch in range(2):
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH):
            batch = [examples[i] for i in order[start:start+BATCH]]
            optimizer.zero_grad(set_to_none=True)
            scores = score_edge_batch(model, batch, store, device)
            terms = _student_edge_losses(model, batch, scores, None, scores, None,
                ranking_weight=1.0, temperature=1.0, distillation_weight=0.0,
                edge_bce_weight=0.0, anchor_weight=ANCHOR_WEIGHT,
                anchor_weight_evidence=ANCHOR_WEIGHT)
            uniform = scores.logits.new_zeros(())
            n_non = 0
            if arm == "N-U":
                mask = torch.tensor([not (e.source_type == "table" and e.destination_type == "table") for e in batch], device=device)
                n_non = int(mask.sum())
                if n_non:
                    # Uniform logits are zero; the mask and candidate lengths
                    # still ensure padding never contributes.
                    uniform = distillation_kl(scores.logits[mask], torch.zeros_like(scores.logits[mask]), scores.candidate_mask[mask], 1.0)
                    terms["loss"] = terms["loss"] + UNIFORM_WEIGHT * n_non / len(batch) * uniform
            terms["loss"].backward()
            grads = student_gradient_norms(model)
            optimizer.step(); step += 1
            value = float(terms["loss"].detach().cpu()); losses.append(value)
            if step <= 2 or step % 100 == 0:
                diagnostics.append({"step": step, "epoch": epoch + 1, "loss": value,
                    "supervised_loss": float(terms["supervised_loss"].detach().cpu()),
                    "uniform_loss": float(uniform.detach().cpu()), "n_non_tt": n_non,
                    "gradient": grads, "elapsed_seconds": time.monotonic() - started})
                print(json.dumps({"arm": arm, "seed": seed, "step": step, "loss": value}), flush=True)
        ck_hash = _save_model(model, optimizer, job, arm, seed, step, "r24-edge")
        history.append({"epoch": epoch + 1, "step": step, "loss": statistics.fmean(losses), "checkpoint_sha256": ck_hash})
        write_rows(job / "train_history.jsonl", history)
        write_rows(job / "step_diagnostics.jsonl", diagnostics)
    cfg = {"format_version": 1, "status": "complete", "arm": arm, "seed": seed,
        "stage": "edge", "manifest_sha256": _sha(manifest), "initial_checkpoint_sha256": init_hash,
        "uniform_weight": UNIFORM_WEIGHT if arm == "N-U" else 0.0, "teacher": None,
        "optimizer": {"relation_lr": 1e-5, "projection_lr": 1e-6, "weight_decay": 0.01},
        "anchor_weight": ANCHOR_WEIGHT, "batch_size": BATCH, "updates": step,
        "history": history, "diagnostics": str((job / "step_diagnostics.jsonl").resolve()),
        "device": device_name, "completed_at_utc": now()}
    write_json(job / "config.json", cfg)
    return cfg


def build_path_pool(root: Path, seed: int) -> dict[str, Any]:
    """Freeze a common train target/path graph from the legal train path input."""
    source = root / "work/stage1_optimization_r12_20260908/taskC_training/c2_candidates_seed13/path_hard.jsonl"
    # The R12 path-hard file is a train-only retrieval graph.  Keep it intact
    # and record that it is an existing graph, rather than pretending it was
    # generated online from a dev label.
    rows = []
    truncated_paths = 0
    for raw in read_rows(source):
        row = dict(raw)
        candidates = []
        for candidate in row["candidates"]:
            candidate = dict(candidate)
            evidence = [str(x) for x in candidate.get("evidence_ids", [])]
            if len(evidence) > 8:
                truncated_paths += len(evidence) - 8
                candidate["evidence_ids"] = evidence[:8]
            candidates.append(candidate)
        row["candidates"] = candidates
        rows.append(row)
    destination = out(root) / "path_pool" / f"common_seed{seed}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_rows(destination, rows)
    result = {"format_version": 1, "status": "complete", "seed": seed,
        "source": str(source.resolve()), "source_sha256": _sha(source),
        "output": str(destination.resolve()), "output_sha256": _sha(destination),
        "rows": len(rows), "candidate_graph": "train-only R12 path_hard reused as frozen common graph",
        "max_evidence_paths_per_target": 8, "truncated_evidence_path_count": truncated_paths,
        "positive_closure": "source train target lists",
        "created_at_utc": now()}
    write_json(destination.with_suffix(".manifest.json"), result)
    return result


def _r24_teacher_feature_paths(root: Path) -> list[Path]:
    """Return frozen Teacher shards plus any R24 backfill shards."""

    paths = list(teacher_feature_paths(root))
    for backfill_root in (out(root) / "teacher_backfill", RAM_TEACHER_ROOT):
        for shard in sorted(backfill_root.glob("gpu*/teacher_manifest.jsonl")):
            paths.append(shard.parent)
    return paths


def _reuse_h1_index(root: Path, seed: int, checkpoint: Path, index_dir: Path) -> dict[str, Any]:
    """Reuse the immutable G2 HNSW payload for H1-PC's frozen projections."""

    source = _r23_root(root) / "indexes" / "G2-QTKD-U" / f"seed{seed}" / "step_001318"
    source_manifest_path = source / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("student_checkpoint_sha256") != _sha(
        _r23_root(root) / "G2-QTKD-U" / f"seed{seed}" / "checkpoints" / "step_001318.pt"
    ):
        raise RuntimeError("R23 G2 index manifest does not match its recorded checkpoint")
    source_payload = torch.load(
        _r23_root(root) / "G2-QTKD-U" / f"seed{seed}" / "checkpoints" / "step_001318.pt",
        map_location="cpu",
        weights_only=True,
    )
    target_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_state = source_payload["state_dict"]
    target_state = target_payload["state_dict"]
    projection_keys = [
        key for key in source_state
        if key in target_state and ("projection" in key or "project" in key)
    ]
    if not projection_keys or not all(torch.equal(source_state[key], target_state[key]) for key in projection_keys):
        raise RuntimeError("H1-PC projection parameters differ from the G2 index source")
    corpus_sha = _sha(r21_paths(root)["corpus"])
    if source_manifest.get("corpus_sha256") != corpus_sha:
        raise RuntimeError("R23 G2 index corpus hash differs from the frozen corpus")
    index_dir.mkdir(parents=True, exist_ok=True)
    for name in ("table.hnsw", "table_ids.json", "text.hnsw", "text_ids.json", "image.hnsw", "image_ids.json"):
        link = index_dir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source / name)
    manifest = dict(source_manifest)
    manifest["student_checkpoint_sha256"] = _sha(checkpoint)
    manifest["index_reuse"] = {
        "source_index": str(source.resolve()),
        "source_checkpoint_sha256": source_manifest["student_checkpoint_sha256"],
        "reason": "H1-PC opens only ET relations; shared destination projections and frozen query projections match G2",
        "projection_keys_verified": projection_keys,
    }
    write_json(index_dir / "manifest.json", manifest)
    return manifest


def prepare_teacher_backfill(root: Path) -> dict[str, Any]:
    """Materialize two deterministic input shards for missing text Teacher states."""

    pool = out(root) / "path_pool" / "common_seed13.jsonl"
    if not pool.exists():
        build_path_pool(root, 13)
    path_ids: set[str] = set()
    for row in read_rows(pool):
        path_ids.add(str(row["query_id"]))
        for candidate in row["candidates"]:
            path_ids.add(str(candidate["target_id"]))
            path_ids.update(str(value) for value in candidate.get("evidence_ids", []))
    frozen_teacher_ids: set[str] = set()
    frozen_shards = [r23_paths(root)["features"], *teacher_feature_paths(root)]
    for shard in frozen_shards:
        manifest = shard / "teacher_manifest.jsonl"
        if manifest.exists():
            frozen_teacher_ids.update(str(row["object_id"]) for row in read_rows(manifest))
    missing = sorted(path_ids - frozen_teacher_ids)
    source = root / "work/stage1_optimization_r10_20260907/stage1_data/stage1_objects.jsonl"
    records = {}
    for row in read_rows(source):
        object_id = str(row["object_id"])
        if object_id in missing:
            records[object_id] = dict(row)
    absent = sorted(set(missing) - set(records))
    if absent:
        raise RuntimeError(f"Teacher backfill source is missing {len(absent)} objects")
    input_root = out(root) / "teacher_backfill_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    shard_rows = [[], []]
    for index, object_id in enumerate(missing):
        shard_rows[index % 2].append(records[object_id])
    outputs = []
    for index, rows in enumerate(shard_rows):
        destination = input_root / f"gpu{index}.jsonl"
        write_rows(destination, rows)
        outputs.append({"path": str(destination.resolve()), "sha256": _sha(destination), "rows": len(rows)})
    result = {
        "format_version": 1,
        "status": "complete",
        "source": str(source.resolve()),
        "source_sha256": _sha(source),
        "common_path_pool_sha256": _sha(pool),
        "path_object_count": len(path_ids),
        "frozen_teacher_object_count": len(frozen_teacher_ids),
        "missing_object_count": len(missing),
        "missing_object_types": {"text": len(missing)},
        "inputs": outputs,
        "teacher_output_root": str(RAM_TEACHER_ROOT.resolve()),
        "created_at_utc": now(),
    }
    write_json(out(root) / "TEACHER_BACKFILL_INPUT.json", result)
    return result


def _teacher_target_cache(root: Path, seed: int, examples: list[TargetExample], device_name: str) -> dict[str, Any]:
    destination = out(root) / "path_pool" / f"teacher_target_seed{seed}.jsonl.gz"
    blocked_path = out(root) / "path_pool" / f"teacher_target_seed{seed}.blocked.json"
    if blocked_path.exists() and not destination.exists():
        # A previous coverage audit may have been blocked before an R24
        # backfill shard was generated.  Re-audit whenever new shards exist;
        # only reuse the old block if no R24 Teacher source has changed.
        backfill_manifests = list((out(root) / "teacher_backfill").glob("gpu*/teacher_manifest.jsonl"))
        backfill_manifests += list(RAM_TEACHER_ROOT.glob("gpu*/teacher_manifest.jsonl"))
        if not backfill_manifests:
            return {"status": "blocked", "audit": str(blocked_path.resolve()), **json.loads(blocked_path.read_text())}
    if destination.exists():
        blocked_path.unlink(missing_ok=True)
        return {"path": str(destination.resolve()), "sha256": _sha(destination), "rows": sum(1 for _ in read_rows(destination))}
    device = torch.device(device_name)
    teacher_path = r22_out(root) / "fresh_lineage" / "T1-B" / f"seed{seed}" / "checkpoints" / "step_010536.pt"
    # R24's fixed transfer teacher is T1-B (R22 fresh lineage), not the R21
    # D2 helper returned by ``run_stage1_r21._load_teacher``.
    _teacher_arm, saved_seed, _teacher_step, teacher, _teacher_payload = load_r19_checkpoint(teacher_path, device)
    if saved_seed != seed:
        raise RuntimeError(f"T1-B seed mismatch: expected {seed}, got {saved_seed}")
    teacher.eval()
    # Hidden states are multi-megabyte tensors.  Do not populate a giant
    # object cache merely to audit manifest coverage: that can exceed host
    # memory before scoring starts.  The bounded cache below is sufficient
    # for the subsequent RAM/disk-backed score pass.
    store = FeatureStore.from_path(
        r23_paths(root)["features"],
        cache_size=512,
        cache_bytes=8 * 1024**3,
        teacher_paths=_r24_teacher_feature_paths(root),
    )
    teacher_index = getattr(store, "_teacher_index", {})
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    rows = []
    missing_objects: set[str] = set()
    # Fail closed on incomplete hidden-state coverage.  The R24 contract
    # requires Teacher scores on the exact new target/path list; ranking a
    # partial list would silently change the experiment.
    for ex in examples:
        object_ids = [ex.query_id]
        for candidate in ex.candidates:
            object_ids.append(candidate.target_id)
            object_ids.extend(candidate.evidence_ids)
        for object_id in object_ids:
            object_id = str(object_id)
            teacher_feature = teacher_index.get(object_id)
            if teacher_feature is None or not teacher_feature.is_file():
                missing_objects.add(str(object_id))
    if missing_objects:
        audit = out(root) / "path_pool" / f"teacher_target_seed{seed}.blocked.json"
        payload = {"format_version": 1, "status": "blocked", "seed": seed,
            "reason": "T1-B hidden-state cache does not cover all common path graph objects",
            "missing_object_count": len(missing_objects), "missing_object_sample": sorted(missing_objects)[:20],
            "teacher_checkpoint": str(teacher_path.resolve()), "created_at_utc": now()}
        write_json(audit, payload)
        return {"status": "blocked", "audit": str(audit.resolve()), **payload}
    with torch.inference_mode():
        for start in range(0, len(examples), BATCH):
            batch = examples[start:start+BATCH]
            scored = score_target_batch(teacher, batch, store, device, aggregator)
            for ex, d, e in zip(batch, scored.direct.logits.cpu(), scored.evidence.logits.cpu()):
                n = len(ex.candidates)
                rows.append({"query_id": ex.query_id, "candidate_ids": [c.target_id for c in ex.candidates],
                    "direct_logits": [float(x) for x in d[:n]], "evidence_logits": [float(x) for x in e[:n]]})
            if (start // BATCH) % 25 == 0:
                print(json.dumps({"stage": "teacher_target_cache", "seed": seed, "rows": start + len(batch)}), flush=True)
    write_rows(destination, rows)
    blocked_path.unlink(missing_ok=True)
    return {"path": str(destination.resolve()), "sha256": _sha(destination), "rows": len(rows), "teacher_checkpoint_sha256": _sha(teacher_path)}


def _attach_teacher(examples: list[TargetExample], cache_path: Path) -> list[TargetExample]:
    records = {str(r["query_id"]): r for r in read_rows(cache_path)}
    result = []
    for ex in examples:
        r = records.get(ex.query_id)
        if r is None or list(r["candidate_ids"]) != [c.target_id for c in ex.candidates]:
            raise RuntimeError(f"Teacher target cache identity mismatch for {ex.query_id}")
        result.append(dataclasses.replace(ex, teacher_direct_logits=tuple(r["direct_logits"]), teacher_evidence_logits=tuple(r["evidence_logits"])))
    return result


def _unfreeze_all(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(True)
    if hasattr(model, "set_projection_frozen"):
        model.set_projection_frozen(False)


def _fused_loss(scores: TargetScores, *, positive_loss_mode: str = "sum_probability") -> torch.Tensor:
    # Evidence has an explicit valid-candidate mask.  F has D when no path and
    # logsumexp(D,E) otherwise; this avoids a synthetic zero-score fake path.
    d = scores.direct.logits
    e = scores.evidence.logits.masked_fill(~scores.evidence.candidate_mask, -torch.inf)
    f = torch.logsumexp(torch.stack((d, e), dim=-1), dim=-1)
    mask = scores.direct.candidate_mask
    return listwise_cross_entropy(f, scores.direct.positive_indices, mask, scores.direct.positive_mask, positive_loss_mode=positive_loss_mode)


def train_path(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    if arm not in PATH_ARMS:
        raise ValueError(arm)
    pool = out(root) / "path_pool" / f"common_seed{seed}.jsonl"
    if not pool.exists():
        build_path_pool(root, seed)
    examples = load_target_examples(pool, split="train")
    cache_info: dict[str, Any] | None = None
    if arm in {"P-Split-KD", "P-LSE-KD"}:
        cache_info = _teacher_target_cache(root, seed, examples, device_name)
        if cache_info.get("status") == "blocked":
            raise RuntimeError("P target Teacher cache is blocked; see " + cache_info["audit"])
        teacher_examples = _attach_teacher(examples, Path(cache_info["path"]))
    else:
        teacher_examples = examples
    parent = out(root) / "H1-PC" / f"seed{seed}" / "checkpoints" / f"step_{EDGE_UPDATES:06d}.pt"
    if not parent.exists():
        raise FileNotFoundError(parent)
    device = torch.device(device_name)
    job = out(root) / arm / f"seed{seed}"
    final = job / "checkpoints" / f"step_{TARGET_BATCHES:06d}.pt"
    if final.exists() and (job / "config.json").exists():
        return json.loads((job / "config.json").read_text())
    model = load_student(parent, device).train(); _unfreeze_all(model)
    optimizer = _optimizer(model)
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=50000)
    job.mkdir(parents=True, exist_ok=True)
    # Step 0 is exactly the immutable H1-PC parent; reference its hash rather
    # than duplicating another ~190 MB checkpoint.
    init_hash = _sha(parent)
    rng = random.Random(240913 + seed)
    order = list(range(len(examples))); rng.shuffle(order)
    history = []; started = time.monotonic()
    for step, start in enumerate(range(0, len(order), BATCH), 1):
        batch = [examples[i] for i in order[start:start+BATCH]]
        t_batch = [teacher_examples[i] for i in order[start:start+BATCH]]
        optimizer.zero_grad(set_to_none=True)
        scores = score_target_batch(model, batch, store, device, aggregator)
        teacher_scores = _target_scores_from_examples(t_batch, device)
        base = _student_path_losses(model, scores, teacher_scores if arm in {"P-Split-KD", "P-LSE-KD"} else None, None,
            temperature=TEMPERATURE, distillation_weight=KD_WEIGHT if arm in {"P-Split-KD", "P-LSE-KD"} else 0.0,
            anchor_weight=ANCHOR_WEIGHT, anchor_weight_evidence=ANCHOR_WEIGHT,
            distillation_rows=None, positive_loss_mode="sum_probability")
        if arm == "P-LSE-KD":
            sup = _fused_loss(scores)
            kd = distillation_kl(torch.logsumexp(torch.stack((scores.direct.logits, scores.evidence.logits.masked_fill(~scores.evidence.candidate_mask, -torch.inf)), -1), -1),
                torch.logsumexp(torch.stack((teacher_scores.direct.logits, teacher_scores.evidence.logits.masked_fill(~teacher_scores.evidence.candidate_mask, -torch.inf)), -1), -1), scores.direct.candidate_mask, TEMPERATURE)
            loss = sup + KD_WEIGHT * kd + base["weighted_anchor_loss"]
        elif arm == "P-Split-SUP":
            loss = 0.5 * (base["direct_supervised_loss"] + base["evidence_supervised_loss"]) + base["weighted_anchor_loss"]
        else:
            loss = base["loss"]
        loss.backward(); grads = student_gradient_norms(model); optimizer.step()
        row = {"step": step, "loss": float(loss.detach().cpu()), "direct_supervised_loss": float(base["direct_supervised_loss"].detach().cpu()),
            "evidence_supervised_loss": float(base["evidence_supervised_loss"].detach().cpu()), "distillation_loss": float(base["distillation_loss"].detach().cpu()),
            "gradient": grads, "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        if step <= 2 or step % 25 == 0 or step == TARGET_BATCHES:
            print(json.dumps({"arm": arm, "seed": seed, "step": step, "loss": row["loss"]}), flush=True)
    ck_hash = _save_model(model, optimizer, job, arm, seed, TARGET_BATCHES, "r24-path", include_optimizer=False)
    write_rows(job / "train_history.jsonl", history)
    cfg = {"format_version": 1, "status": "complete", "arm": arm, "seed": seed, "stage": "path",
        "parent_checkpoint_sha256": _sha(parent), "path_pool_sha256": _sha(pool), "teacher_target_cache": cache_info,
        "aggregation": aggregator.config(), "updates": TARGET_BATCHES, "initial_checkpoint_sha256": init_hash,
        "final_checkpoint_sha256": ck_hash, "history": history, "device": device_name, "completed_at_utc": now()}
    write_json(job / "config.json", cfg)
    return cfg


def _target_scores_from_examples(examples: list[TargetExample], device: torch.device) -> TargetScores:
    from mmdd_stage1.training import _target_teacher_scores
    return _target_teacher_scores(examples, device)


@torch.inference_mode()
def evaluate_fixed(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    ck = out(root) / arm / f"seed{seed}" / "checkpoints"
    candidates = sorted(ck.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = load_student(candidates, device).eval()
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=40000)
    records = []
    for row in read_rows(r23_paths(root)["candidate_pools"]):
        pos = set(map(str, row["positive_target_ids"]))
        pools = {"U": list(map(str, row["natural_candidate_ids"])), "D": list(map(str, row["ann_direct100_ids"])), "M": list(map(str, row["matched_direct_candidate_ids"]))}
        ids = list(dict.fromkeys(x for v in pools.values() for x in v)); scores = _score_target_ids(model, store, str(row["query_id"]), ids, device)
        rec = {"query_id": str(row["query_id"]), "query_kind": row.get("query_kind"), "positive_target_ids": sorted(pos), "pools": {}}
        for name, vals in pools.items():
            rank = sorted(vals, key=lambda x: (-scores[x], x))
            rec["pools"][name] = {"candidate_ids": vals, "ranking": rank, "raw_recall": len(pos & set(vals))/len(pos) if pos else 0.0, **{f"recall@{k}": len(pos & set(rank[:k]))/len(pos) if pos else 0.0 for k in (10,20,50)}}
        records.append(rec)
    dest = out(root) / "evaluations" / arm / f"seed{seed}"; dest.mkdir(parents=True, exist_ok=True)
    write_rows(dest / "fixed_U_M_D_rankings.jsonl.gz", records)
    metrics = {"format_version": 1, "status": "complete", "arm": arm, "seed": seed, "checkpoint_sha256": _sha(candidates), "queries": len(records),
        "pools": {name: {k: statistics.fmean(r["pools"][name][k] for r in records) for k in ("raw_recall", "recall@10", "recall@20", "recall@50")} for name in ("U","D","M")}, "rankings": str((dest / "fixed_U_M_D_rankings.jsonl.gz").resolve()), "completed_at_utc": now()}
    write_json(dest / "metrics.json", metrics); return metrics


@torch.inference_mode()
def evaluate_full_lake(root: Path, arm: str, seed: int, device_name: str, *, keep_index: bool = False) -> dict[str, Any]:
    """Evaluate direct/evidence admission on the frozen full lake.

    Indices are built one job at a time because a single HNSW table index is
    roughly a gigabyte on this corpus.  The manifest/hash is retained even
    when ``keep_index`` is false, allowing the run to be reproduced without
    exhausting the research volume.
    """
    ck_dir = out(root) / arm / f"seed{seed}" / "checkpoints"
    ck = sorted(ck_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
    dest = out(root) / "full_lake" / arm / f"seed{seed}"; dest.mkdir(parents=True, exist_ok=True)
    metrics_path = dest / "metrics.json"
    if metrics_path.exists() and (dest / "rankings.jsonl.gz").exists():
        return json.loads(metrics_path.read_text())
    device = torch.device(device_name)
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=120000)
    model = load_student(ck, device).eval()
    ids_by_type = load_corpus_ids(r21_paths(root)["corpus"], store)
    index_root = Path(os.environ.get("R24_INDEX_ROOT", str(out(root) / "indexes"))).expanduser()
    index_dir = index_root / arm / f"seed{seed}"; index_dir.mkdir(parents=True, exist_ok=True)
    if arm == "H1-PC":
        index_manifest = _reuse_h1_index(root, seed, ck, index_dir)
    else:
        index_manifest = build_indices(model, store, ids_by_type, index_dir, device=device, checkpoint_sha256=_sha(ck), corpus_sha256=_sha(r21_paths(root)["corpus"]), batch_size=4096)
    index_manifest = index_dir / "manifest.json"
    indices = StudentANNIndices(model, store, index_dir, device=device, checkpoint_sha256=_sha(ck), corpus_sha256=_sha(r21_paths(root)["corpus"]))
    pools = list(read_rows(r23_paths(root)["candidate_pools"])); rows = []
    detailed = retrieve_zero_one_hop_detailed_many([str(r["query_id"]) for r in pools], indices, k=100, direct_k=100, evidence_k=20, targets_per_evidence=20, query_batch_size=16)
    for pool, ret in zip(pools, detailed):
        pos = set(map(str, pool["positive_target_ids"])); direct = [str(x["target_id"]) for x in ret["direct"]]; evidence = [str(x["target_id"]) for x in ret["evidence"]]; union = list(dict.fromkeys([*direct, *evidence]))
        scores = _score_target_ids(model, store, str(pool["query_id"]), union, device); ranking = sorted(union, key=lambda x: (-scores[x], x))
        rows.append({"query_id": str(pool["query_id"]), "query_kind": pool.get("query_kind"), "positive_target_ids": sorted(pos), "direct_ann": direct, "evidence_ann": evidence, "U": union, "u_exact_ranking": ranking, "evidence_paths": ret.get("evidence", []), "direct_raw_recall@100": len(pos & set(direct))/len(pos) if pos else 0.0, "u_raw_recall": len(pos & set(union))/len(pos) if pos else 0.0, **{f"u_recall@{k}": len(pos & set(ranking[:k]))/len(pos) if pos else 0.0 for k in (10,20,50)}})
    ranking_path = dest / "rankings.jsonl.gz"; write_rows(ranking_path, rows)
    # The ANN payload may live on a RAM disk and is removed below.  Preserve
    # a compact manifest copy beside the rankings so the recorded path remains
    # inspectable after cleanup.
    manifest_copy = dest / "index_manifest.json"
    manifest_copy.write_text(index_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    metrics = {"format_version": 1, "status": "complete", "arm": arm, "seed": seed, "checkpoint_sha256": _sha(ck), "corpus_tables": len(ids_by_type.get("table", [])), "queries": len(rows), "direct_raw@100": statistics.fmean(r["direct_raw_recall@100"] for r in rows), "u_raw_recall": statistics.fmean(r["u_raw_recall"] for r in rows), "u_r10": statistics.fmean(r["u_recall@10"] for r in rows), "u_r20": statistics.fmean(r["u_recall@20"] for r in rows), "u_cr50": statistics.fmean(r["u_recall@50"] for r in rows), "u_size_mean": statistics.fmean(len(r["U"]) for r in rows), "index_manifest": str(manifest_copy.resolve()), "index_manifest_sha256": _sha(manifest_copy), "index_payload_root": str(index_dir.resolve()), "rankings": str(ranking_path.resolve()), "completed_at_utc": now()}
    write_json(metrics_path, metrics)
    if not keep_index:
        # Keep the manifest hash in metrics, but remove multi-gigabyte ANN
        # payloads after the raw ranking has been materialized.
        import shutil
        shutil.rmtree(index_dir)
    return metrics


@torch.inference_mode()
def evaluate_exact_direct(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    # Exact Direct100 is independent of ANN admission: always score every
    # corpus table for the frozen R23 query set.  In particular, do not reuse
    # an ANN U union if a full-lake ANN artifact happens to exist.
    existing = [dict(r) for r in read_rows(r23_paths(root)["candidate_pools"])]
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"); store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=50000)
    table_ids = json.loads((r21_out(root) / "indexes" / "Qwen-Raw" / "table_ids.json").read_text())
    target = torch.stack([store.embedding_features(str(x)).embedding for x in table_ids]).to(device=device, dtype=torch.float32)
    ck_dir = out(root) / arm / f"seed{seed}" / "checkpoints"; ck = sorted(ck_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]; model = load_student(ck, device).eval(); tv = model.project(target, "table", role="target"); rel = model.relations[model.relation_key("table", "table")]
    rows = []
    for start in range(0, len(existing), 8):
        batch = existing[start:start+8]; q = torch.stack([store.embedding_features(str(r["query_id"])).embedding for r in batch]).to(device=device, dtype=torch.float32); vals, idx = (model.project(q, "table", role="query") @ rel @ tv.T).topk(k=100, dim=1)
        for old, ii, vv in zip(batch, idx.cpu(), vals.cpu()):
            ids = [str(table_ids[int(i)]) for i in ii]; pos = set(map(str, old["positive_target_ids"])); rows.append({"query_id": old["query_id"], "query_kind": old.get("query_kind"), "positive_target_ids": sorted(pos), "exact_ids": ids, "exact_scores": [float(x) for x in vv], **{f"exact_recall@{k}": len(pos & set(ids[:k]))/len(pos) if pos else 0.0 for k in (10,20,50,100)}})
    path = out(root) / "evaluations" / arm / f"seed{seed}" / "direct_exact_full_lake.jsonl.gz"; write_rows(path, rows)
    result = {"format_version": 1, "status": "complete", "arm": arm, "seed": seed, "step": int(ck.stem.split("_")[-1]), "corpus_tables": len(table_ids), "queries": len(rows), "exact": {f"recall@{k}": statistics.fmean(r[f"exact_recall@{k}"] for r in rows) for k in (10,20,50,100)}, "rankings": str(path.resolve()), "checkpoint_sha256": _sha(ck), "admission_source": "full-lake exact Direct over frozen R23 query set", "query_source": str(r23_paths(root)["candidate_pools"].resolve()), "completed_at_utc": now()}; write_json(path.with_suffix(".json"), result); return result


def run_p0(root: Path) -> dict[str, Any]:
    audit_r23(root); registry = build_positive_registry(root)
    return {"status": "complete", "input_audit": str((out(root)/"P0_INPUT_AUDIT.json").resolve()), "positive_registry": registry}


def report(root: Path) -> dict[str, Any]:
    """Emit an execution matrix and compact result table without inventing CI."""
    jobs = []
    for arm in ("N-U", "H1-PC", *PATH_ARMS):
        for seed in SEEDS:
            train_cfg = out(root) / arm / f"seed{seed}" / "config.json"
            fixed = out(root) / "evaluations" / arm / f"seed{seed}" / "metrics.json"
            exact = out(root) / "evaluations" / arm / f"seed{seed}" / "direct_exact_full_lake.jsonl.json"
            if arm in {"P-Split-KD", "P-LSE-KD"} and not train_cfg.exists():
                status = "blocked" if (out(root) / "path_pool" / f"teacher_target_seed{seed}.blocked.json").exists() else "planned"
            elif train_cfg.exists() and fixed.exists():
                status = "completed" if exact.exists() else "partial"
            elif train_cfg.exists():
                status = "partial"
            else:
                status = "planned"
            jobs.append({"arm": arm, "seed": seed, "status": status,
                "train_config": str(train_cfg.resolve()) if train_cfg.exists() else None,
                "fixed_metrics": str(fixed.resolve()) if fixed.exists() else None,
                "exact_metrics": str(exact.resolve()) if exact.exists() else None})
    full_lake_arms = ("N-U", "H1-PC", *PATH_ARMS)
    expected_full_lake = [(arm, seed) for arm in full_lake_arms for seed in SEEDS]
    completed_full_lake = [
        (arm, seed) for arm, seed in expected_full_lake
        if (out(root) / "full_lake" / arm / f"seed{seed}" / "metrics.json").exists()
    ]
    missing_full_lake = [item for item in expected_full_lake if item not in completed_full_lake]
    h1_full_lake = [seed for arm, seed in completed_full_lake if arm == "H1-PC"]
    if not missing_full_lake:
        full_lake_status = "complete"
    elif completed_full_lake:
        full_lake_status = "partial"
    else:
        full_lake_status = "blocked"
    blocked_full_lake_arms = sorted({arm for arm, _seed in missing_full_lake})
    full_lake_reason = (
        "All 12 full-lake ANN jobs completed; new-arm indices were built on the host RAM disk and removed after raw rankings were materialized, while H1-PC reused the frozen R23 G2 index after projection/hash checks."
        if not missing_full_lake else
        "Full-lake ANN is partial: H1-PC can reuse the frozen R23 G2 index, while missing arms require new indices."
    )
    jobs_complete = not any(j["status"] in {"planned", "partial", "blocked"} for j in jobs)
    matrix = {"format_version": 1, "status": "complete" if jobs_complete and full_lake_status == "complete" else "partial",
        "p0": "complete" if (out(root)/"POSITIVE_REGISTRY.json").exists() else "planned", "jobs": jobs,
        "full_lake_admission": {
            "status": full_lake_status,
            "completed_h1_pc_seeds": h1_full_lake,
            "completed_jobs": [[arm, seed] for arm, seed in completed_full_lake],
            "blocked_arms": blocked_full_lake_arms,
            "reason": full_lake_reason,
            "fixed_pool_and_exact_fallback_are_recorded": True,
        },
        "conditional_teacher_feedback": "not_triggered", "fresh_c1_c2": "not_triggered", "created_at_utc": now()}
    write_json(out(root) / "EXECUTION_MATRIX.json", matrix)
    summary = []
    for arm in ("N-U", "H1-PC", *PATH_ARMS):
        for seed in SEEDS:
            m = out(root) / "evaluations" / arm / f"seed{seed}" / "metrics.json"
            e = out(root) / "evaluations" / arm / f"seed{seed}" / "direct_exact_full_lake.jsonl.json"
            f = out(root) / "full_lake" / arm / f"seed{seed}" / "metrics.json"
            if m.exists() or e.exists():
                md = json.loads(m.read_text()) if m.exists() else {}
                ed = json.loads(e.read_text()) if e.exists() else {}
                fd = json.loads(f.read_text()) if f.exists() else {}
                summary.append({"arm": arm, "seed": seed, "fixed_U_R10": md.get("pools", {}).get("U", {}).get("recall@10"), "fixed_U_raw": md.get("pools", {}).get("U", {}).get("raw_recall"), "exact_D_R10": ed.get("exact", {}).get("recall@10"), "full_lake_U_R10": fd.get("u_r10"), "admission_source": ed.get("admission_source")})
    kd_complete = all(
        (out(root) / arm / f"seed{seed}" / "config.json").exists()
        for arm in ("P-Split-KD", "P-LSE-KD")
        for seed in SEEDS
    )
    kd_note = (
        "P-Split-KD and P-LSE-KD used T1-B target logits recomputed on the common path pool; no Student logits were substituted."
        if kd_complete else
        "P-Split-KD and P-LSE-KD remain blocked by missing T1-B hidden states; no Student logits were substituted."
    )
    full_lake_note = (
        "All full-lake ANN jobs completed; new-arm indices used per-job RAM-disk builds, while H1-PC used a verified frozen R23 G2 index reuse."
        if full_lake_status == "complete" else
        "H1-PC full-lake ANN metrics are complete for both seeds via a verified frozen-index reuse; remaining arms are blocked or pending per EXECUTION_MATRIX.json."
    )
    result = {"format_version": 1, "status": matrix["status"], "execution_matrix": str((out(root)/"EXECUTION_MATRIX.json").resolve()), "summary": summary, "notes": ["U/M/D fixed-pool metrics are not full-lake admission.", full_lake_note, "Exact Direct metrics are full-lake brute-force Direct100 over the frozen R23 query set; they do not provide ANN U/E admission.", kd_note], "created_at_utc": now()}
    write_json(out(root) / "RESULTS.json", result)
    lines = [
        "# R24 Stage-1 execution results",
        "",
        f"Status: **{result['status']}**. The machine-readable artifacts are `EXECUTION_MATRIX.json` and `RESULTS.json`.",
        "",
        "The fixed U/M/D rows reuse the frozen R23 candidate pools, so they measure reranking only; they do not establish a full-lake admission gain.",
        "Exact Direct rows score the complete table corpus for the frozen R23 query set. Their `admission_source` field distinguishes this from ANN U/E admission.",
        full_lake_note + " (see `EXECUTION_MATRIX.json`).",
        "",
        "| Arm | Seed | Fixed U R@10 | Exact Direct R@10 | Full-lake ANN U R@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        fixed = "—" if row["fixed_U_R10"] is None else f"{row['fixed_U_R10']:.4f}"
        exact = "—" if row["exact_D_R10"] is None else f"{row['exact_D_R10']:.4f}"
        full = "—" if row["full_lake_U_R10"] is None else f"{row['full_lake_U_R10']:.4f}"
        lines.append(f"| {row['arm']} | {row['seed']} | {fixed} | {exact} | {full} |")
    lines.extend([
        "",
        kd_note,
        "Conditional Teacher-feedback and fresh C1/C2 extensions were not triggered.",
        "",
    ])
    (out(root) / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit-r23"); sub.add_parser("build-positive-registry"); sub.add_parser("p0"); sub.add_parser("prepare-teacher-backfill")
    p = sub.add_parser("train"); p.add_argument("--arm", required=True); p.add_argument("--seed", required=True, type=int); p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("build-path-pool"); p.add_argument("--seed", required=True, type=int)
    p = sub.add_parser("cache-target-teacher"); p.add_argument("--seed", required=True, type=int); p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("evaluate"); p.add_argument("--arm", required=True); p.add_argument("--seed", required=True, type=int); p.add_argument("--device", default="cuda:0")
    sub.add_parser("evaluate-all")
    sub.add_parser("report")
    args = parser.parse_args(); root = args.root.resolve()
    if args.cmd == "audit-r23": print(json.dumps(audit_r23(root), indent=2))
    elif args.cmd == "build-positive-registry": print(json.dumps(build_positive_registry(root), indent=2))
    elif args.cmd == "p0": print(json.dumps(run_p0(root), indent=2))
    elif args.cmd == "prepare-teacher-backfill": print(json.dumps(prepare_teacher_backfill(root), indent=2))
    elif args.cmd == "train":
        if args.arm in {"N-U", "H1-PC"}: print(json.dumps(_edge_training(root, args.arm, args.seed, args.device), indent=2))
        else: print(json.dumps(train_path(root, args.arm, args.seed, args.device), indent=2))
    elif args.cmd == "build-path-pool": print(json.dumps(build_path_pool(root, args.seed), indent=2))
    elif args.cmd == "cache-target-teacher":
        pool = out(root) / "path_pool" / f"common_seed{args.seed}.jsonl"; examples = load_target_examples(pool, split="train"); print(json.dumps(_teacher_target_cache(root, args.seed, examples, args.device), indent=2))
    elif args.cmd == "evaluate":
        fixed = evaluate_fixed(root, args.arm, args.seed, args.device)
        full = evaluate_full_lake(root, args.arm, args.seed, args.device)
        exact = evaluate_exact_direct(root, args.arm, args.seed, args.device)
        print(json.dumps({"fixed": fixed, "full_lake": full, "exact": exact}, indent=2))
    elif args.cmd == "evaluate-all":
        results = []
        for arm in ("N-U", "H1-PC", *PATH_ARMS):
            for seed in SEEDS:
                results.append({"fixed": evaluate_fixed(root, arm, seed, "cuda:0"), "full_lake": evaluate_full_lake(root, arm, seed, "cuda:0"), "exact": evaluate_exact_direct(root, arm, seed, "cuda:0")})
        print(json.dumps(results, indent=2))
    elif args.cmd == "report": print(json.dumps(report(root), indent=2))


if __name__ == "__main__":
    main()
