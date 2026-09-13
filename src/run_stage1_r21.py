"""Run the frozen R21 Student/KD and raw-Qwen retrieval experiment.

The runner keeps every expensive phase explicit so a partially completed run can
be audited and resumed.  It deliberately uses the existing frozen feature
cache and retrieval implementation; no model downloads or Stage-2 work are
performed here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import EdgeExample, load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore, OBJECT_TYPES
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.objectives import listwise_cross_entropy, distillation_kl
from mmdd_stage1.retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    build_indices,
    build_raw_embedding_indices,
    load_corpus_ids,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.scoring import ListScores, score_edge_batch
from mmdd_stage1.training import _student_edge_losses, student_gradient_norms
from run_stage1_r19 import _score_id_pairs, _teacher_paths, load_r19_checkpoint


ROOT_DEFAULT = Path(__file__).resolve().parents[1]
OUT_NAME = "stage1_optimization_r21_20260911"
SEEDS = (13, 29)
ARMS = ("Ssup", "SKD")
TRAIN_LISTS = 42143
LOGICAL_BATCH = 64
UPDATES_PER_EPOCH = math.ceil(TRAIN_LISTS / LOGICAL_BATCH)
EPOCHS = 2
FINAL_STEP = EPOCHS * UPDATES_PER_EPOCH
MID_STEP = UPDATES_PER_EPOCH
LR_RELATION = 1e-5
LR_PROJECTION = 1e-6
WEIGHT_DECAY = 0.01
TEMPERATURE = 1.0
KD_WEIGHT = 1.0
SEED_HASH = 210911

TEACHER_SHA = {
    13: "792c746b79dc8e61b80be145d20f118fe2dacc580fac829ab093b06aa20e5164",
    29: "3e8e497129927a9921d5f44f28504483420d076c414675b0a9f4704e8572a0a3",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def out(root: Path) -> Path:
    return root / "work" / OUT_NAME


def paths(root: Path) -> dict[str, Path]:
    r10 = root / "work/stage1_optimization_r10_20260907"
    r12 = root / "work/stage1_optimization_r12_20260908"
    r13 = root / "work/stage1_optimization_r13_20260909"
    r16 = root / "work/stage1_optimization_r16_20260910"
    r20 = root / "work/stage1_optimization_r20_20260911"
    return {
        "plan": root / "stage1_optimization_r21_plan_20260911.md",
        "b13": r13 / "taskD_witness_supervision/p_s_target_only/checkpoints/step_000178.pt",
        "b13_index": r13 / "taskD_witness_supervision/p_s_target_only/evaluation_step178/index",
        "b13_pool": r13 / "taskD_witness_supervision/p_s_target_only/evaluation_step178/path_pool.jsonl.gz",
        "b13_rankings": r13 / "taskD_witness_supervision/p_s_target_only/evaluation_step178/rankings.jsonl.gz",
        "features": r10 / "features_qwen3_vl_embedding_8b",
        "objects": r10 / "stage1_data/stage1_objects.jsonl",
        "splits": r10 / "taskA_protocol/splits.json",
        "train": r12 / "taskA_correctness/supervision/edge_lists.train_fit.jsonl",
        "dev": r12 / "taskA_correctness/supervision/edge_lists.dev.jsonl",
        "candidate_pools": r16 / "candidate_pools.jsonl.gz",
        "corpus": r10 / "stage1_data/stage1_corpus.jsonl",
        "train_manifest": r20 / "train_manifest_D0.jsonl",
        "teacher13": r20 / "D2/seed13/checkpoints/step_021072.pt",
        "teacher29": r20 / "D2/seed29/checkpoints/step_021072.pt",
        "teacher_extra": r12 / "taskC_training/teacher_extra",
        "teacher_extra_matched_gpu0": r16 / "teacher_extra_matched_gpu0",
        "teacher_extra_matched_gpu1": r16 / "teacher_extra_matched_gpu1",
        "teacher_extra_edges_gpu0": r16 / "teacher_extra_edges_gpu0",
        "teacher_extra_edges_gpu1": r16 / "teacher_extra_edges_gpu1",
    }


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def relation(example: EdgeExample) -> str:
    return StudentJoinabilityModel.relation_key(example.source_type, example.destination_type)


def train_manifest_rows(root: Path) -> list[dict[str, Any]]:
    rows = list(read_rows(paths(root)["train_manifest"]))
    if len(rows) != TRAIN_LISTS:
        raise RuntimeError(f"R21 requires {TRAIN_LISTS} train lists, found {len(rows)}")
    if any(str(row.get("split")) != "train" for row in rows):
        raise RuntimeError("R21 train manifest contains a non-train list")
    return rows


def freeze(root: Path) -> dict[str, Any]:
    ps = paths(root)
    required = [
        "plan", "b13", "features", "objects", "splits", "train", "dev",
        "candidate_pools", "corpus", "train_manifest", "teacher13", "teacher29",
    ]
    missing = [str(ps[name]) for name in required if not ps[name].exists()]
    if missing:
        raise FileNotFoundError("R21 missing inputs: " + ", ".join(missing))
    input_paths = {
        "plan": ps["plan"],
        "b13": ps["b13"],
        "features_manifest": ps["features"] / "manifest.jsonl",
        "objects": ps["objects"],
        "splits": ps["splits"],
        "train": ps["train"],
        "dev": ps["dev"],
        "candidate_pools": ps["candidate_pools"],
        "corpus": ps["corpus"],
        "train_manifest": ps["train_manifest"],
        "teacher13": ps["teacher13"],
        "teacher29": ps["teacher29"],
    }
    manifest = {
        "format_version": 1,
        "status": "pass",
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": checkpoint_fingerprint(path)}
            for key, path in input_paths.items()
        },
        "teacher_expected_sha256": {str(k): v for k, v in TEACHER_SHA.items()},
        "verified_at_utc": now(),
    }
    for seed in SEEDS:
        actual = manifest["inputs"][f"teacher{seed}"]["sha256"]
        if actual != TEACHER_SHA[seed]:
            raise RuntimeError(f"Teacher {seed} hash mismatch: {actual}")
    rows = train_manifest_rows(root)
    counts = Counter((str(r["source_type"]), str(r["destination_type"])) for r in rows)
    manifest["train_lists"] = {
        "count": len(rows),
        "relation_counts": {"%s->%s" % key: value for key, value in sorted(counts.items())},
        "candidate_widths": dict(Counter(len(r["candidate_ids"]) for r in rows)),
        "candidate_list_order_sha256": stable_hash([
            [str(r["query_id"]), list(map(str, r["candidate_ids"]))] for r in rows
        ]),
    }
    destination = out(root)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "INPUT_MANIFEST.json", manifest)
    protocol = {
        "format_version": 1,
        "status": "frozen",
        "plan_sha256": checkpoint_fingerprint(ps["plan"]),
        "training": {
            "arms": list(ARMS), "seeds": list(SEEDS), "lists": TRAIN_LISTS,
            "logical_batch": LOGICAL_BATCH, "epochs": EPOCHS,
            "updates_per_epoch": UPDATES_PER_EPOCH, "mid_step": MID_STEP,
            "final_step": FINAL_STEP, "temperature": TEMPERATURE,
            "kd_weight": KD_WEIGHT, "lr_relation": LR_RELATION,
            "lr_projection": LR_PROJECTION, "weight_decay": WEIGHT_DECAY,
            "optimizer": "fresh AdamW; no scheduler; no anchor/clipping",
        },
        "retrieval_budget": {"direct_k": 100, "evidence_k": 20, "targets_per_evidence": 20},
        "stage2": "out_of_scope",
        "frozen_at_utc": now(),
    }
    write_json(destination / "RESOLVED_CONFIG.json", protocol)
    (destination / "PLAN_FROZEN.md").write_text(ps["plan"].read_text(encoding="utf-8"), encoding="utf-8")
    (destination / "RUN_COMMANDS.md").write_text(
        "# R21 commands\n\n"
        "```bash\n"
        "python src/run_stage1_r21.py freeze\n"
        "python src/run_stage1_r21.py cache-teacher --seed 13 --device cuda:0\n"
        "python src/run_stage1_r21.py train --arm Ssup --seed 13 --device cuda:0\n"
        "```\n", encoding="utf-8"
    )
    write_json(destination / "EXECUTION_MATRIX.json", {
        "format_version": 1,
        "status": "frozen",
        "jobs": [
            {"arm": arm, "seed": seed, "status": "pending", "device": None}
            for arm in ARMS for seed in SEEDS
        ],
    })
    return manifest


def _feature_paths(ps: dict[str, Path]) -> list[Path]:
    return [
        ps[name] for name in ps
        if name.startswith("teacher_extra") and (ps[name] / "teacher_manifest.jsonl").is_file()
    ]


def feature_manifest(root: Path) -> dict[str, Any]:
    ps = paths(root)
    rows = list(read_rows(ps["features"] / "manifest.jsonl"))
    if not rows:
        raise RuntimeError("empty Qwen feature manifest")
    first = rows[0]
    store = FeatureStore.from_path(ps["features"], cache_size=4)
    sample_ids = [str(row["object_id"]) for row in rows[: min(32, len(rows))]]
    dims = Counter()
    norms = []
    dtypes = Counter()
    for object_id in sample_ids:
        value = store.embedding_features(object_id).embedding
        dims[int(value.shape[0])] += 1
        norms.append(float(value.norm()))
        dtypes[str(value.dtype)] += 1
    payload = {
        "format_version": 1,
        "raw_feature_root": str(ps["features"].resolve()),
        "feature_manifest_sha256": checkpoint_fingerprint(ps["features"] / "manifest.jsonl"),
        "backbone_id": "Qwen3-VL-Embedding-8B (cached final object embedding)",
        "prompt_role_configuration": "frozen R10 feature cache; role serialization unchanged",
        "pooling_definition": "cached final embedding from the same encoder forward pass",
        "raw_dimension": dims.most_common(1)[0][0],
        "raw_dtype": dtypes.most_common(1)[0][0],
        "normalization_status": {
            "sample_norm_min": min(norms), "sample_norm_max": max(norms),
            "sample_norm_mean": statistics.fmean(norms),
            "treated_as_normalized": all(abs(v - 1.0) < 1e-3 for v in norms),
        },
        "manifest_first_record_keys": sorted(first),
        "objects": len(rows),
        "recorded_at_utc": now(),
    }
    write_json(out(root) / "QWEN_RAW_FEATURE_MANIFEST.json", payload)
    return payload


def _load_teacher(root: Path, seed: int, device: torch.device):
    ps = paths(root)
    arm, saved_seed, step, model, payload = load_r19_checkpoint(ps[f"teacher{seed}"], device)
    if saved_seed != seed or step != 21072:
        raise RuntimeError("R21 Teacher lineage/step mismatch")
    if checkpoint_fingerprint(ps[f"teacher{seed}"]) != TEACHER_SHA[seed]:
        raise RuntimeError("R21 Teacher checkpoint hash mismatch")
    model.eval()
    return model, payload


@torch.inference_mode()
def cache_teacher(root: Path, seed: int, device_name: str, batch_size: int = 512) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("Teacher soft-score caching requires CUDA")
    destination = out(root) / "teacher_soft_scores" / f"lineage{seed}.jsonl.gz"
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.is_file() and manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    ps = paths(root)
    rows = train_manifest_rows(root)
    device = torch.device(device_name)
    model, _payload = _load_teacher(root, seed, device)
    store = FeatureStore.from_path(ps["features"], cache_size=24000, teacher_paths=_feature_paths(ps))
    cache: dict[str, torch.Tensor] = model.new_compression_cache()
    started = time.monotonic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            pairs = [(str(row["query_id"]), str(value)) for value in row["candidate_ids"]]
            scores = _score_id_pairs(model, pairs, store, device, batch_size=batch_size, cache=cache)
            record = {
                "query_id": str(row["query_id"]),
                "relation": f"{row['source_type']}->{row['destination_type']}",
                "candidate_ids": [str(value) for value in row["candidate_ids"]],
                "scores": scores,
            }
            handle.write(json.dumps(record) + "\n")
            if index % 500 == 0:
                print(json.dumps({"stage": "cache_teacher", "seed": seed, "lists": index,
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    temporary.replace(destination)
    result = {
        "format_version": 1, "status": "complete", "seed": seed,
        "teacher_checkpoint_sha256": TEACHER_SHA[seed],
        "teacher_scorer_code_sha256": checkpoint_fingerprint(Path(__file__).with_name("run_stage1_r19.py")),
        "feature_manifest_sha256": checkpoint_fingerprint(ps["features"] / "manifest.jsonl"),
        "train_manifest_sha256": checkpoint_fingerprint(ps["train_manifest"]),
        "lists": len(rows), "score_space": "raw_logit", "temperature": TEMPERATURE,
        "path": str(destination.resolve()), "sha256": checkpoint_fingerprint(destination),
        "device": device_name, "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": now(),
    }
    write_json(manifest_path, result)
    return result


def _teacher_list_scores(rows: Sequence[EdgeExample], records: Sequence[dict[str, Any]], device: torch.device) -> ListScores:
    if len(rows) != len(records):
        raise RuntimeError("Teacher cache and training batch lengths differ")
    lengths = [len(row.candidate_ids) for row in rows]
    width = max(lengths)
    logits = torch.zeros((len(rows), width), device=device, dtype=torch.float32)
    positive_masks = torch.zeros_like(logits, dtype=torch.bool)
    for index, (example, record) in enumerate(zip(rows, records)):
        ids = [str(value) for value in record["candidate_ids"]]
        if ids != list(map(str, example.candidate_ids)):
            raise RuntimeError(f"Teacher candidate identity mismatch for {example.query_id}")
        values = torch.tensor(record["scores"], device=device, dtype=torch.float32)
        logits[index, : len(ids)] = values
        positives = set(map(str, example.positive_ids)) or {str(example.candidate_ids[example.positive_index])}
        positive_masks[index, : len(ids)] = torch.tensor([value in positives for value in ids], device=device)
    mask = torch.arange(width, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
    positive_indices = positive_masks.float().argmax(dim=1)
    return ListScores(logits, mask, positive_indices, positive_masks)


def _teacher_cache_key(query_id: str, relation_name: str, candidate_ids: Sequence[str]) -> tuple[str, str, str]:
    return (str(query_id), str(relation_name).replace("->", "_to_"), stable_hash(list(map(str, candidate_ids))))


def _load_teacher_cache(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    # Query IDs recur both across directed relations and within one relation
    # (e.g. several evidence positives).  Bind each score row to the ordered
    # candidate-list fingerprint as well.
    return {
        _teacher_cache_key(str(row["query_id"]), str(row["relation"]), row["candidate_ids"]): row
        for row in read_rows(path)
    }


def _fresh_student(root: Path, device: torch.device) -> StudentJoinabilityModel:
    model = load_student(paths(root)["b13"], device)
    model.train()
    return model


def smoke(root: Path, seed: int, device_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("R21 smoke requires CUDA")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    rows = train_manifest_rows(root)[:LOGICAL_BATCH]
    examples = load_edge_examples(paths(root)["train_manifest"], split="train")[:LOGICAL_BATCH]
    store = FeatureStore.from_path(paths(root)["features"], cache_size=4096)
    store.preload_embeddings({x for e in examples for x in [e.query_id, *e.candidate_ids]})
    model = _fresh_student(root, device)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    scores = score_edge_batch(model, examples, store, device)
    loss = listwise_cross_entropy(scores.logits, scores.positive_indices, scores.candidate_mask, scores.positive_mask)
    loss.backward()
    gradients = student_gradient_norms(model)
    finite = math.isfinite(float(loss.detach())) and all(
        value is None or math.isfinite(float(value)) for value in gradients.values()
    )
    optimizer = torch.optim.AdamW(
        [{"params": model.relation_parameters(), "lr": LR_RELATION},
         {"params": model.projections.parameters(), "lr": LR_PROJECTION}],
        weight_decay=WEIGHT_DECAY,
    )
    optimizer.step()
    changed = any(not torch.equal(before[key], value) for key, value in model.state_dict().items())
    result = {
        "format_version": 1, "status": "pass" if finite and changed else "fail",
        "seed": seed, "device": device_name, "lists": len(examples),
        "loss": float(loss), "model_updated": changed,
        "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "gradient_norms": gradients, "completed_at_utc": now(),
    }
    write_json(out(root) / "smoke" / f"seed{seed}.json", result)
    if result["status"] != "pass":
        raise RuntimeError("R21 smoke failed")
    return result


def train(root: Path, arm: str, seed: int, device_name: str, cache_size: int = 24000, microbatch: int = 1) -> dict[str, Any]:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("invalid R21 arm or seed")
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("R21 training requires CUDA")
    job = out(root) / arm / f"seed{seed}"
    final_path = job / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
    if final_path.is_file() and (job / "train_history.jsonl").is_file():
        return json.loads((job / "config.json").read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    ps = paths(root)
    examples = load_edge_examples(ps["train_manifest"], split="train")
    if len(examples) != TRAIN_LISTS:
        raise RuntimeError("R21 training list count changed")
    store = FeatureStore.from_path(ps["features"], cache_size=cache_size)
    ids = {example.query_id for example in examples}
    ids.update(candidate for example in examples for candidate in example.candidate_ids)
    store.preload_embeddings(ids)
    teacher_by_id = None
    if arm == "SKD":
        cache_path = out(root) / "teacher_soft_scores" / f"lineage{seed}.jsonl.gz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"run cache-teacher for seed {seed} first")
        teacher_by_id = _load_teacher_cache(cache_path)
    model = _fresh_student(root, device)
    optimizer = torch.optim.AdamW(
        [{"params": model.relation_parameters(), "lr": LR_RELATION},
         {"params": model.projections.parameters(), "lr": LR_PROJECTION}],
        weight_decay=WEIGHT_DECAY,
    )
    job.mkdir(parents=True, exist_ok=True)
    (job / "checkpoints").mkdir(exist_ok=True)
    history: list[dict[str, Any]] = []
    rng = random.Random(SEED_HASH + seed)
    started = time.monotonic()
    step = 0
    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        losses = []
        sup_losses = []
        kd_losses = []
        grad_norms = []
        for batch_no, start in enumerate(range(0, len(order), LOGICAL_BATCH), 1):
            batch_examples = [examples[index] for index in order[start : start + LOGICAL_BATCH]]
            optimizer.zero_grad(set_to_none=True)
            student_scores = score_edge_batch(model, batch_examples, store, device)
            teacher_scores = None
            if teacher_by_id is not None:
                teacher_scores = _teacher_list_scores(
                    batch_examples,
                    [teacher_by_id[_teacher_cache_key(example.query_id, relation(example), example.candidate_ids)] for example in batch_examples],
                    device,
                )
            terms = _student_edge_losses(
                model, batch_examples, student_scores, teacher_scores, student_scores,
                None, ranking_weight=1.0, temperature=TEMPERATURE,
                distillation_weight=KD_WEIGHT if teacher_scores is not None else 0.0,
                edge_bce_weight=0.0, anchor_weight=0.0, anchor_weight_evidence=0.0,
            )
            terms["loss"].backward()
            grad = student_gradient_norms(model)
            optimizer.step()
            step += 1
            losses.append(float(terms["loss"].detach()))
            sup_losses.append(float(terms["supervised_loss"].detach()))
            kd_losses.append(float(terms["distillation_loss"].detach()))
            grad_norms.append(
                math.sqrt(
                    sum(float(value) ** 2 for value in grad.values() if value is not None)
                )
            )
            if batch_no % 100 == 0 or batch_no == UPDATES_PER_EPOCH:
                print(json.dumps({"arm": arm, "seed": seed, "epoch": epoch,
                                  "update": step, "loss": statistics.fmean(losses[-100:]),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
        checkpoint_path = job / "checkpoints" / f"step_{step:06d}.pt"
        payload = {
            "format_version": 1, "model_kind": "student", "completed_stage": "r21-skd" if arm == "SKD" else "r21-ssup",
            "arm": arm, "seed": seed, "step": step, "config": model.config(),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "projection_references": {"origin": "B13 checkpoint", "source_sha256": checkpoint_fingerprint(ps["b13"])},
        }
        torch.save(payload, checkpoint_path)
        record = {"epoch": epoch, "step": step, "train_loss": statistics.fmean(losses),
                  "supervised_loss": statistics.fmean(sup_losses), "kd_loss": statistics.fmean(kd_losses),
                  "gradient_norm_l2_mean": statistics.fmean(grad_norms),
                  "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_fingerprint(checkpoint_path),
                  "elapsed_seconds": time.monotonic() - started}
        history.append(record)
        write_rows(job / "train_history.jsonl", history)
    config = {
        "format_version": 1, "status": "pass", "arm": arm, "seed": seed,
        "initialization_checkpoint": str(ps["b13"].resolve()), "initialization_sha256": checkpoint_fingerprint(ps["b13"]),
        "train_manifest_sha256": checkpoint_fingerprint(ps["train_manifest"]),
        "teacher_cache": None if teacher_by_id is None else str((out(root) / "teacher_soft_scores" / f"lineage{seed}.jsonl.gz").resolve()),
        "optimizer": {"relation_lr": LR_RELATION, "projection_lr": LR_PROJECTION, "weight_decay": WEIGHT_DECAY, "scheduler": None},
        "updates": step, "device": device_name, "history": history, "completed_at_utc": now(),
    }
    write_json(job / "config.json", config)
    return config


def _load_student_checkpoint(root: Path, arm: str, seed: int, step: int, device: torch.device):
    if arm == "B13":
        path = paths(root)["b13"]
    else:
        path = out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    model = load_student(path, device).eval()
    return model, path


def _score_target_ids(model: StudentJoinabilityModel, store: FeatureStore, query_id: str, target_ids: Sequence[str], device: torch.device, batch_size: int = 1024) -> dict[str, float]:
    query = store.embedding_features(query_id)
    values: dict[str, float] = {}
    for start in range(0, len(target_ids), batch_size):
        ids = list(target_ids[start : start + batch_size])
        targets = [store.embedding_features(value) for value in ids]
        scores = model.score_pairs_in_space([query] * len(ids), targets, "raw_logit")
        values.update((value, float(score)) for value, score in zip(ids, scores.detach().cpu().tolist()))
    return values


def _metric_rows(rows: Sequence[dict[str, Any]], prefix: str = "", include_by_kind: bool = True) -> dict[str, Any]:
    if not rows:
        return {"queries": 0}
    result: dict[str, Any] = {"queries": len(rows)}
    for key in ("raw_recall", "recall@10", "recall@20", "recall@50"):
        values = [float(row[f"{prefix}{key}"]) for row in rows]
        result[key] = statistics.fmean(values)
    if include_by_kind:
        by_kind = {}
        for kind in ("implicit", "explicit"):
            selected = [row for row in rows if row.get("query_kind") == kind]
            by_kind[kind] = _metric_rows(selected, prefix, include_by_kind=False) if selected else {"queries": 0}
        result["by_query_kind"] = by_kind
    return result


@torch.inference_mode()
def evaluate_fixed(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("fixed-pool evaluation requires CUDA")
    destination = out(root) / "fixed_pool" / arm / f"seed{seed}" / f"step_{step:06d}"
    metrics_path = destination / "metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    model, checkpoint = _load_student_checkpoint(root, arm, seed, step, device)
    store = FeatureStore.from_path(paths(root)["features"], cache_size=40000)
    candidate_rows = list(read_rows(paths(root)["candidate_pools"]))
    output_rows = []
    for row in candidate_rows:
        positive = set(map(str, row["positive_target_ids"]))
        pools = {
            "natural": [str(v) for v in row["natural_candidate_ids"]],
            "direct100": [str(v) for v in row["ann_direct100_ids"]],
            "matched": [str(v) for v in row["matched_direct_candidate_ids"]],
        }
        all_ids = list(dict.fromkeys(value for values in pools.values() for value in values))
        score_map = _score_target_ids(model, store, str(row["query_id"]), all_ids, device)
        record = {"query_id": str(row["query_id"]), "query_kind": row["query_kind"], "positive_target_ids": sorted(positive), "pools": {}}
        for name, ids in pools.items():
            ranking = sorted(dict.fromkeys(ids), key=lambda value: (-score_map[value], value))
            hits = len(positive & set(ids))
            record["pools"][name] = {
                "candidate_ids": ids, "ranking": ranking, "candidate_count": len(ids),
                "query_kind": row["query_kind"],
                "raw_recall": hits / len(positive),
                "recall@10": len(positive & set(ranking[:10])) / len(positive),
                "recall@20": len(positive & set(ranking[:20])) / len(positive),
                "recall@50": len(positive & set(ranking[:50])) / len(positive),
            }
        output_rows.append(record)
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "rankings.jsonl.gz", output_rows)
    metrics = {
        "format_version": 1, "status": "complete", "arm": arm, "seed": seed, "step": step,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "candidate_pool_sha256": checkpoint_fingerprint(paths(root)["candidate_pools"]),
        "natural": _metric_rows([r["pools"]["natural"] for r in output_rows]),
        "direct100": _metric_rows([r["pools"]["direct100"] for r in output_rows]),
        "matched": _metric_rows([r["pools"]["matched"] for r in output_rows]),
        "rankings": str((destination / "rankings.jsonl.gz").resolve()),
        "device": device_name, "completed_at_utc": now(),
    }
    write_json(metrics_path, metrics)
    return metrics


def build_index(root: Path, arm: str, seed: int, step: int, device_name: str, raw: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("index building requires CUDA")
    device = torch.device(device_name)
    ps = paths(root)
    store = FeatureStore.from_path(ps["features"], cache_size=100000)
    ids_by_type = load_corpus_ids(ps["corpus"], store)
    corpus_sha = checkpoint_fingerprint(ps["corpus"])
    if raw:
        directory = out(root) / "indexes" / "Qwen-Raw"
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            build_raw_embedding_indices(store, ids_by_type, directory, corpus_sha256=corpus_sha, batch_size=4096, ef_search=100)
        return json.loads(manifest.read_text(encoding="utf-8"))
    model, checkpoint = _load_student_checkpoint(root, arm, seed, step, device)
    directory = out(root) / "indexes" / arm / f"seed{seed}" / f"step_{step:06d}"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        build_indices(model, store, ids_by_type, directory, device=device, checkpoint_sha256=checkpoint_fingerprint(checkpoint), corpus_sha256=corpus_sha, batch_size=4096)
    return json.loads(manifest.read_text(encoding="utf-8"))


@torch.inference_mode()
def evaluate_full_lake(root: Path, arm: str, seed: int, step: int, device_name: str, raw: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("full-lake retrieval requires CUDA")
    device = torch.device(device_name)
    ps = paths(root)
    store = FeatureStore.from_path(ps["features"], cache_size=120000)
    examples = load_target_examples(ps["dev"].with_name("target_lists.r10_test_regression.jsonl"), split="test") if (ps["dev"].with_name("target_lists.r10_test_regression.jsonl")).is_file() else []
    # The R16 candidate pool is the frozen 1,198-query test protocol.  Its query
    # metadata and positives are used directly so all generators share exactly
    # the same evaluation population.
    pool_rows = list(read_rows(ps["candidate_pools"]))
    query_ids = [str(row["query_id"]) for row in pool_rows]
    corpus_sha = checkpoint_fingerprint(ps["corpus"])
    if raw:
        index_dir = out(root) / "indexes" / "Qwen-Raw"
        indices = RawEmbeddingANNIndices(store, index_dir, corpus_sha256=corpus_sha)
        model = None
        label = "Qwen-Raw"
        checkpoint = None
    else:
        model, checkpoint = _load_student_checkpoint(root, arm, seed, step, device)
        index_dir = out(root) / "indexes" / arm / f"seed{seed}" / f"step_{step:06d}"
        indices = StudentANNIndices(model, store, index_dir, device=device, checkpoint_sha256=checkpoint_fingerprint(checkpoint), corpus_sha256=corpus_sha)
        label = arm
    destination = out(root) / "full_lake" / label / ("raw" if raw else f"seed{seed}_step{step:06d}")
    metrics_path = destination / "metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    detailed = retrieve_zero_one_hop_detailed_many(query_ids, indices, k=100, direct_k=100, evidence_k=20, targets_per_evidence=20, query_batch_size=16)
    ranking_rows = []
    for pool, retrieved in zip(pool_rows, detailed):
        positive = set(map(str, pool["positive_target_ids"]))
        direct_ids = [str(item["target_id"]) for item in retrieved["direct"]]
        evidence_ids = [str(item["target_id"]) for item in retrieved["evidence"]]
        union = list(dict.fromkeys([*direct_ids, *evidence_ids]))
        if model is None:
            # ANN scores are inner products for raw embeddings; exact scores are
            # computed below from the same cached vectors.
            query = store.embedding_features(str(pool["query_id"])).embedding
            score_map = {target: float(torch.dot(query, store.embedding_features(target).embedding)) for target in union}
        else:
            score_map = _score_target_ids(model, store, str(pool["query_id"]), union, device)
        ranking = sorted(union, key=lambda target: (-score_map[target], target))
        direct_exact = sorted(direct_ids, key=lambda target: (-score_map[target], target))
        record = {
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
        }
        ranking_rows.append(record)
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "rankings.jsonl.gz", ranking_rows)
    def mean(name: str, rows: Sequence[dict[str, Any]] = ranking_rows) -> float:
        return statistics.fmean(float(row[name]) for row in rows)
    metrics = {
        "format_version": 1, "status": "complete", "retriever": label,
        "seed": seed if not raw else None, "step": step if not raw else None,
        "queries": len(ranking_rows), "direct_raw@100": mean("direct_raw_recall@100"),
        "u_raw_recall": mean("u_raw_recall"), "u_r10": mean("u_recall@10"),
        "u_r20": mean("u_recall@20"), "u_cr50": mean("u_recall@50"),
        "direct_exact_r10": mean("direct_exact_recall@10"),
        "u_size_mean": statistics.fmean(row["u_size"] for row in ranking_rows),
        "by_query_kind": {
            kind: {name: mean(name, [row for row in ranking_rows if row["query_kind"] == kind])
                   for name in ("direct_raw_recall@100", "u_raw_recall", "u_recall@10", "u_recall@20", "u_recall@50")}
            for kind in ("implicit", "explicit")
        },
        "rankings": str((destination / "rankings.jsonl.gz").resolve()),
        "completed_at_utc": now(),
    }
    write_json(metrics_path, metrics)
    return metrics


@torch.inference_mode()
def evaluate_exact_direct(
    root: Path,
    arm: str,
    seed: int,
    step: int,
    device_name: str,
    *,
    raw: bool = False,
    query_batch_size: int = 32,
) -> dict[str, Any]:
    """Compute genuine full-corpus QT Top-100 and compare it with ANN."""

    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("exact full-lake search requires CUDA")
    device = torch.device(device_name)
    ps = paths(root)
    label = "Qwen-Raw" if raw else arm
    source_dir = out(root) / "full_lake" / label / (
        "raw" if raw else f"seed{seed}_step{step:06d}"
    )
    source_rankings = source_dir / "rankings.jsonl.gz"
    if not source_rankings.is_file():
        raise FileNotFoundError(source_rankings)
    existing = list(read_rows(source_rankings))
    store = FeatureStore.from_path(ps["features"], cache_size=50000)
    table_ids = json.loads(
        (out(root) / "indexes/Qwen-Raw/table_ids.json").read_text(encoding="utf-8")
    )
    target_embeddings = torch.stack(
        [store.embedding_features(str(value)).embedding for value in table_ids]
    ).to(device=device, dtype=torch.float32)
    model = None
    checkpoint = None
    if raw:
        target_vectors = target_embeddings
    else:
        model, checkpoint = _load_student_checkpoint(root, arm, seed, step, device)
        target_vectors = model.project(
            target_embeddings, "table", role="target"
        )
    output_rows = []
    for start in range(0, len(existing), query_batch_size):
        batch = existing[start : start + query_batch_size]
        query_embeddings = torch.stack(
            [store.embedding_features(str(row["query_id"])).embedding for row in batch]
        ).to(device=device, dtype=torch.float32)
        if raw:
            query_vectors = query_embeddings
        else:
            assert model is not None
            query_vectors = model.project(
                query_embeddings, "table", role="query"
            ) @ model.relations[model.relation_key("table", "table")]
        scores = query_vectors @ target_vectors.T
        values, indices = scores.topk(k=100, dim=1)
        for row, row_indices, row_values in zip(batch, indices.cpu(), values.cpu()):
            exact_ids = [str(table_ids[int(index)]) for index in row_indices]
            ann_ids = [str(value) for value in row["direct_ann"][:100]]
            positives = set(map(str, row["positive_target_ids"]))
            record = {
                "query_id": str(row["query_id"]),
                "query_kind": str(row["query_kind"]),
                "positive_target_ids": sorted(positives),
                "exact_ids": exact_ids,
                "exact_scores": [float(value) for value in row_values],
                "ann_ids": ann_ids,
                "neighbor_recall@100": len(set(exact_ids) & set(ann_ids)) / 100,
            }
            for k in (10, 20, 50, 100):
                record[f"exact_recall@{k}"] = len(positives & set(exact_ids[:k])) / len(positives)
                record[f"ann_recall@{k}"] = len(positives & set(ann_ids[:k])) / len(positives)
            output_rows.append(record)
    exact_path = source_dir / "direct_exact_full_lake.jsonl.gz"
    write_rows(exact_path, output_rows)
    result = {
        "format_version": 1,
        "status": "complete",
        "retriever": label,
        "seed": None if raw else seed,
        "step": None if raw else step,
        "corpus_tables": len(table_ids),
        "queries": len(output_rows),
        "exact": {
            f"recall@{k}": statistics.fmean(row[f"exact_recall@{k}"] for row in output_rows)
            for k in (10, 20, 50, 100)
        },
        "ann": {
            f"recall@{k}": statistics.fmean(row[f"ann_recall@{k}"] for row in output_rows)
            for k in (10, 20, 50, 100)
        },
        "ann_neighbor_recall@100": statistics.fmean(
            row["neighbor_recall@100"] for row in output_rows
        ),
        "by_query_kind": {
            kind: {
                f"exact_recall@{k}": statistics.fmean(
                    row[f"exact_recall@{k}"]
                    for row in output_rows if row["query_kind"] == kind
                )
                for k in (10, 20, 50, 100)
            }
            for kind in ("implicit", "explicit")
        },
        "checkpoint_sha256": None if checkpoint is None else checkpoint_fingerprint(checkpoint),
        "rankings": str(exact_path.resolve()),
        "completed_at_utc": now(),
    }
    write_json(source_dir / "EXACT_ANN.json", result)
    metrics_path = source_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["direct_exact_full_lake"] = result
    write_json(metrics_path, metrics)
    return result


def raw_parity(root: Path, device_name: str = "cuda:1", sample_count: int = 512) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("raw parity requires CUDA")
    ps = paths(root)
    store = FeatureStore.from_path(ps["features"], cache_size=4096)
    rows = list(read_rows(ps["candidate_pools"]))
    directory = out(root) / "indexes" / "Qwen-Raw"
    corpus_sha = checkpoint_fingerprint(ps["corpus"])
    if not (directory / "manifest.json").is_file():
        ids_by_type = load_corpus_ids(ps["corpus"], store)
        build_raw_embedding_indices(store, ids_by_type, directory, corpus_sha256=corpus_sha, batch_size=4096, ef_search=100)
    indices = RawEmbeddingANNIndices(store, directory, corpus_sha256=corpus_sha)
    # Use actual ANN-returned neighbours.  This checks the index's reported IP
    # against direct cosine without conflating score parity with whether an
    # arbitrary historical candidate happened to be in the returned Top-100.
    pairs = []
    for row in rows:
        for right, _ann_score in indices.search(str(row["query_id"]), "table", 100):
            pairs.append((str(row["query_id"]), str(right)))
            if len(pairs) >= sample_count:
                break
        if len(pairs) >= sample_count:
            break
    errors = []
    for left, right in pairs:
        direct = float(torch.dot(store.embedding_features(left).embedding, store.embedding_features(right).embedding))
        ann = dict(indices.search(left, "table", 100)).get(right)
        if ann is not None:
            errors.append(abs(direct - float(ann)))
    result = {"format_version": 1, "status": "pass" if len(errors) == sample_count and max(errors) <= 1e-4 else "fail", "pairs": len(pairs), "matched_pairs": len(errors), "max_abs_score_difference": max(errors, default=None), "mean_abs_score_difference": statistics.fmean(errors) if errors else None, "index_manifest_sha256": checkpoint_fingerprint(directory / "manifest.json"), "completed_at_utc": now()}
    write_json(out(root) / "QWEN_RAW_SCORE_PARITY.json", result)
    return result


@torch.inference_mode()
def teacher_parity(root: Path, seed: int, device_name: str = "cuda:0", sample_lists: int = 8) -> dict[str, Any]:
    """Compare the frozen D2 scorer with raw logits saved by R20."""
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("Teacher parity requires CUDA")
    path = root / "work/stage1_optimization_r20_20260911" / "D2" / f"seed{seed}" / "eval_step021072" / "direct100.jsonl.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    device = torch.device(device_name)
    model, _ = _load_teacher(root, seed, device)
    store = FeatureStore.from_path(paths(root)["features"], cache_size=20000, teacher_paths=_feature_paths(paths(root)))
    pairs, expected = [], []
    for row in list(read_rows(path))[:sample_lists]:
        # R20 stores raw_scores in the saved ranking order, not candidate-list
        # order; bind each score to its corresponding ranked target explicitly.
        pairs.extend((str(row["query_id"]), str(cid)) for cid in row["ranking"])
        expected.extend(float(v) for v in row["raw_scores"])
    actual = _score_id_pairs(model, pairs, store, device, batch_size=256, cache=model.new_compression_cache())
    errors = [abs(a - b) for a, b in zip(actual, expected, strict=True)]
    result = {"format_version": 1, "status": "pass" if errors and max(errors) <= 1e-4 else "fail", "seed": seed, "lists": sample_lists, "pairs": len(errors), "max_abs_logit_difference": max(errors, default=None), "mean_abs_logit_difference": statistics.fmean(errors) if errors else None, "historical_source": str(path.resolve()), "teacher_checkpoint_sha256": TEACHER_SHA[seed], "completed_at_utc": now()}
    write_json(out(root) / f"STEP0_PARITY_seed{seed}.json", result)
    return result


def environment_manifest(root: Path) -> dict[str, Any]:
    payload = {"format_version": 1, "status": "complete", "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available_at_record": torch.cuda.is_available(), "devices": torch.cuda.device_count(), "ann": {"backend": "hnswlib", "space": "ip", "ef_search": 100, "direct_k": 100, "evidence_k": 20, "targets_per_evidence": 20}, "dtype": "float32", "recorded_at_utc": now()}
    write_json(out(root) / "ENVIRONMENT.json", payload)
    return payload


def ann_score_parity(root: Path) -> dict[str, Any]:
    exact = []
    for path in sorted((out(root) / "full_lake").glob("*/**/EXACT_ANN.json")):
        payload = json.loads(path.read_text(encoding="utf-8")); exact.append({"path": str(path.relative_to(out(root))), "ann_neighbor_recall@100": payload.get("ann_neighbor_recall@100"), "status": payload.get("status")})
    raw = json.loads((out(root) / "QWEN_RAW_SCORE_PARITY.json").read_text(encoding="utf-8")) if (out(root) / "QWEN_RAW_SCORE_PARITY.json").is_file() else {}
    teacher = {str(seed): json.loads((out(root) / f"STEP0_PARITY_seed{seed}.json").read_text(encoding="utf-8")) for seed in SEEDS if (out(root) / f"STEP0_PARITY_seed{seed}.json").is_file()}
    payload = {"format_version": 1, "status": "pass" if raw.get("status") == "pass" and len(exact) == 6 and all(v.get("status") == "pass" for v in teacher.values()) else "partial", "raw_score_parity": raw, "teacher_score_parity": teacher, "student_exact_ann": exact, "completed_at_utc": now()}
    write_json(out(root) / "ANN_SCORE_PARITY.json", payload)
    return payload


def independent_metrics(root: Path) -> dict[str, Any]:
    base = out(root) / "full_lake"
    records = {}
    for metrics_path in base.glob("*/**/metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        records[str(metrics_path.relative_to(base))] = payload
    for metrics_path in (out(root) / "fixed_pool").glob("*/**/metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        records[str(metrics_path.relative_to(out(root) / "fixed_pool"))] = payload
    result = {"format_version": 1, "status": "complete", "records": records, "recorded_at_utc": now()}
    write_json(out(root) / "INDEPENDENT_METRICS.json", result)
    return result


def refresh_fixed_metrics(root: Path) -> dict[str, Any]:
    """Recompute fixed-pool aggregates, including implicit/explicit splits."""
    refreshed = 0
    for metrics_path in (out(root) / "fixed_pool").glob("*/**/metrics.json"):
        ranking_path = metrics_path.parent / "rankings.jsonl.gz"
        if not ranking_path.is_file():
            continue
        rows = list(read_rows(ranking_path))
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        for pool in ("natural", "direct100", "matched"):
            pool_rows = []
            for row in rows:
                item = dict(row["pools"][pool]); item["query_kind"] = row["query_kind"]; pool_rows.append(item)
            payload[pool] = _metric_rows(pool_rows)
        write_json(metrics_path, payload)
        refreshed += 1
    result = {"format_version": 1, "status": "complete", "refreshed": refreshed, "completed_at_utc": now()}
    write_json(out(root) / "FIXED_METRICS_REFRESH.json", result)
    return result


def mechanism_analysis(root: Path) -> dict[str, Any]:
    """Quantify evidence additions and source overlap without changing rankings."""
    result: dict[str, Any] = {"format_version": 1, "status": "complete", "generators": {}}
    for path in sorted((out(root) / "full_lake").glob("*/**/rankings.jsonl.gz")):
        rows = list(read_rows(path))
        if not rows:
            continue
        generator = path.relative_to(out(root) / "full_lake").parts[0]
        per = []
        for row in rows:
            direct, evidence = set(row["direct_ann"]), set(row["evidence_ann"])
            positives = set(row["positive_target_ids"])
            added = evidence - direct
            per.append({
                "query_kind": row["query_kind"],
                "direct_size": len(direct), "evidence_size": len(evidence), "u_size": len(set(row["U"])),
                "evidence_added": len(added), "jaccard": len(direct & evidence) / len(direct | evidence) if direct | evidence else 1.0,
                "direct_positive": len(positives & direct) / len(positives),
                "evidence_added_positive": len(positives & added) / len(positives),
                "u_positive": len(positives & set(row["U"])) / len(positives),
            })
        def agg(items: Sequence[dict[str, Any]]) -> dict[str, float]:
            keys = ("direct_size", "evidence_size", "u_size", "evidence_added", "jaccard", "direct_positive", "evidence_added_positive", "u_positive")
            return {key: statistics.fmean(float(item[key]) for item in items) for key in keys}
        result["generators"][str(path.relative_to(out(root) / "full_lake"))] = {
            "queries": len(per), "overall": agg(per),
            "by_query_kind": {kind: agg([item for item in per if item["query_kind"] == kind]) for kind in ("implicit", "explicit")},
            "rankings_sha256": checkpoint_fingerprint(path),
        }
    write_json(out(root) / "MECHANISM_ANALYSIS.json", result)
    return result


def matched_cardinality(root: Path) -> dict[str, Any]:
    """Apply the same per-query cardinality to all generator U sets."""
    base = out(root) / "full_lake"
    specs = [("Qwen-Raw", "raw", "Qwen-Raw/raw"), ("B13", "seed13_step001318", "B13/raw"),
             ("Ssup-13", "seed13_step001318", "Ssup/seed13_step001318"), ("Ssup-29", "seed29_step001318", "Ssup/seed29_step001318"),
             ("SKD-13", "seed13_step001318", "SKD/seed13_step001318"), ("SKD-29", "seed29_step001318", "SKD/seed29_step001318")]
    loaded = {}
    for label, subdir, teacher_rel in specs:
        generator = label.split("-")[0] if label.startswith(("Ssup-", "SKD-")) else label
        source = base / generator / subdir / "rankings.jsonl.gz"
        teacher_seed = 13 if label in {"Qwen-Raw", "B13", "Ssup-13", "SKD-13"} else 29
        teacher_dir = out(root) / "fixed_teacher" / generator / ("raw" if generator in {"Qwen-Raw", "B13"} else subdir) / f"teacher{teacher_seed}"
        teacher_path = teacher_dir / "rankings.jsonl.gz"
        if source.is_file() and teacher_path.is_file():
            loaded[label] = (list(read_rows(source)), list(read_rows(teacher_path)))
    if len(loaded) != len(specs):
        raise FileNotFoundError("matched-cardinality requires all full-lake and fixed-Teacher rankings")
    by_query = defaultdict(dict)
    for label, (source_rows, teacher_rows) in loaded.items():
        for row, teacher in zip(source_rows, teacher_rows, strict=True):
            by_query[str(row["query_id"])][label] = (row, teacher["ranking"])
    output_rows = []
    for query_id, entries in by_query.items():
        m = min(len(value[0]["U"]) for value in entries.values())
        query_kind = next(iter(entries.values()))[0]["query_kind"]
        positives = set(next(iter(entries.values()))[0]["positive_target_ids"])
        result = {"query_id": query_id, "query_kind": query_kind, "m": m, "generators": {}}
        for label, (row, teacher_ranking) in entries.items():
            selected = sorted(row["U"], key=lambda value: stable_hash([query_id, value]))[:m]
            selected_set = set(selected)
            ranking = [value for value in teacher_ranking if value in selected_set]
            result["generators"][label] = {"candidate_count": m, "raw_recall": len(positives & selected_set) / len(positives), **{f"recall@{k}": len(positives & set(ranking[:k])) / len(positives) for k in (10, 20, 50)}}
        output_rows.append(result)
    def aggregate(label: str, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
        return {key: statistics.fmean(row["generators"][label][key] for row in rows) for key in ("raw_recall", "recall@10", "recall@20", "recall@50")}
    labels = [spec[0] for spec in specs]
    metrics = {"format_version": 1, "status": "complete", "queries": len(output_rows), "m_mean": statistics.fmean(row["m"] for row in output_rows), "generators": {label: aggregate(label, output_rows) for label in labels}, "by_query_kind": {kind: {label: aggregate(label, [row for row in output_rows if row["query_kind"] == kind]) for label in labels} for kind in ("implicit", "explicit")}}
    destination = out(root) / "matched_cardinality"
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "rankings.jsonl.gz", output_rows)
    metrics["rankings"] = str((destination / "rankings.jsonl.gz").resolve()); metrics["completed_at_utc"] = now()
    write_json(destination / "metrics.json", metrics)
    return metrics


def training_candidate_analysis(root: Path) -> dict[str, Any]:
    """Summarize train-only candidate membership and D2 hard-negative scores."""
    rows = train_manifest_rows(root)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_relation[f"{row['source_type']}->{row['destination_type']}"].append(row)
    output: dict[str, Any] = {"format_version": 1, "status": "complete", "lists": len(rows), "scope": "frozen R20 D0 natural manifest; no GT is used", "relations": {}}
    for rel, rel_rows in sorted(by_relation.items()):
        widths = [len(row["candidate_ids"]) for row in rel_rows]
        positive_hits = []
        unique_pairs = set()
        for row in rel_rows:
            candidates = list(map(str, row["candidate_ids"]))
            unique_pairs.update((str(row["query_id"]), value) for value in candidates)
            positives = set(map(str, row.get("positive_ids", [])))
            positive_hits.append(len(positives & set(candidates)) / len(positives) if positives else 1.0)
        output["relations"][rel] = {"lists": len(rel_rows), "candidate_width_mean": statistics.fmean(widths), "candidate_width_min": min(widths), "candidate_width_max": max(widths), "known_positive_coverage": statistics.fmean(positive_hits), "unique_query_candidate_pairs": len(unique_pairs)}
    output["candidate_list_order_sha256"] = stable_hash([[str(row["query_id"]), list(map(str, row["candidate_ids"]))] for row in rows])
    output["completed_at_utc"] = now()
    write_json(out(root) / "TRAIN_CANDIDATE_ANALYSIS.json", output)
    return output


def statistical_comparisons(root: Path) -> dict[str, Any]:
    """Paired W/L/T and deterministic bootstrap intervals on saved rankings."""
    base = out(root) / "full_lake"
    sources = {
        "Qwen-Raw": base / "Qwen-Raw/raw/rankings.jsonl.gz",
        "B13": base / "B13/seed13_step001318/rankings.jsonl.gz",
        "Ssup-13": base / "Ssup/seed13_step001318/rankings.jsonl.gz",
        "SKD-13": base / "SKD/seed13_step001318/rankings.jsonl.gz",
        "Ssup-29": base / "Ssup/seed29_step001318/rankings.jsonl.gz",
        "SKD-29": base / "SKD/seed29_step001318/rankings.jsonl.gz",
    }
    values = {}
    for label, path in sources.items():
        values[label] = {str(row["query_id"]): row for row in read_rows(path)}
    comparisons = [("B13-QwenRaw", "B13", "Qwen-Raw"), ("Ssup13-B13", "Ssup-13", "B13"), ("SKD13-Ssup13", "SKD-13", "Ssup-13"), ("Ssup29-B13", "Ssup-29", "B13"), ("SKD29-Ssup29", "SKD-29", "Ssup-29")]
    rng = random.Random(SEED_HASH)
    output = {"format_version": 1, "status": "complete", "metric": "u_recall@10", "comparisons": {}}
    for name, left, right in comparisons:
        ids = sorted(set(values[left]) & set(values[right]))
        deltas = [float(values[left][qid]["u_recall@10"]) - float(values[right][qid]["u_recall@10"]) for qid in ids]
        wins = sum(delta > 0 for delta in deltas); losses = sum(delta < 0 for delta in deltas)
        boots = []
        for _ in range(1000):
            boots.append(statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas))
        boots.sort()
        output["comparisons"][name] = {"queries": len(ids), "wins": wins, "losses": losses, "ties": len(deltas) - wins - losses, "mean_delta": statistics.fmean(deltas), "bootstrap_95ci": [boots[25], boots[975]], "by_query_kind": {kind: {"mean_delta": statistics.fmean([deltas[i] for i, qid in enumerate(ids) if values[left][qid]["query_kind"] == kind])} for kind in ("implicit", "explicit")}}
    output["completed_at_utc"] = now()
    write_json(out(root) / "STATISTICAL_COMPARISONS.json", output)
    return output


def prepare_teacher_requests(root: Path) -> dict[str, Any]:
    """Materialize the full-lake candidate universe for Teacher feature caching."""
    destination = out(root) / "teacher_feature_requests.jsonl"
    ids: dict[str, dict[str, Any]] = {}
    # Include every query and candidate that any fixed-budget generator exposed.
    for ranking_path in (out(root) / "full_lake").glob("*/**/rankings.jsonl.gz"):
        for row in read_rows(ranking_path):
            query_id = str(row["query_id"])
            ids.setdefault(query_id, {"object_id": query_id})
            for candidate_id in row.get("U", []):
                ids.setdefault(str(candidate_id), {"object_id": str(candidate_id)})
    # Include the train manifest as well, so future fixed-T* diagnostics can
    # score all train-only hard-negative pairs with the same cache.
    for row in train_manifest_rows(root):
        ids.setdefault(str(row["query_id"]), {"object_id": str(row["query_id"])})
        for candidate_id in row["candidate_ids"]:
            ids.setdefault(str(candidate_id), {"object_id": str(candidate_id)})
    request_rows = []
    for object_id in sorted(ids):
        request_rows.append({"object_id": object_id})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in request_rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(destination)
    payload = {
        "format_version": 1,
        "status": "complete",
        "path": str(destination.resolve()),
        "objects": len(request_rows),
        "sha256": checkpoint_fingerprint(destination),
        "source_rankings": len(list((out(root) / "full_lake").glob("*/**/rankings.jsonl.gz"))),
        "completed_at_utc": now(),
    }
    write_json(out(root) / "TEACHER_FEATURE_REQUESTS.json", payload)
    return payload


@torch.inference_mode()
def evaluate_fixed_teacher(
    root: Path,
    generator: str,
    generator_seed: int,
    teacher_seed: int,
    device_name: str,
) -> dict[str, Any]:
    """Score each generator's fixed-budget U with one frozen D2 Teacher."""
    if generator not in ("Qwen-Raw", "B13", *ARMS):
        raise ValueError("unknown candidate generator")
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("fixed-T* evaluation requires CUDA")
    source_subdir = (
        "raw" if generator == "Qwen-Raw"
        else f"seed{13 if generator == 'B13' else generator_seed}_step{FINAL_STEP:06d}"
    )
    source = out(root) / "full_lake" / generator / source_subdir / "rankings.jsonl.gz"
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = out(root) / "fixed_teacher" / generator / (
        "raw" if generator in {"Qwen-Raw", "B13"}
        else f"seed{generator_seed}_step{FINAL_STEP:06d}"
    ) / f"teacher{teacher_seed}"
    metrics_path = destination / "metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    model, _payload = _load_teacher(root, teacher_seed, device)
    ps = paths(root)
    store = FeatureStore.from_path(
        ps["features"], cache_size=30000, teacher_paths=_feature_paths(ps)
    )
    scorer_cache: dict[str, torch.Tensor] = model.new_compression_cache()
    output_rows = []
    started = time.monotonic()
    for position, row in enumerate(read_rows(source), 1):
        candidate_ids = list(dict.fromkeys(map(str, row["U"])))
        scores = _score_id_pairs(
            model,
            [(str(row["query_id"]), candidate_id) for candidate_id in candidate_ids],
            store,
            device,
            batch_size=256,
            cache=scorer_cache,
        )
        ranking = [
            candidate_id
            for candidate_id, _score in sorted(
                zip(candidate_ids, scores, strict=True), key=lambda value: (-value[1], value[0])
            )
        ]
        positives = set(map(str, row["positive_target_ids"]))
        result = {
            "query_id": str(row["query_id"]),
            "query_kind": str(row["query_kind"]),
            "positive_target_ids": sorted(positives),
            "candidate_ids": candidate_ids,
            "ranking": ranking,
            "raw_recall": len(positives & set(candidate_ids)) / len(positives),
            "candidate_count": len(candidate_ids),
        }
        for k in (10, 20, 50):
            result[f"recall@{k}"] = len(positives & set(ranking[:k])) / len(positives)
        output_rows.append(result)
        if position % 100 == 0:
            print(json.dumps({"stage": "fixed_teacher", "generator": generator,
                              "generator_seed": generator_seed, "teacher_seed": teacher_seed,
                              "queries": position, "elapsed_seconds": time.monotonic() - started}), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    ranking_path = destination / "rankings.jsonl.gz"
    write_rows(ranking_path, output_rows)
    def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
        return {key: statistics.fmean(float(row[key]) for row in rows) for key in ("raw_recall", "recall@10", "recall@20", "recall@50")}
    metrics = {
        "format_version": 1, "status": "complete", "generator": generator,
        "generator_seed": generator_seed, "teacher_seed": teacher_seed,
        "teacher_checkpoint_sha256": TEACHER_SHA[teacher_seed],
        "source_rankings_sha256": checkpoint_fingerprint(source),
        "queries": len(output_rows), "u": aggregate(output_rows),
        "by_query_kind": {
            kind: aggregate([row for row in output_rows if row["query_kind"] == kind])
            for kind in ("implicit", "explicit")
        },
        "rankings": str(ranking_path.resolve()),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": now(),
    }
    write_json(metrics_path, metrics)
    return metrics


@torch.inference_mode()
def matched_control(
    root: Path,
    generator: str,
    generator_seed: int,
    device_name: str,
) -> dict[str, Any]:
    """Build M(q)=Direct top-|U(q)| and compare it with U under fixed T*."""
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("matched control requires CUDA")
    label = generator
    source_subdir = "raw" if generator == "Qwen-Raw" else f"seed{13 if generator == 'B13' else generator_seed}_step{FINAL_STEP:06d}"
    source = out(root) / "full_lake" / label / source_subdir / "rankings.jsonl.gz"
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = out(root) / "matched_control" / label / ("raw" if generator in {"Qwen-Raw", "B13"} else f"seed{generator_seed}_step{FINAL_STEP:06d}")
    metrics_path = destination / "metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    ps = paths(root)
    store = FeatureStore.from_path(ps["features"], cache_size=60000)
    corpus_sha = checkpoint_fingerprint(ps["corpus"])
    if generator == "Qwen-Raw":
        index_dir = out(root) / "indexes/Qwen-Raw"
        indices = RawEmbeddingANNIndices(store, index_dir, corpus_sha256=corpus_sha)
        model = None
    else:
        model, checkpoint = _load_student_checkpoint(root, generator, generator_seed, FINAL_STEP, device)
        index_dir = out(root) / "indexes" / generator / f"seed{generator_seed}" / f"step_{FINAL_STEP:06d}"
        indices = StudentANNIndices(model, store, index_dir, device=device, checkpoint_sha256=checkpoint_fingerprint(checkpoint), corpus_sha256=corpus_sha)
    rows = list(read_rows(source))
    max_k = max(len(row["U"]) for row in rows)
    direct_hits = indices.search_many([str(row["query_id"]) for row in rows], "table", max_k)
    teacher, _payload = _load_teacher(root, generator_seed, device)
    teacher_store = FeatureStore.from_path(ps["features"], cache_size=40000, teacher_paths=_feature_paths(ps))
    teacher_cache: dict[str, torch.Tensor] = teacher.new_compression_cache()
    output_rows = []
    for row, hits in zip(rows, direct_hits):
        u_ids = list(dict.fromkeys(map(str, row["U"])))
        m_ids = [str(value) for value, _score in hits[: len(u_ids)]]
        if model is None:
            def own_score(ids: Sequence[str]) -> dict[str, float]:
                query = store.embedding_features(str(row["query_id"])).embedding
                return {value: float(torch.dot(query, store.embedding_features(value).embedding)) for value in ids}
        else:
            def own_score(ids: Sequence[str]) -> dict[str, float]:
                return _score_target_ids(model, store, str(row["query_id"]), ids, device)
        u_own = own_score(u_ids); m_own = own_score(m_ids)
        t_pairs_u = [(str(row["query_id"]), value) for value in u_ids]
        t_pairs_m = [(str(row["query_id"]), value) for value in m_ids]
        t_u = dict(zip(u_ids, _score_id_pairs(teacher, t_pairs_u, teacher_store, device, batch_size=256, cache=teacher_cache), strict=True))
        t_m = dict(zip(m_ids, _score_id_pairs(teacher, t_pairs_m, teacher_store, device, batch_size=256, cache=teacher_cache), strict=True))
        positives = set(map(str, row["positive_target_ids"]))
        def summary(ids: Sequence[str], scores: dict[str, float]) -> dict[str, Any]:
            ranking = sorted(ids, key=lambda value: (-scores[value], value))
            return {"candidate_count": len(ids), "raw_recall": len(positives & set(ids)) / len(positives), **{f"recall@{k}": len(positives & set(ranking[:k])) / len(positives) for k in (10, 20, 50)}}
        output_rows.append({"query_id": str(row["query_id"]), "query_kind": row["query_kind"], "positive_target_ids": sorted(positives), "U": u_ids, "M": m_ids, "own_U": summary(u_ids, u_own), "own_M": summary(m_ids, m_own), "teacher_U": summary(u_ids, t_u), "teacher_M": summary(m_ids, t_m)})
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "rankings.jsonl.gz", output_rows)
    def aggregate(channel: str) -> dict[str, float]:
        return {key: statistics.fmean(row[channel][key] for row in output_rows) for key in ("raw_recall", "recall@10", "recall@20", "recall@50")}
    def aggregate_kind(channel: str, kind: str) -> dict[str, float]:
        return {key: statistics.fmean(row[channel][key] for row in output_rows if row["query_kind"] == kind) for key in ("raw_recall", "recall@10", "recall@20", "recall@50")}
    metrics = {"format_version": 1, "status": "complete", "generator": generator, "generator_seed": generator_seed, "teacher_seed": generator_seed, "queries": len(output_rows), "own_U": aggregate("own_U"), "own_M": aggregate("own_M"), "teacher_U": aggregate("teacher_U"), "teacher_M": aggregate("teacher_M"), "u_minus_m_teacher": {key: aggregate("teacher_U")[key] - aggregate("teacher_M")[key] for key in ("raw_recall", "recall@10", "recall@20", "recall@50")}, "by_query_kind": {kind: {"teacher_U": aggregate_kind("teacher_U", kind), "teacher_M": aggregate_kind("teacher_M", kind)} for kind in ("implicit", "explicit")}, "rankings": str((destination / "rankings.jsonl.gz").resolve()), "completed_at_utc": now()}
    write_json(metrics_path, metrics)
    return metrics


def finalize(root: Path) -> dict[str, Any]:
    destination = out(root)
    checks: dict[str, bool] = {}
    required = ["PLAN_FROZEN.md", "RESOLVED_CONFIG.json", "INPUT_MANIFEST.json", "ENVIRONMENT.json", "QWEN_RAW_FEATURE_MANIFEST.json", "QWEN_RAW_SCORE_PARITY.json", "ANN_SCORE_PARITY.json", "CODE_HASH_MANIFEST.json", "INDEPENDENT_METRICS.json", "MECHANISM_ANALYSIS.json", "TRAIN_CANDIDATE_ANALYSIS.json", "STATISTICAL_COMPARISONS.json", "matched_cardinality/metrics.json"]
    required += [f"STEP0_PARITY_seed{seed}.json" for seed in SEEDS]
    for name in required:
        checks[name] = (destination / name).is_file()
    for arm in ARMS:
        for seed in SEEDS:
            checks[f"{arm}/seed{seed}/train"] = (destination / arm / f"seed{seed}/config.json").is_file()
            checks[f"{arm}/seed{seed}/fixed"] = (destination / "fixed_pool" / arm / f"seed{seed}" / f"step_{FINAL_STEP:06d}/metrics.json").is_file()
            checks[f"{arm}/seed{seed}/full"] = (destination / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}/metrics.json").is_file()
    checks["Qwen-Raw/full"] = (destination / "full_lake/Qwen-Raw/raw/metrics.json").is_file()
    checks["B13/full"] = (destination / "full_lake/B13/seed13_step001318/metrics.json").is_file()
    checks["B13/fixed"] = (destination / "fixed_pool/B13/seed13/step_001318/metrics.json").is_file()
    checks["fixed-teacher/all"] = len(list((destination / "fixed_teacher").glob("*/**/metrics.json"))) == 8
    checks["matched-control/all"] = len(list((destination / "matched_control").glob("*/**/metrics.json"))) == 6
    checks["exact-direct/all"] = len(list((destination / "full_lake").glob("*/**/EXACT_ANN.json"))) == 6
    missing = [name for name, passed in checks.items() if not passed]
    payload = {"format_version": 1, "status": "complete" if not missing else "partial", "requirements": checks, "missing": missing, "stage2": "out_of_scope", "completed_at_utc": now()}
    write_json(destination / "COMPLETION_AUDIT.json", payload)
    write_json(destination / "FAILURE_NOTES.json", {"format_version": 1, "status": "none" if not missing else "open", "missing": missing, "notes": [], "completed_at_utc": now()})
    return payload


def write_results(root: Path) -> dict[str, Any]:
    """Write a compact, auditable narrative report from frozen JSON artifacts."""
    destination = out(root)
    lines = ["# Stage 1 R21 results", "", "All figures below are recomputed from the frozen 1,198-query test population and the artifacts in this directory.", "", "## Retrieval and evidence", "", "| Generator | Direct raw@100 | U raw recall | U R10 | U R20 | U CR50 | U size |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, subdir in [("Qwen-Raw", "raw"), ("B13", "seed13_step001318"), ("Ssup-13", "seed13_step001318"), ("Ssup-29", "seed29_step001318"), ("SKD-13", "seed13_step001318"), ("SKD-29", "seed29_step001318")]:
        generator = label.split("-")[0] if label.startswith(("Ssup-", "SKD-")) else label
        path = destination / "full_lake" / generator / subdir / "metrics.json"
        if not path.is_file():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"| {label} | {m['direct_raw@100']:.4f} | {m['u_raw_recall']:.4f} | {m['u_r10']:.4f} | {m['u_r20']:.4f} | {m['u_cr50']:.4f} | {m['u_size_mean']:.1f} |")
    lines += ["", "## Interpretation", "", "The Qwen-Raw and B13 baselines retain substantially higher direct and union recall than the newly trained Ssup/SKD projections. KD improves the student modestly in some exact/Teacher views but does not recover baseline retrieval quality. Evidence expansion should therefore be reported as a measured mechanism (added candidates and positive recovery), not as proof of an end-to-end gain.", "", "## Audit artifacts", "", "- `INDEPENDENT_METRICS.json`: metric recomputation index.", "- `MECHANISM_ANALYSIS.json`: direct/evidence overlap, additions, and positive recovery by query kind.", "- `matched_control/`: M(q)=Direct top-|U(q)| cardinality controls under own and frozen Teacher scoring.", "- `fixed_teacher/`: fixed-T* cross-generator diagnostics.", "- `COMPLETION_AUDIT.json`: required-artifact checklist.", ""]
    report_path = destination / "RESULTS.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    payload = {"format_version": 1, "status": "complete", "path": str(report_path.resolve()), "sha256": checkpoint_fingerprint(report_path), "completed_at_utc": now()}
    write_json(destination / "RESULTS_MANIFEST.json", payload)
    return payload


def code_manifest(root: Path) -> dict[str, Any]:
    destination = out(root)
    files = [Path(__file__), Path(__file__).with_name("run_stage1_r19.py"), Path(__file__).with_name("run_stage1_r18.py"), Path(__file__).with_name("mmdd_stage1") / "models.py", Path(__file__).with_name("mmdd_stage1") / "retrieval.py", Path(__file__).with_name("mmdd_stage1") / "scoring.py"]
    payload = {"format_version": 1, "status": "complete", "files": {str(path.relative_to(Path(__file__).parent)): {"path": str(path.resolve()), "sha256": checkpoint_fingerprint(path)} for path in files}, "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "recorded_at_utc": now()}
    write_json(destination / "CODE_HASH_MANIFEST.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("feature-manifest")
    cache = sub.add_parser("cache-teacher"); cache.add_argument("--seed", type=int, choices=SEEDS, required=True); cache.add_argument("--device", required=True); cache.add_argument("--batch-size", type=int, default=512)
    smoke_parser = sub.add_parser("smoke"); smoke_parser.add_argument("--seed", type=int, choices=SEEDS, required=True); smoke_parser.add_argument("--device", required=True)
    train_parser = sub.add_parser("train"); train_parser.add_argument("--arm", choices=ARMS, required=True); train_parser.add_argument("--seed", type=int, choices=SEEDS, required=True); train_parser.add_argument("--device", required=True); train_parser.add_argument("--feature-cache-size", type=int, default=24000)
    ev = sub.add_parser("evaluate-fixed"); ev.add_argument("--arm", choices=("B13", *ARMS), required=True); ev.add_argument("--seed", type=int, choices=SEEDS, required=True); ev.add_argument("--step", type=int, choices=(MID_STEP, FINAL_STEP), required=True); ev.add_argument("--device", required=True)
    idx = sub.add_parser("build-index"); idx.add_argument("--arm", choices=("Qwen-Raw", "B13", *ARMS), required=True); idx.add_argument("--seed", type=int, choices=SEEDS, default=13); idx.add_argument("--step", type=int, choices=(MID_STEP, FINAL_STEP), default=FINAL_STEP); idx.add_argument("--device", required=True)
    full = sub.add_parser("evaluate-full"); full.add_argument("--arm", choices=("Qwen-Raw", "B13", *ARMS), required=True); full.add_argument("--seed", type=int, choices=SEEDS, default=13); full.add_argument("--step", type=int, choices=(MID_STEP, FINAL_STEP), default=FINAL_STEP); full.add_argument("--device", required=True)
    exact = sub.add_parser("exact-direct"); exact.add_argument("--arm", choices=("Qwen-Raw", "B13", *ARMS), required=True); exact.add_argument("--seed", type=int, choices=SEEDS, default=13); exact.add_argument("--step", type=int, choices=(MID_STEP, FINAL_STEP), default=FINAL_STEP); exact.add_argument("--device", required=True); exact.add_argument("--query-batch-size", type=int, default=32)
    ft = sub.add_parser("fixed-teacher"); ft.add_argument("--generator", choices=("Qwen-Raw", "B13", *ARMS), required=True); ft.add_argument("--generator-seed", type=int, choices=SEEDS, default=13); ft.add_argument("--teacher-seed", type=int, choices=SEEDS, required=True); ft.add_argument("--device", required=True)
    parity = sub.add_parser("raw-parity"); parity.add_argument("--device", default="cuda:1"); parity.add_argument("--sample-count", type=int, default=512)
    tp = sub.add_parser("teacher-parity"); tp.add_argument("--seed", type=int, choices=SEEDS, required=True); tp.add_argument("--device", required=True); tp.add_argument("--sample-lists", type=int, default=8)
    sub.add_parser("environment"); sub.add_parser("ann-score-parity")
    sub.add_parser("prepare-teacher-requests"); sub.add_parser("independent-metrics"); sub.add_parser("refresh-fixed-metrics"); sub.add_parser("mechanism-analysis"); sub.add_parser("matched-cardinality"); sub.add_parser("training-candidate-analysis"); sub.add_parser("statistical-comparisons")
    mc = sub.add_parser("matched-control"); mc.add_argument("--generator", choices=("Qwen-Raw", "B13", *ARMS), required=True); mc.add_argument("--generator-seed", type=int, choices=SEEDS, default=13); mc.add_argument("--device", required=True)
    sub.add_parser("code-manifest"); sub.add_parser("report"); sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze": result = freeze(args.root)
    elif args.command == "feature-manifest": result = feature_manifest(args.root)
    elif args.command == "cache-teacher": result = cache_teacher(args.root, args.seed, args.device, args.batch_size)
    elif args.command == "smoke": result = smoke(args.root, args.seed, args.device)
    elif args.command == "train": result = train(args.root, args.arm, args.seed, args.device, args.feature_cache_size)
    elif args.command == "evaluate-fixed": result = evaluate_fixed(args.root, args.arm, args.seed, args.step, args.device)
    elif args.command == "build-index": result = build_index(args.root, args.arm, args.seed, args.step, args.device, raw=args.arm == "Qwen-Raw")
    elif args.command == "evaluate-full": result = evaluate_full_lake(args.root, args.arm, args.seed, args.step, args.device, raw=args.arm == "Qwen-Raw")
    elif args.command == "exact-direct": result = evaluate_exact_direct(args.root, args.arm, args.seed, args.step, args.device, raw=args.arm == "Qwen-Raw", query_batch_size=args.query_batch_size)
    elif args.command == "fixed-teacher": result = evaluate_fixed_teacher(args.root, args.generator, args.generator_seed, args.teacher_seed, args.device)
    elif args.command == "raw-parity": result = raw_parity(args.root, args.device, args.sample_count)
    elif args.command == "teacher-parity": result = teacher_parity(args.root, args.seed, args.device, args.sample_lists)
    elif args.command == "environment": result = environment_manifest(args.root)
    elif args.command == "ann-score-parity": result = ann_score_parity(args.root)
    elif args.command == "prepare-teacher-requests": result = prepare_teacher_requests(args.root)
    elif args.command == "independent-metrics": result = independent_metrics(args.root)
    elif args.command == "refresh-fixed-metrics": result = refresh_fixed_metrics(args.root)
    elif args.command == "mechanism-analysis": result = mechanism_analysis(args.root)
    elif args.command == "matched-cardinality": result = matched_cardinality(args.root)
    elif args.command == "training-candidate-analysis": result = training_candidate_analysis(args.root)
    elif args.command == "statistical-comparisons": result = statistical_comparisons(args.root)
    elif args.command == "matched-control": result = matched_control(args.root, args.generator, args.generator_seed, args.device)
    elif args.command == "code-manifest": result = code_manifest(args.root)
    elif args.command == "report": result = write_results(args.root)
    else: result = finalize(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
