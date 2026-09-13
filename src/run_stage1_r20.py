"""Execute the frozen R20 continuation/mining experiment.

The runner is intentionally stage-oriented.  ``freeze`` and ``prepare`` are
CPU-only and create auditable manifests; ``mine-d2`` and ``train`` require a
CUDA device.  No stage silently falls back to CPU for the six full runs.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_checkpoint
from mmdd_stage1.data import EdgeExample, load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import TeacherJoinabilityModel
from run_stage1_r18 import protect_known_positives
from run_stage1_r19 import (
    LOGICAL_BATCH_SIZE,
    _checkpoint_payload,
    _edge_loss,
    _evaluate_edges,
    _gradient_summary,
    _ranking_summary,
    _score_id_pairs,
    _teacher_paths,
    _write_jsonl_gz,
    backward_logical_batch,
    load_r19_checkpoint,
    state_dict_content_hash,
    stable_json_hash,
)
from run_stage1_r16 import paired_group_bootstrap

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
OUT_NAME = "stage1_optimization_r20_20260911"
ARMS = ("D0", "D1", "D2")
SEEDS = (13, 29)
EPOCHS = 2
UPDATES_PER_EPOCH = 5268
PARENT_STEP = 10536
FINAL_STEP = PARENT_STEP + EPOCHS * UPDATES_PER_EPOCH
LR = 5e-5
WEIGHT_DECAY = 0.01
MINING_SEED = 200911
EXPECTED = {
    "parent13": "ee8ad16cd3e31145211bb04bc0cc5ab128830d93fd705e1832cdc9f3581d8333",
    "parent29": "bc30bf817440e715946c73361780ca8708925cefe31a1ceacd483cde10e3890f",
    "long32": "ed32939668c9b29ba8837e10891f2aa9deec3bdcd3f204c0aa104865beddb025",
    "reservoir": "55d888b3cf19d35fa639f9838b213b49e606ef2398b95693c6539cf9884de484",
    "source": "4505a22f4f972ae2384b62a432feea5e563ca0e4e19039137c8949362aaf4bbf",
    "nested": "5f95a9b03b926a19e50adf789082e23d56e42ba340a2740147dd08872def4737",
    "candidates": "4186b5bdd436a14fa61b83c3c6127507c075fd6f16804c5cdb2d4a06e85a01d1",
    "features": "c32099430feca4dae5d2f8fbbae60f965e3b0353fdd62c62a7bc929ba24216e1",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def output(root: Path) -> Path:
    return root / "work" / OUT_NAME


def paths(root: Path) -> dict[str, Path]:
    r19 = root / "work/stage1_optimization_r19_20260911"
    r10 = root / "work/stage1_optimization_r10_20260907"
    r12 = root / "work/stage1_optimization_r12_20260908"
    r16 = root / "work/stage1_optimization_r16_20260910"
    return {
        "parent13": r19 / "C3/seed13/checkpoints/step_010536.pt",
        "parent29": r19 / "C3/seed29/checkpoints/step_010536.pt",
        "long32": r19 / "train_negative_manifest.natural_tt_long32.jsonl",
        "reservoir": r19 / "train_hard_reservoir.jsonl.gz",
        "source": r19 / "train_candidate_source_manifest.jsonl.gz",
        "nested": r19 / "nested_list_audit_per_tt.jsonl.gz",
        "candidates": r16 / "candidate_pools.jsonl.gz",
        "features": r10 / "features_qwen3_vl_embedding_8b",
        "train_original": r12 / "taskA_correctness/supervision/edge_lists.train_fit.jsonl",
        "dev": r12 / "taskA_correctness/supervision/edge_lists.dev.jsonl",
        "teacher_extra": r12 / "taskC_training/teacher_extra",
        "teacher_extra_matched_gpu0": r16 / "teacher_extra_matched_gpu0",
        "teacher_extra_matched_gpu1": r16 / "teacher_extra_matched_gpu1",
        "teacher_extra_edges_gpu0": r16 / "teacher_extra_edges_gpu0",
        "teacher_extra_edges_gpu1": r16 / "teacher_extra_edges_gpu1",
    }


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def row_record(example: EdgeExample) -> dict[str, Any]:
    positives = tuple(example.positive_ids) or (example.candidate_ids[example.positive_index],)
    return {
        "query_id": example.query_id,
        "source_type": example.source_type,
        "destination_type": example.destination_type,
        "candidate_ids": list(example.candidate_ids),
        "positive_id": example.candidate_ids[example.positive_index],
        "positive_ids": list(positives),
        "confirmed_labels": list(example.confirmed_labels) if example.confirmed_labels is not None else None,
        "dataset": example.dataset,
        "split": example.split,
        "protocol_bucket": "train_fit",
    }


def stable_priority(list_id: str, target_id: str) -> str:
    return hashlib.sha256(
        f"{MINING_SEED}\0{list_id}\0{target_id}".encode("utf-8")
    ).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    ps = paths(root)
    mismatches = {}
    for key, expected in EXPECTED.items():
        target = ps[key] / "manifest.jsonl" if key == "features" else ps[key]
        actual = checkpoint_fingerprint(target)
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    out = output(root)
    out.mkdir(parents=True, exist_ok=True)
    status = "pass" if not mismatches else "blocked_missing_input"
    manifest = {
        "format_version": 1,
        "status": status,
        "inputs": {
            key: {"path": str((ps[key] / "manifest.jsonl" if key == "features" else ps[key]).resolve()),
                  "sha256": checkpoint_fingerprint(ps[key] / "manifest.jsonl" if key == "features" else ps[key])}
            for key in EXPECTED if ps[key].exists()
        },
        "mismatches": mismatches,
        "verified_at_utc": now(),
    }
    protocol = {
        "format_version": 1,
        "status": "frozen" if status == "pass" else status,
        "plan": str((root / "mmdd_r19_review/R20_EXPERIMENT_PLAN.md").resolve()),
        "arms": list(ARMS), "lineages": list(SEEDS), "parent_step": PARENT_STEP,
        "epochs": EPOCHS, "updates_per_epoch": UPDATES_PER_EPOCH,
        "final_step": FINAL_STEP, "logical_batch_lists": LOGICAL_BATCH_SIZE,
        "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
        "positive_loss_mode": "sum_probability", "mining_seed": MINING_SEED,
        "main_comparisons": ["D2-D1", "D1-D0", "D0-R19-C3"],
        "stage2": "out_of_scope", "mismatches": mismatches,
    }
    write_json(out / "INPUT_MANIFEST.json", manifest)
    write_json(out / "PLAN_FROZEN.json", protocol)
    (out / "PLAN_FROZEN.md").write_text(
        (root / "mmdd_r19_review/R20_EXPERIMENT_PLAN.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    matrix = {"format_version": 1, "status": "planned", "jobs": [
        {"arm": arm, "seed": seed, "status": "pending", "device": None}
        for seed in SEEDS for arm in ARMS
    ], "gpu_assignment": {"cuda:0": ["D0/seed13", "D1/seed13", "D0/seed29"],
                           "cuda:1": ["D2/seed13", "D1/seed29", "D2/seed29"]}}
    write_json(out / "EXECUTION_MATRIX.json", matrix)
    (out / "RUN_COMMANDS.md").write_text(
        "# R20 commands\n\n"
        "```bash\npython src/run_stage1_r20.py --root . freeze\n"
        "python src/run_stage1_r20.py --root . prepare\n"
        "python src/run_stage1_r20.py --root . mine-d2 --lineage 13 --device cuda:0\n"
        "python src/run_stage1_r20.py --root . train --arm D0 --seed 13 --device cuda:0\n"
        "```\n", encoding="utf-8")
    return {"protocol": protocol, "input_manifest": manifest, "matrix": matrix}


def prepare(root: Path) -> dict[str, Any]:
    ps, out = paths(root), output(root)
    freeze_info = json.loads((out / "INPUT_MANIFEST.json").read_text())
    if freeze_info.get("status") != "pass":
        raise RuntimeError("R20 freeze gate did not pass")
    base = [row for row in read_jsonl(ps["long32"])]
    nested = list(read_jsonl(ps["nested"]))
    reservoirs = {str(row["query_id"]): row for row in read_jsonl(ps["reservoir"])}
    if len(base) != 42143 or len(nested) != 12041:
        raise RuntimeError(f"unexpected frozen list counts: base={len(base)} nested={len(nested)}")
    d0 = base
    d1: list[dict[str, Any]] = []
    changed = 0
    tt_seen = 0
    for index, row in enumerate(base):
        ids = [str(x) for x in row["candidate_ids"]]
        positives = set(str(x) for x in row.get("positive_ids", [row["positive_id"]]))
        if str(row["source_type"]) == "table" and str(row["destination_type"]) == "table":
            audit = nested[tt_seen]
            tt_seen += 1
            list_id = str(audit["list_id"])
            reservoir = reservoirs.get(str(row["query_id"]))
            if reservoir is None:
                raise RuntimeError(f"missing reservoir for {row['query_id']}")
            candidates = list(dict.fromkeys(map(str, reservoir.get("top50", []))))
            eligible = [x for x in candidates if x not in positives]
            eligible.sort(key=lambda x: (stable_priority(list_id, x), x))
            n = len(ids) - len(positives)
            if len(eligible) < n:
                raise RuntimeError(f"{list_id}: only {len(eligible)} eligible negatives for {n} slots")
            selected = iter(eligible[:n])
            new_ids = [x if x in positives else next(selected) for x in ids]
            if new_ids != ids:
                changed += 1
            row = dict(row)
            row["candidate_ids"] = new_ids
            row["positive_ids"] = [x for x in new_ids if x in positives]
            row["confirmed_labels"] = [1 if x in positives else None for x in new_ids]
        d1.append(row)
    if tt_seen != len(nested):
        raise RuntimeError(f"nested audit alignment mismatch: {tt_seen} != {len(nested)}")
    write_jsonl(out / "train_manifest_D0.jsonl", d0)
    write_jsonl(out / "train_manifest_D1.jsonl", d1)
    write_json(out / "PREPARE_AUDIT.json", {
        "format_version": 1, "status": "pass", "lists": len(base),
        "tt_lists": tt_seen, "d0_sha256": checkpoint_fingerprint(out / "train_manifest_D0.jsonl"),
        "d1_sha256": checkpoint_fingerprint(out / "train_manifest_D1.jsonl"),
        "d1_lists_changed": changed, "d1_negative_priority": "sha256(200911\\0+list_id+\\0+target_id)",
        "known_positive_protection": True, "dev_or_test_qrels_used": False,
        "completed_at_utc": now(),
    })
    matrix = json.loads((out / "EXECUTION_MATRIX.json").read_text())
    for job in matrix["jobs"]:
        job["status"] = "ready_for_gpu" if job["arm"] != "D2" else "pending_mining"
    write_json(out / "EXECUTION_MATRIX.json", matrix)
    return json.loads((out / "PREPARE_AUDIT.json").read_text())


def mine_d2(root: Path, lineage: int, device_name: str, batch_size: int, cache_size: int) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("R20 D2 mining requires a visible CUDA device; refusing CPU substitution")
    ps, out = paths(root), output(root)
    parent_path = ps[f"parent{lineage}"]
    device = torch.device(device_name)
    _arm, seed, step, model, _payload = load_r19_checkpoint(parent_path, device)
    if step != PARENT_STEP or seed != lineage:
        raise RuntimeError("parent checkpoint identity mismatch")
    model.eval()
    store = FeatureStore.from_path(
        ps["features"], cache_size=cache_size, teacher_paths=_teacher_paths(ps)
    )
    reservoirs = list(read_jsonl(ps["reservoir"]))
    nested = list(read_jsonl(ps["nested"]))
    base = list(read_jsonl(ps["long32"]))
    current_top50: dict[str, list[str]] = {}
    reservoir_path = out / f"refreshed_reservoir_lineage{lineage}.jsonl.gz"
    reservoir_tmp = reservoir_path.with_suffix(reservoir_path.suffix + ".tmp")
    monitor_ids = {
        str(row["query_id"])
        for row in sorted(
            reservoirs,
            key=lambda row: hashlib.sha256(
                f"r20-monitor\0{row['query_id']}".encode("utf-8")
            ).digest(),
        )[:256]
    }
    tt_index = 0
    output_rows = []
    cache = model.new_compression_cache()
    with gzip.open(reservoir_tmp, "wt", encoding="utf-8") as handle:
        for position, reservoir in enumerate(reservoirs, 1):
            query_id = str(reservoir["query_id"])
            positives = set(map(str, reservoir.get("known_positive_ids", [])))
            candidates = list(dict.fromkeys(map(str, reservoir.get("top50", []) + reservoir.get("remaining_natural", []))))
            candidates = [x for x in candidates if x not in positives]
            scores = _score_id_pairs(
                model, [(query_id, x) for x in candidates], store, device,
                batch_size=batch_size, cache=cache,
            )
            ordered_pairs = sorted(zip(candidates, scores, strict=True), key=lambda z: (-z[1], z[0]))
            top = [x for x, _score in ordered_pairs[:50]]
            current_top50[query_id] = top
            record = {
                "query_id": query_id,
                "known_positive_ids": sorted(positives),
                "top50": top,
                "top50_scores": [score for _x, score in ordered_pairs[:50]],
                "remaining_natural": [x for x, _score in ordered_pairs[50:]],
                "remaining_scores": [score for _x, score in ordered_pairs[50:]],
                "candidate_count": len(ordered_pairs),
                "monitor_query": query_id in monitor_ids,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if position % 100 == 0:
                print(json.dumps({"stage": "mine_d2", "lineage": lineage, "queries": position}), flush=True)
    reservoir_tmp.replace(reservoir_path)
    for row in base:
        if str(row["source_type"]) != "table" or str(row["destination_type"]) != "table":
            output_rows.append(row); continue
        audit = nested[tt_index]; tt_index += 1
        positives = set(map(str, row.get("positive_ids", [row["positive_id"]])))
        top50 = current_top50[str(row["query_id"])]
        eligible = sorted(top50, key=lambda x: (stable_priority(str(audit["list_id"]), x), x))
        n = len(row["candidate_ids"]) - len(positives)
        selected = iter(eligible[:n])
        ids = [x if x in positives else next(selected) for x in row["candidate_ids"]]
        new_row = dict(row); new_row["candidate_ids"] = ids; new_row["positive_ids"] = [x for x in ids if x in positives]; new_row["confirmed_labels"] = [1 if x in positives else None for x in ids]
        output_rows.append(new_row)
    path = out / f"train_manifest_D2_lineage{lineage}.jsonl"
    write_jsonl(path, output_rows)
    result = {"format_version": 1, "status": "pass", "lineage": lineage, "teacher": str(parent_path.resolve()), "teacher_sha256": checkpoint_fingerprint(parent_path), "manifest": str(path.resolve()), "manifest_sha256": checkpoint_fingerprint(path), "refreshed_reservoir": str(reservoir_path.resolve()), "refreshed_reservoir_sha256": checkpoint_fingerprint(reservoir_path), "monitor_queries": len(monitor_ids), "completed_at_utc": now()}
    write_json(out / f"MINE_D2_lineage{lineage}.json", result)
    return result


def train(root: Path, arm: str, seed: int, device_name: str, cache_size: int, microbatch: int) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("R20 training requires a visible CUDA device; refusing CPU substitution")
    out, ps = output(root), paths(root)
    manifest = out / ("train_manifest_D0.jsonl" if arm == "D0" else "train_manifest_D1.jsonl" if arm == "D1" else f"train_manifest_D2_lineage{seed}.jsonl")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    smoke_path = out / arm / f"seed{seed}" / "smoke_test.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text()).get("status") != "pass":
        raise RuntimeError(f"R20 correctness smoke gate missing/failed: {smoke_path}")
    examples, protected = protect_known_positives(load_edge_examples(manifest, split="train"))
    if len(examples) != 42143 or math.ceil(len(examples) / LOGICAL_BATCH_SIZE) != UPDATES_PER_EPOCH:
        raise RuntimeError("R20 list/update budget mismatch")
    device = torch.device(device_name)
    parent_path = ps[f"parent{seed}"]
    _arm, parent_seed, parent_step, model, payload = load_r19_checkpoint(parent_path, device)
    if parent_seed != seed or parent_step != PARENT_STEP:
        raise RuntimeError("R20 parent lineage/step mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor): state[key] = value.to(device)
    if payload.get("sampler_state", {}).get("random_state"):
        random.setstate(payload["sampler_state"]["random_state"])
    if payload.get("rng_state", {}).get("torch_cpu") is not None:
        torch.set_rng_state(payload["rng_state"]["torch_cpu"])
    parent_cuda_states = payload.get("rng_state", {}).get("torch_cuda", [])
    if not parent_cuda_states:
        raise RuntimeError("R20 parent is missing the CUDA RNG state")
    torch.cuda.set_rng_state(parent_cuda_states[0], device)
    store = FeatureStore.from_path(
        ps["features"], cache_size=cache_size, teacher_paths=_teacher_paths(ps)
    )
    run = out / arm / f"seed{seed}"; ckpt_dir = run / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    step0 = ckpt_dir / f"step_{PARENT_STEP:06d}.pt"
    torch.save(
        _checkpoint_payload(
            model, arm, seed, PARENT_STEP, optimizer=optimizer,
            sampler_state=payload["sampler_state"],
        ),
        step0,
    )
    history = []; cumulative = PARENT_STEP; started = time.monotonic(); rng = random.Random(); rng.setstate(payload["sampler_state"]["random_state"])
    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(examples))); rng.shuffle(order); losses = []; slots = 0
        model.train()
        for local, start in enumerate(range(0, len(order), LOGICAL_BATCH_SIZE), 1):
            logical = [examples[i] for i in order[start:start + LOGICAL_BATCH_SIZE]]
            optimizer.zero_grad(); loss, used = backward_logical_batch(model, logical, store, device, microbatch_lists=microbatch); optimizer.step()
            cumulative += 1; losses.append(loss); slots += used
            if local % 250 == 0 or start + len(logical) == len(order):
                print(json.dumps({"arm": arm, "seed": seed, "epoch": epoch, "epoch_step": local, "global_step": cumulative, "loss": statistics.fmean(losses[-250:]), "pair_slots": slots, "elapsed_seconds": time.monotonic() - started}), flush=True)
        ckpt = ckpt_dir / f"step_{cumulative:06d}.pt"
        torch.save(_checkpoint_payload(model, arm, seed, cumulative, optimizer=optimizer, sampler_state={"completed_epochs": epoch, "random_state": rng.getstate()}), ckpt)
        history.append({"epoch": epoch, "global_step": cumulative, "train_loss": statistics.fmean(losses), "pair_slots": slots, "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": checkpoint_fingerprint(ckpt), "elapsed_seconds": time.monotonic() - started})
        write_jsonl(run / "train_history.jsonl", history)
    result = {"format_version": 1, "status": "pass", "arm": arm, "seed": seed, "parent_step": PARENT_STEP, "final_step": cumulative, "parent_state_dict_content_hash": state_dict_content_hash(payload["state_dict"]), "step0_checkpoint": str(step0.resolve()), "step0_checkpoint_sha256": checkpoint_fingerprint(step0), "manifest_sha256": checkpoint_fingerprint(manifest), "protected": protected, "cuda_rng_mapping": {"saved_device": "cuda:0", "runtime_device": device_name}, "history": history, "completed_at_utc": now()}
    write_json(run / "config.json", result)
    return result


def smoke(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    """Run one discarded optimizer update from the exact resumed state."""
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("R20 smoke requires a visible CUDA device")
    out, ps = output(root), paths(root)
    manifest = out / ("train_manifest_D0.jsonl" if arm == "D0" else "train_manifest_D1.jsonl" if arm == "D1" else f"train_manifest_D2_lineage{seed}.jsonl")
    examples, _ = protect_known_positives(load_edge_examples(manifest, split="train"))
    device = torch.device(device_name)
    _parent_arm, parent_seed, parent_step, model, payload = load_r19_checkpoint(ps[f"parent{seed}"], device)
    if (parent_seed, parent_step) != (seed, PARENT_STEP):
        raise RuntimeError("R20 smoke parent mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    store = FeatureStore.from_path(ps["features"], cache_size=128, teacher_paths=_teacher_paths(ps))
    model.train(); optimizer.zero_grad()
    before = state_dict_content_hash(model.state_dict())
    loss, slots = backward_logical_batch(model, examples[:LOGICAL_BATCH_SIZE], store, device, microbatch_lists=2)
    optimizer.step()
    after = state_dict_content_hash(model.state_dict())
    result = {"format_version": 1, "status": "pass" if math.isfinite(loss) and before != after else "failed_correctness", "arm": arm, "seed": seed, "loss": loss, "pair_slots": slots, "model_updated": before != after, "optimizer_state_entries": len(optimizer.state), "device": device_name, "discarded_update": True, "completed_at_utc": now()}
    write_json(out / arm / f"seed{seed}" / "smoke_test.json", result)
    if result["status"] != "pass":
        raise RuntimeError("R20 smoke failed")
    return result


@torch.inference_mode()
def evaluate(
    root: Path,
    arm: str,
    seed: int,
    step: int,
    device_name: str,
    batch_size: int,
    cache_size: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("R20 evaluation requires a visible CUDA device")
    ps, out = paths(root), output(root)
    checkpoint = out / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
    saved_arm, saved_seed, saved_step, model, _payload = load_r19_checkpoint(
        checkpoint, torch.device(device_name)
    )
    if (saved_arm, saved_seed, saved_step) != (arm, seed, step):
        raise RuntimeError("R20 evaluation checkpoint identity mismatch")
    device = torch.device(device_name)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    store = FeatureStore.from_path(
        ps["features"], cache_size=cache_size, teacher_paths=_teacher_paths(ps)
    )
    cache = model.new_compression_cache()
    natural_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    latencies = []
    total_pairs = 0
    started = time.monotonic()
    for position, row in enumerate(read_jsonl(ps["candidates"]), 1):
        query_id = str(row["query_id"])
        combined = [str(x) for x in row["combined_score_candidate_ids"]]
        query_started = time.perf_counter()
        scores = _score_id_pairs(
            model, [(query_id, target) for target in combined], store, device,
            batch_size=batch_size, cache=cache,
        )
        torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - query_started)
        score_map = dict(zip(combined, scores, strict=True))
        total_pairs += len(combined)
        positives = [str(x) for x in row["positive_target_ids"]]
        common = {
            "query_id": query_id,
            "source_table_id": str(row["source_table_id"]),
            "query_kind": str(row["query_kind"]),
            "positive_target_ids": positives,
        }
        for records, key, label in (
            (natural_rows, "natural_candidate_ids", "natural"),
            (direct_rows, "ann_direct100_ids", "direct100"),
            (matched_rows, "matched_direct_candidate_ids", "matched"),
        ):
            ids = [str(x) for x in row[key]]
            ranking = sorted(ids, key=lambda target: (-score_map[target], target))
            id_set = set(ids)
            record = {
                **common, "pool": label, "candidate_ids": ids,
                "candidate_count": len(ids), "ranking": ranking,
                "raw_scores": [score_map[target] for target in ranking],
                "positive_ranks": {
                    target: ranking.index(target) + 1 for target in positives if target in id_set
                },
                f"{label}_raw_recall": len(set(positives) & id_set) / len(set(positives)),
            }
            for k in (10, 20, 50):
                record[f"{label}_recall@{k}"] = len(set(positives) & set(ranking[:k])) / len(set(positives))
            records.append(record)
        if position % 50 == 0:
            print(json.dumps({"arm": arm, "seed": seed, "step": step, "queries": position, "pairs": total_pairs, "elapsed_seconds": time.monotonic() - started}), flush=True)
    eval_out = out / arm / f"seed{seed}" / f"eval_step{step:06d}"
    eval_out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("natural_union.jsonl.gz", natural_rows), ("direct100.jsonl.gz", direct_rows), ("matched_direct_M.jsonl.gz", matched_rows)):
        _write_jsonl_gz(eval_out / name, rows)
    dev_examples, _ = protect_known_positives(load_edge_examples(ps["dev"], split="dev"))
    dev, dev_rows = _evaluate_edges(model, dev_examples, store, device, include_rows=True)
    _write_jsonl_gz(eval_out / "edge_dev_per_list.jsonl.gz", dev_rows)
    metrics = {
        "format_version": 1, "status": "complete", "arm": arm, "seed": seed,
        "global_step": step, "local_step": step - PARENT_STEP,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "natural_union": _ranking_summary(natural_rows, "natural"),
        "direct100": _ranking_summary(direct_rows, "direct100"),
        "matched_direct_M": _ranking_summary(matched_rows, "matched"),
        "edge_dev": dev,
        "runtime": {"pairs": total_pairs, "elapsed_seconds": time.monotonic() - started,
                    "per_query_seconds_p50": statistics.median(latencies),
                    "per_query_seconds_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                    "cache_objects": len(cache), "device": device_name},
        "completed_at_utc": now(),
    }
    write_json(eval_out / "metrics.json", metrics)
    return metrics


def _natural_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in read_jsonl(path)}


def compare(root: Path) -> dict[str, Any]:
    """Compute pre-registered query-macro comparisons once endpoint files exist."""
    out = output(root)
    endpoint_step = FINAL_STEP
    arm_seed: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    missing = []
    for arm in ARMS:
        for seed in SEEDS:
            path = out / arm / f"seed{seed}" / f"eval_step{endpoint_step:06d}" / "natural_union.jsonl.gz"
            if path.is_file():
                arm_seed[(arm, seed)] = _natural_rows(path)
            else:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing R20 endpoint rankings: " + ", ".join(missing))
    keys = sorted(next(iter(arm_seed.values())))
    mean_rows: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for query_id in keys:
            first, second = arm_seed[(arm, 13)][query_id], arm_seed[(arm, 29)][query_id]
            mean_rows[f"{arm}:{query_id}"] = {
                "query_id": query_id,
                "source_table_id": first["source_table_id"],
                "query_kind": first["query_kind"],
                "r10": statistics.fmean((first["natural_recall@10"], second["natural_recall@10"])),
                "r20": statistics.fmean((first["natural_recall@20"], second["natural_recall@20"])),
                "cr50": statistics.fmean((first["natural_recall@50"], second["natural_recall@50"])),
            }
    comparisons = {}
    def add_comparison(name: str, left_rows: dict[str, dict[str, Any]], right_rows: dict[str, dict[str, Any]]) -> None:
        rows = []
        for query_id in keys:
            l, r = left_rows[query_id], right_rows[query_id]
            rows.append({"query_id": query_id, "source_table_id": l["source_table_id"], "query_kind": l["query_kind"], "left_r10": l["r10"], "right_r10": r["r10"], "left_r20": l["r20"], "right_r20": r["r20"], "left_cr50": l["cr50"], "right_cr50": r["cr50"]})
        comparisons[name] = {
            "r10": paired_group_bootstrap(rows, "left_r10", "right_r10", seed=MINING_SEED),
            "r20": paired_group_bootstrap(rows, "left_r20", "right_r20", seed=MINING_SEED),
            "cr50": paired_group_bootstrap(rows, "left_cr50", "right_cr50", seed=MINING_SEED),
            "wlt_r10": {"win": sum(x["left_r10"] > x["right_r10"] for x in rows), "loss": sum(x["left_r10"] < x["right_r10"] for x in rows), "tie": sum(x["left_r10"] == x["right_r10"] for x in rows)},
            "per_query": rows,
        }
    for left, right in (("D2", "D1"), ("D1", "D0"), ("D2", "D0")):
        add_comparison(f"{left}-{right}",
                       {q: mean_rows[f"{left}:{q}"] for q in keys},
                       {q: mean_rows[f"{right}:{q}"] for q in keys})
    parent_rows = {}
    for seed in SEEDS:
        source = root / "work/stage1_optimization_r19_20260911" / "C3" / f"seed{seed}" / "eval_step010536" / "natural_union.jsonl.gz"
        parent_rows[seed] = _natural_rows(source)
    parent_mean = {
        q: {"query_id": q, "source_table_id": parent_rows[13][q]["source_table_id"], "query_kind": parent_rows[13][q]["query_kind"],
            "r10": statistics.fmean((parent_rows[13][q]["natural_recall@10"], parent_rows[29][q]["natural_recall@10"])),
            "r20": statistics.fmean((parent_rows[13][q]["natural_recall@20"], parent_rows[29][q]["natural_recall@20"])),
            "cr50": statistics.fmean((parent_rows[13][q]["natural_recall@50"], parent_rows[29][q]["natural_recall@50"]))}
        for q in keys
    }
    add_comparison("D0-R19-C3", {q: mean_rows[f"D0:{q}"] for q in keys}, parent_mean)
    b13_path = root / "work/stage1_optimization_r19_20260911" / "baseline_reference" / "B13_per_query.jsonl.gz"
    b13 = {str(row["query_id"]): row for row in read_jsonl(b13_path)}
    b13_mean = {q: {"query_id": q, "source_table_id": b13[q]["source_table_id"], "query_kind": b13[q]["query_kind"], "r10": b13[q]["recall@10"], "r20": b13[q]["recall@20"], "cr50": b13[q]["recall@50"]} for q in keys}
    for arm in ARMS:
        add_comparison(f"{arm}-B13", {q: mean_rows[f"{arm}:{q}"] for q in keys}, b13_mean)
    metrics = {
        arm: {
            str(seed): json.loads((out / arm / f"seed{seed}" / f"eval_step{endpoint_step:06d}" / "metrics.json").read_text())
            for seed in SEEDS
        }
        for arm in ARMS
    }
    result = {"format_version": 1, "status": "complete", "endpoint_step": endpoint_step, "comparisons": comparisons, "metrics": metrics, "completed_at_utc": now()}
    write_json(out / "PAIRED_COMPARISONS.json", result)
    return result


def audit_refresh(root: Path, lineage: int) -> dict[str, Any]:
    """Summarize old/current reservoir overlap and source composition."""
    ps, out = paths(root), output(root)
    refreshed_path = out / f"refreshed_reservoir_lineage{lineage}.jsonl.gz"
    if not refreshed_path.is_file():
        raise FileNotFoundError(refreshed_path)
    source_rows = {str(row["query_id"]): row for row in read_jsonl(ps["source"])}
    old_rows = {str(row["query_id"]): row for row in read_jsonl(ps["reservoir"])}
    refreshed = {str(row["query_id"]): row for row in read_jsonl(refreshed_path)}
    overlaps10, overlaps50, outside50 = [], [], []
    source_counts = {"direct": 0, "evidence": 0, "both": 0, "unknown": 0}
    for query_id, new in refreshed.items():
        old = old_rows[query_id]
        old10, old50 = set(map(str, old.get("top50", [])[:10])), set(map(str, old.get("top50", [])))
        new10, new50 = set(map(str, new.get("top50", [])[:10])), set(map(str, new.get("top50", [])))
        overlaps10.append(len(old10 & new10) / max(1, len(old10 | new10)))
        overlaps50.append(len(old50 & new50) / max(1, len(old50 | new50)))
        outside50.append(len(new50 - old50) / max(1, len(new50)))
        metadata = {str(item["target_id"]): item for item in source_rows[query_id].get("candidates", [])}
        for target_id in new50:
            source_counts[str(metadata.get(target_id, {}).get("source", "unknown"))] = source_counts.get(str(metadata.get(target_id, {}).get("source", "unknown")), 0) + 1
    result = {
        "format_version": 1, "status": "complete", "lineage": lineage,
        "queries": len(refreshed), "old_new_top10_jaccard_mean": statistics.fmean(overlaps10),
        "old_new_top50_jaccard_mean": statistics.fmean(overlaps50),
        "new_top50_outside_old_fraction_mean": statistics.fmean(outside50),
        "new_top50_source_memberships": source_counts,
        "refreshed_reservoir_sha256": checkpoint_fingerprint(refreshed_path),
        "completed_at_utc": now(),
    }
    write_json(out / f"MINING_DIAGNOSTICS_lineage{lineage}.json", result)
    return result


def finalize(root: Path) -> dict[str, Any]:
    out = output(root)
    comparison_path = out / "PAIRED_COMPARISONS.json"
    if not comparison_path.is_file():
        compare(root)
    comparison = json.loads(comparison_path.read_text())
    checks: dict[str, bool] = {}
    missing: list[str] = []
    for arm in ARMS:
        for seed in SEEDS:
            base = out / arm / f"seed{seed}"
            for name in ("smoke_test.json", "config.json"):
                path = base / name
                checks[f"{arm}/seed{seed}/{name}"] = path.is_file() and json.loads(path.read_text()).get("status") == "pass"
                if not checks[f"{arm}/seed{seed}/{name}"]:
                    missing.append(str(path))
            for step in (15804, 21072):
                path = base / f"eval_step{step:06d}/metrics.json"
                checks[f"{arm}/seed{seed}/eval{step}"] = path.is_file()
                if not checks[f"{arm}/seed{seed}/eval{step}"]:
                    missing.append(str(path))
    for lineage in SEEDS:
        for name in (f"MINE_D2_lineage{lineage}.json", f"MINING_DIAGNOSTICS_lineage{lineage}.json"):
            path = out / name
            checks[name] = path.is_file() and json.loads(path.read_text()).get("status") in {"complete", "pass"} if path.is_file() else False
            if not checks[name]:
                missing.append(str(path))
    for name in ("PLAN_FROZEN.json", "INPUT_MANIFEST.json", "PREPARE_AUDIT.json", "CODE_HASH_MANIFEST.json", "PARENT_EVAL_REFERENCES.json"):
        checks[name] = (out / name).is_file()
        if not checks[name]:
            missing.append(str(out / name))
    audit = {
        "format_version": 1,
        "status": "complete" if not missing else "partial",
        "requirements": checks,
        "missing": missing,
        "step0_parent_reused_from_r19": True,
        "stage2": "out_of_scope",
        "completed_at_utc": now(),
    }
    write_json(out / "COMPLETION_AUDIT.json", audit)
    failures = {
        "format_version": 1,
        "status": "none" if not missing else "open",
        "notes": [
            "GPU1 had an unrelated 22.3 GiB process during initial scheduling; R20 lineage29 D2 was moved there after it exited.",
            "Training used feature-cache-size=12000 uniformly across all six jobs to fit concurrent 4090 runs; listwise objective and pair budget were unchanged.",
        ],
        "missing": missing,
        "completed_at_utc": now(),
    }
    write_json(out / "FAILURE_NOTES.json", failures)
    lines = ["# R20 Results", "", f"Status: **{audit['status']}**", "", "Endpoint: natural-union query-macro target recall at global step 21072; two lineages (13, 29), no ensemble.", "", "| Arm | R@10 mean | R@20 mean | CR@50 mean |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        values = [comparison["metrics"][arm][str(seed)]["natural_union"]["all"] for seed in SEEDS]
        lines.append(f"| {arm} | {100*statistics.fmean(v['recall@10'] for v in values):.4f}% | {100*statistics.fmean(v['recall@20'] for v in values):.4f}% | {100*statistics.fmean(v['CandidateRecall@50'] for v in values):.4f}% |")
    lines.extend(["", "Pre-registered comparisons (source-group bootstrap, seed=200911):"])
    for name in ("D2-D1", "D1-D0", "D2-D0"):
        item = comparison["comparisons"][name]["r10"]
        lines.append(f"- {name} R@10: {100*item['observed_delta']:+.4f}pp, 95% CI [{100*item['ci95'][0]:+.4f}, {100*item['ci95'][1]:+.4f}]pp")
    lines.extend(["", "Step0 is the frozen R19 C3 parent checkpoint; step15804 and step21072 were evaluated independently for each arm/lineage. Stage2, KD, candidate generation, and online fusion changes remain out of scope."])
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    matrix = json.loads((out / "EXECUTION_MATRIX.json").read_text())
    for job in matrix["jobs"]:
        arm, seed = job["arm"], job["seed"]
        base = out / arm / f"seed{seed}"
        job["status"] = "complete" if (base / "config.json").is_file() and json.loads((base / "config.json").read_text()).get("status") == "pass" else "running"
        job["device"] = "cuda:1" if arm == "D2" and seed == 29 else "cuda:0"
    matrix["status"] = audit["status"]
    write_json(out / "EXECUTION_MATRIX.json", matrix)
    return audit


def materialize_parent_refs(root: Path) -> dict[str, Any]:
    """Record the frozen R19 C3 endpoint as R20 local_step=0 (no rescoring)."""
    out = output(root)
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            source = root / "work/stage1_optimization_r19_20260911" / "C3" / f"seed{seed}" / "eval_step010536"
            target = out / arm / f"seed{seed}" / "eval_step010536"
            target.mkdir(parents=True, exist_ok=True)
            payload = {
                "format_version": 1,
                "status": "reused_frozen_parent",
                "arm": arm,
                "seed": seed,
                "global_step": PARENT_STEP,
                "local_step": 0,
                "source_arm": "C3",
                "source_dir": str(source.resolve()),
                "source_metrics_sha256": checkpoint_fingerprint(source / "metrics.json"),
                "candidate_pool_sha256": EXPECTED["candidates"],
                "note": "Parent state and candidate pool are identical; endpoint was not rescored.",
                "completed_at_utc": now(),
            }
            write_json(target / "PARENT_REFERENCE.json", payload)
            records.append(payload)
    write_json(out / "PARENT_EVAL_REFERENCES.json", {"format_version": 1, "status": "complete", "records": records, "completed_at_utc": now()})
    return {"format_version": 1, "status": "complete", "records": len(records)}


def write_code_manifest(root: Path) -> dict[str, Any]:
    out = output(root)
    files = {
        "run_stage1_r20.py": Path(__file__),
        "run_stage1_r19.py": Path(__file__).with_name("run_stage1_r19.py"),
        "run_stage1_r18.py": Path(__file__).with_name("run_stage1_r18.py"),
        "mmdd_stage1/scoring.py": Path(__file__).with_name("mmdd_stage1") / "scoring.py",
        "mmdd_stage1/objectives.py": Path(__file__).with_name("mmdd_stage1") / "objectives.py",
    }
    records = {name: {"path": str(path.resolve()), "sha256": checkpoint_fingerprint(path)} for name, path in files.items()}
    payload = {"format_version": 1, "status": "complete", "files": records, "completed_at_utc": now()}
    write_json(out / "CODE_HASH_MANIFEST.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze"); sub.add_parser("prepare")
    mine = sub.add_parser("mine-d2"); mine.add_argument("--lineage", type=int, choices=SEEDS, required=True); mine.add_argument("--device", required=True); mine.add_argument("--batch-size", type=int, default=512); mine.add_argument("--cache-size", type=int, default=40000)
    tr = sub.add_parser("train"); tr.add_argument("--arm", choices=ARMS, required=True); tr.add_argument("--seed", type=int, choices=SEEDS, required=True); tr.add_argument("--device", required=True); tr.add_argument("--feature-cache-size", type=int, default=24000); tr.add_argument("--microbatch-lists", type=int, default=2)
    sm = sub.add_parser("smoke"); sm.add_argument("--arm", choices=ARMS, required=True); sm.add_argument("--seed", type=int, choices=SEEDS, required=True); sm.add_argument("--device", required=True)
    ev = sub.add_parser("evaluate"); ev.add_argument("--arm", choices=ARMS, required=True); ev.add_argument("--seed", type=int, choices=SEEDS, required=True); ev.add_argument("--step", type=int, choices=(15804, 21072), required=True); ev.add_argument("--device", required=True); ev.add_argument("--batch-size", type=int, default=512); ev.add_argument("--feature-cache-size", type=int, default=40000)
    sub.add_parser("compare")
    au = sub.add_parser("audit-refresh"); au.add_argument("--lineage", type=int, choices=SEEDS, required=True)
    sub.add_parser("finalize")
    sub.add_parser("materialize-parent-refs")
    sub.add_parser("code-manifest")
    args = parser.parse_args()
    if args.command == "freeze": result = freeze(args.root)
    elif args.command == "prepare": result = prepare(args.root)
    elif args.command == "mine-d2": result = mine_d2(args.root, args.lineage, args.device, args.batch_size, args.cache_size)
    elif args.command == "smoke": result = smoke(args.root, args.arm, args.seed, args.device)
    elif args.command == "evaluate": result = evaluate(args.root, args.arm, args.seed, args.step, args.device, args.batch_size, args.feature_cache_size)
    elif args.command == "compare": result = compare(args.root)
    elif args.command == "audit-refresh": result = audit_refresh(args.root, args.lineage)
    elif args.command == "finalize": result = finalize(args.root)
    elif args.command == "materialize-parent-refs": result = materialize_parent_refs(args.root)
    elif args.command == "code-manifest": result = write_code_manifest(args.root)
    else: result = train(args.root, args.arm, args.seed, args.device, args.feature_cache_size, args.microbatch_lists)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
