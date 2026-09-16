"""Materialize and train the gated R30 F-P-ETNAT arm.

Only text-to-table unknown candidate IDs in Bridge C1 batches 357--659 are
replaced.  Positive slots, list widths, occurrence order, all other relations,
and the frozen-P optimizer continuation remain unchanged.  New KD logits are
scored by the historical C1 T_core, never by online T0.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

import run_stage1_r30 as fp
from mmdd_stage1.checkpoints import load_student, load_teacher
from mmdd_stage1.data import EdgeExample
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.scoring import score_edge_batch
from mmdd_stage1.training import checkpoint, student_gradient_norms, student_projection_references
from run_stage1_bridge import ROOT, sha256, stable_sha, write_json
from run_stage1_r12_task_c import _optimizer
from run_stage1_r25 import _r25_teacher_feature_paths


OUT = ROOT / "work/stage1_r30_c1_et_20260916"
BRIDGE = ROOT / "work/stage1_bridge_20260915"
FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
T_CORE = ROOT / "work/stage1_optimization_r11_20260908/taskC_clean/teacher/teacher_edge.pt"
T_CORE_SHA = "fd544cc166f2a24f2c55a040c0a2645bc14b22c3c82b49176823db458fe75bc1"
SAVE_STEPS = (357, 428, 500, 580, 659)


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    temporary = path.with_suffix(path.suffix + ".tmp")
    with opener(temporary, "wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def stage_dir(seed: int) -> Path:
    return OUT / f"C1/F-P-ETNAT/seed{seed}"


def require_gates() -> dict[str, Any]:
    paths = {"G-F": OUT / "C1/F-P/G_F.json", "G-ET": OUT / "diagnostics_repaired/G_ET.json"}
    values = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"{name} has not been evaluated: {path}")
        values[name] = json.loads(path.read_text())
        if values[name].get("status") != "pass":
            raise RuntimeError(f"{name} did not pass; F-P-ETNAT is not triggered")
    return {name: {"path": str(path.resolve()), "sha256": sha256(path), "status": values[name]["status"]} for name, path in paths.items()}


def _historical_scores(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], float], float]:
    rows = list(fp._teacher_scores(path))
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for candidate_id, score in zip(row["candidate_ids"], row["scores"], strict=True):
            values[(row["query_id"], row["relation"], candidate_id)].append(float(score))
    spread = max((max(scores) - min(scores) for scores in values.values()), default=0.0)
    return rows, {key: scores[0] for key, scores in values.items()}, spread


def _score_t_core_pairs(
    pairs: set[tuple[str, str]],
    audit_pairs: list[tuple[str, str]],
    device: torch.device,
    microbatch: int,
) -> dict[tuple[str, str], float]:
    teacher = load_teacher(T_CORE, device).eval()
    store = FeatureStore.from_path(
        FEATURES,
        cache_size=24_000,
        teacher_paths=_r25_teacher_feature_paths(ROOT),
    )
    all_pairs = sorted(pairs | set(audit_pairs))
    missing_features = sorted({value for pair in all_pairs for value in pair if not store.has_teacher_features(value)})
    if missing_features:
        raise RuntimeError(f"T_core feature coverage missing for {len(missing_features)} objects")
    grouped: dict[str, list[str]] = defaultdict(list)
    for source_id, candidate_id in all_pairs:
        grouped[source_id].append(candidate_id)
    examples = []
    for source_id, candidates in sorted(grouped.items()):
        ordered = sorted(set(candidates))
        for start in range(0, len(ordered), 128):
            part = tuple(ordered[start:start + 128])
            examples.append(EdgeExample(
                query_id=source_id,
                candidate_ids=part,
                positive_index=0,
                dataset="r30_etnat_tcore",
                split="train",
                source_type="text",
                destination_type="table",
                positive_ids=(),
            ))
    result: dict[tuple[str, str], float] = {}
    with torch.inference_mode():
        for start in range(0, len(examples), microbatch):
            batch = examples[start:start + microbatch]
            scores = score_edge_batch(teacher, batch, store, device, student_score_space="raw_logit")
            for index, example in enumerate(batch):
                for offset, candidate_id in enumerate(example.candidate_ids):
                    result[(example.query_id, candidate_id)] = float(scores.logits[index, offset].cpu())
            if start % (microbatch * 100) == 0:
                print(json.dumps({"event": "tcore_pairs", "completed_lists": min(start + microbatch, len(examples)), "total_lists": len(examples)}), flush=True)
    return result


def materialize(seed: int, device_name: str, microbatch: int) -> dict[str, Any]:
    gates = require_gates()
    directory = stage_dir(seed)
    receipt_path = directory / "MATERIALIZATION.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text())
        if existing.get("status") == "completed":
            return existing
        raise FileExistsError(f"inspect incomplete ETNAT materialization: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if sha256(T_CORE) != T_CORE_SHA:
        raise ValueError("historical C1 T_core identity changed")
    schedule_path = BRIDGE / f"schedules/seed{seed}_steps659/closure_full.jsonl.gz"
    historical_teacher = ROOT / f"work/stage1_optimization_r25_final_20260914/common/teacher_edge_cache_seed{seed}.jsonl.gz"
    reservoir_path = OUT / f"diagnostics_repaired/natural_text_et_reservoir_seed{seed}.jsonl.gz"
    batches = list(read_rows(schedule_path))
    historical_rows, historical_map, maximum_historical_spread = _historical_scores(historical_teacher)
    reservoirs = {row["source_id"]: row for row in read_rows(reservoir_path)}
    if len(batches) != 659:
        raise ValueError("locked C1 schedule must contain 659 batches")

    modified_batches = []
    old_occurrence = 0
    changed_occurrences = changed_slots = 0
    missing_pairs: set[tuple[str, str]] = set()
    audit_pairs: list[tuple[str, str]] = []
    row_specs = []
    unaffected_errors = positive_slot_errors = width_errors = protection_errors = 0
    for batch in batches:
        step = int(batch["step"])
        modified_examples = []
        for raw in batch["examples"]:
            teacher_row = historical_rows[old_occurrence]
            old_occurrence += 1
            relation = f"{raw['source_type']}->{raw['destination_type']}"
            source_id = str(raw["query_id"])
            old_ids = [str(value) for value in raw["candidate_ids"]]
            if teacher_row["query_id"] != source_id or teacher_row["relation"] != relation or teacher_row["candidate_ids"] != old_ids:
                raise ValueError(f"historical Teacher cache occurrence mismatch at {old_occurrence}")
            new = dict(raw)
            new_ids = list(old_ids)
            labels = None if raw.get("confirmed_labels") is None else list(raw["confirmed_labels"])
            positives = set(str(value) for value in raw.get("positive_ids", ())) | {str(raw["positive_id"])}
            if step >= 357 and relation == "text->table":
                legal = [str(item["candidate_id"]) for item in reservoirs[source_id]["exact_top256_legal_unknown"]]
                if set(legal) & positives:
                    protection_errors += len(set(legal) & positives)
                used = set(positives)
                cursor = 0
                for index, old_id in enumerate(old_ids):
                    if old_id in positives:
                        continue
                    while cursor < len(legal) and legal[cursor] in used:
                        cursor += 1
                    if cursor >= len(legal):
                        raise RuntimeError(f"Natural reservoir cannot preserve width for {source_id}")
                    replacement = legal[cursor]
                    cursor += 1
                    used.add(replacement)
                    new_ids[index] = replacement
                    if labels is not None and replacement != old_id:
                        labels[index] = None
                changed = sum(a != b for a, b in zip(old_ids, new_ids, strict=True))
                changed_occurrences += bool(changed)
                changed_slots += changed
                if len(new_ids) != len(old_ids):
                    width_errors += 1
                if any((old_id in positives) and new_ids[index] != old_id for index, old_id in enumerate(old_ids)):
                    positive_slot_errors += 1
                new["candidate_ids"] = new_ids
                if labels is not None:
                    new["confirmed_labels"] = labels
            elif new_ids != old_ids:
                unaffected_errors += 1
            for candidate_id in new_ids:
                key = (source_id, relation, candidate_id)
                if key not in historical_map:
                    if relation != "text->table" or step < 357:
                        unaffected_errors += 1
                    missing_pairs.add((source_id, candidate_id))
            if relation == "text->table" and step >= 357 and len(audit_pairs) < 256:
                for candidate_id in new_ids:
                    if (source_id, relation, candidate_id) in historical_map:
                        audit_pairs.append((source_id, candidate_id))
                        if len(audit_pairs) == 256:
                            break
            row_specs.append((source_id, relation, old_ids, new_ids, teacher_row["scores"]))
            modified_examples.append(new)
        modified_batches.append({**batch, "examples": modified_examples})
    if old_occurrence != len(historical_rows):
        raise ValueError("historical Teacher cache has trailing rows")

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(device)
    tcore_scores = _score_t_core_pairs(missing_pairs, audit_pairs, device, microbatch)
    audit_differences = [
        abs(tcore_scores[pair] - historical_map[(pair[0], "text->table", pair[1])])
        for pair in audit_pairs
    ]
    if audit_differences and max(audit_differences) > 1e-4:
        raise ValueError("T_core overlap scores disagree with the historical cache")

    modified_teacher = []
    new_pair_reuses = newly_scored = 0
    for source_id, relation, old_ids, new_ids, old_scores in row_specs:
        scores = []
        for index, candidate_id in enumerate(new_ids):
            if candidate_id == old_ids[index]:
                scores.append(float(old_scores[index]))
                continue
            historical_key = (source_id, relation, candidate_id)
            if historical_key in historical_map:
                scores.append(historical_map[historical_key])
                new_pair_reuses += 1
            else:
                scores.append(tcore_scores[(source_id, candidate_id)])
                newly_scored += 1
        modified_teacher.append({
            "query_id": source_id,
            "relation": relation,
            "candidate_ids": new_ids,
            "scores": scores,
        })

    output_schedule = directory / "materialized_schedule.jsonl.gz"
    output_teacher = directory / "teacher_edge_cache.jsonl.gz"
    write_rows(output_schedule, modified_batches)
    write_rows(output_teacher, modified_teacher)
    write_rows(directory / "new_tcore_pair_scores.jsonl.gz", (
        {"source_id": source_id, "target_id": target_id, "score": score}
        for (source_id, target_id), score in sorted(tcore_scores.items())
        if (source_id, target_id) in missing_pairs
    ))
    checks = {
        "only_text_to_table_batches_357_659_changed": unaffected_errors == 0,
        "positive_slots_unchanged": positive_slot_errors == 0,
        "list_widths_unchanged": width_errors == 0,
        "global_positive_protection": protection_errors == 0,
        "tcore_overlap_max_abs_difference_below_1e-4": not audit_differences or max(audit_differences) <= 1e-4,
        "tcore_identity_exact": sha256(T_CORE) == T_CORE_SHA,
    }
    receipt = {
        "status": "completed" if all(checks.values()) else "failed",
        "recipe": "F-P-ETNAT",
        "seed": seed,
        "gates": gates,
        "inputs": {
            "schedule": {"path": str(schedule_path.resolve()), "sha256": sha256(schedule_path)},
            "historical_teacher_cache": {"path": str(historical_teacher.resolve()), "sha256": sha256(historical_teacher)},
            "natural_reservoir": {"path": str(reservoir_path.resolve()), "sha256": sha256(reservoir_path)},
            "T_core": {"path": str(T_CORE.resolve()), "sha256": sha256(T_CORE)},
        },
        "outputs": {
            "schedule": {"path": str(output_schedule.resolve()), "sha256": sha256(output_schedule)},
            "teacher_cache": {"path": str(output_teacher.resolve()), "sha256": sha256(output_teacher)},
        },
        "changed_occurrences": changed_occurrences,
        "changed_unknown_slots": changed_slots,
        "missing_unique_pairs_scored_by_T_core": len(missing_pairs),
        "changed_slots_reusing_historical_pair_score": new_pair_reuses,
        "changed_slots_using_new_T_core_score": newly_scored,
        "historical_pair_score_max_occurrence_spread": maximum_historical_spread,
        "overlap_audit_pairs": len(audit_differences),
        "overlap_audit_max_abs_difference": max(audit_differences, default=0.0),
        "checks": checks,
    }
    write_json(receipt_path, receipt)
    write_json(directory / "candidate_manifest.json", receipt)
    if receipt["status"] != "completed":
        raise RuntimeError("ETNAT materialization checks failed")
    return receipt


def _save(model: torch.nn.Module, optimizer: torch.optim.Optimizer, directory: Path, step: int) -> dict[str, Any]:
    path = directory / f"checkpoints/step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint(model, "student-edge")
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["bridge"] = {"stage": "R30", "training_stage": "C1", "recipe": "F-P-ETNAT", "step": step}
    payload["r30"] = {"freeze_projections": True, "freeze_method": "requires_grad_only", "model_flag_preserved": False,
                        "projection_names": fp._projection_names(model), "optimizer_groups_preserved": True}
    torch.save(payload, path)
    return {
        "step": step,
        "checkpoint": {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size},
        "state_fingerprints": fp._state_fingerprints(model),
        "projection_references": student_projection_references(model),
    }


def train(seed: int, device_name: str) -> dict[str, Any]:
    gates = require_gates()
    directory = stage_dir(seed)
    materialization_path = directory / "MATERIALIZATION.json"
    materialization = json.loads(materialization_path.read_text()) if materialization_path.is_file() else None
    if not materialization or materialization.get("status") != "completed":
        raise RuntimeError("run and audit ETNAT materialization before training")
    receipt_path = directory / "EXECUTION.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text())
        if existing.get("status") == "completed":
            return existing
        raise FileExistsError(f"inspect incomplete ETNAT training: {directory}")
    device = torch.device(device_name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    parent = BRIDGE / f"training/B4/seed{seed}/C1/checkpoints/step_000356.pt"
    schedule_path = Path(materialization["outputs"]["schedule"]["path"])
    teacher_path = Path(materialization["outputs"]["teacher_cache"]["path"])
    schedule = fp._load_schedule(schedule_path)
    batches = fp._edge_batches(schedule, fp._teacher_scores(teacher_path))[356:]
    if len(batches) != 303:
        raise ValueError("ETNAT continuation must consume exactly 303 batches")

    model = load_student(parent, device)
    optimizer = _optimizer(model)
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    before_freeze = fp._optimizer_receipt(model, optimizer)
    projection_names = fp._projection_names(model)
    if len(optimizer.state) != len(list(model.parameters())) or any(float(value.get("step", -1)) != 356.0 for value in optimizer.state.values()):
        raise ValueError("complete step356 optimizer state was not restored")
    for name, parameter in model.named_parameters():
        if name in projection_names:
            parameter.requires_grad_(False)
            parameter.grad = None
    frozen_projection_state = {name: fp._fingerprint(parameter) for name, parameter in model.named_parameters() if name in projection_names}
    if fp._optimizer_receipt(model, optimizer) != before_freeze:
        raise ValueError("freezing changed optimizer groups or moments")
    write_json(directory / "parent_receipt.json", {
        "parent": {"path": str(parent.resolve()), "sha256": sha256(parent), "step": 356},
        "schedule": materialization["outputs"]["schedule"],
        "teacher": {**materialization["outputs"]["teacher_cache"], "role": "historical_C1_T_core"},
        "T_core": materialization["inputs"]["T_core"],
        "gates": gates,
        "projection_names": projection_names,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(device), "pid": os.getpid(), "command": [sys.executable, *sys.argv]},
    })
    write_json(directory / "optimizer_named_state_checks.json", {
        "before_freeze": before_freeze,
        "after_freeze": fp._optimizer_receipt(model, optimizer),
        "required_checks": {"same_optimizer_groups": True, "same_step356": True, "projection_group_retained": True},
    })
    execution: dict[str, Any] = {
        "status": "running", "recipe": "F-P-ETNAT", "seed": seed, "device": str(device),
        "start_step": 356, "updates_planned": 303,
        "parent": {"path": str(parent.resolve()), "sha256": sha256(parent)},
        "schedule": {"path": str(schedule_path.resolve()), "sha256": sha256(schedule_path)},
        "teacher": {"path": str(teacher_path.resolve()), "sha256": sha256(teacher_path), "kind": "historical_C1_T_core"},
        "materialization": {"path": str(materialization_path.resolve()), "sha256": sha256(materialization_path)},
        "freeze": {"projection_names": projection_names, "projection_state_at_freeze": frozen_projection_state,
                    "optimizer_groups_retained": True, "freeze_method": "requires_grad_only", "model_flag_preserved": False, "anchor_reset": False},
        "loss_config": {"ranking_weight": 1.0, "temperature": 1.0, "distillation_weight": 0.3, "edge_bce_weight": 0.0,
                        "anchor_weight": 0.1, "anchor_weight_evidence": 0.1, "positive_loss_mode": "sum_probability"},
        "checkpoints": {"356": {"alias_parent": True, "checkpoint": {"path": str(parent.resolve()), "sha256": sha256(parent)}}},
    }
    write_json(receipt_path, execution)
    store = FeatureStore.from_path(FEATURES, cache_size=60_000)
    consumed_path = directory / "consumed_batches.jsonl.gz"
    trace_path = directory / "step_trace.jsonl.gz"
    started = time.monotonic()
    with gzip.open(consumed_path, "wt", encoding="utf-8") as consumed, gzip.open(trace_path, "wt", encoding="utf-8") as traces:
        for offset, batch in enumerate(batches, 1):
            step = 356 + offset
            optimizer.zero_grad(set_to_none=True)
            terms, _raw = fp._loss(model, batch, store, device)
            terms["loss"].backward()
            if any(parameter.grad is not None for name, parameter in model.named_parameters() if name in projection_names):
                raise RuntimeError(f"projection gradient appeared at step {step}")
            before_projection = {name: fp._fingerprint(parameter) for name, parameter in model.named_parameters() if name in projection_names}
            losses = {key: float(value.detach().cpu()) for key, value in terms.items()}
            gradients = student_gradient_norms(model)
            optimizer.step()
            after_projection = {name: fp._fingerprint(parameter) for name, parameter in model.named_parameters() if name in projection_names}
            if before_projection != after_projection:
                raise RuntimeError(f"projection changed at step {step}")
            consumed_row = {"step": step, "examples": [asdict(example) for example in batch]}
            consumed.write(json.dumps(consumed_row, ensure_ascii=False) + "\n")
            traces.write(json.dumps({
                "step": step,
                "batch_sha256": stable_sha(consumed_row),
                "losses": losses,
                "gradient_norms": gradients,
                "candidate_ids_sha256": stable_sha([[e.query_id, f"{e.source_type}->{e.destination_type}", list(e.candidate_ids)] for e in batch]),
                "teacher_targets_sha256": stable_sha([list(e.teacher_logits or ()) for e in batch]),
                "relations": dict(Counter(f"{e.source_type}->{e.destination_type}" for e in batch)),
                "projection_unchanged": True,
            }, ensure_ascii=False) + "\n")
            execution["updates"] = step
            if step in SAVE_STEPS:
                execution["checkpoints"][str(step)] = _save(model, optimizer, directory, step)
                write_json(receipt_path, execution)
            if step % 25 == 0 or step in SAVE_STEPS:
                consumed.flush()
                traces.flush()
                print(json.dumps({"recipe": "F-P-ETNAT", "seed": seed, "step": step, "loss": losses["loss"], "elapsed": time.monotonic() - started}), flush=True)
    final_projection = {name: fp._fingerprint(parameter) for name, parameter in model.named_parameters() if name in projection_names}
    if final_projection != frozen_projection_state:
        raise RuntimeError("final projection fingerprint differs from freeze boundary")
    execution.update({
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "consumption": {"path": str(consumed_path.resolve()), "sha256": sha256(consumed_path)},
        "traces": {"path": str(trace_path.resolve()), "sha256": sha256(trace_path)},
        "actual_consumed_batch_ids_sha256": sha256(consumed_path),
        "final_projection_state": final_projection,
        "optimizer_after": fp._optimizer_receipt(model, optimizer),
    })
    write_json(receipt_path, execution)
    return execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("materialize", "train"), required=True)
    parser.add_argument("--seed", type=int, choices=(13, 29), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", type=int, default=2)
    args = parser.parse_args()
    result = materialize(args.seed, args.device, args.microbatch) if args.mode == "materialize" else train(args.seed, args.device)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
