"""Execute the preregistered R30 C1 freeze-P continuation.

This runner is deliberately separate from the historical B2--B7 bridge
runner.  It consumes the real Bridge B4 step-356 checkpoint and optimizer,
keeps the complete AdamW parameter groups for an exact state-dict restore,
then freezes every projection parameter before consuming batches 357--659.
The script writes only under ``work/stage1_r30_c1_et_20260916``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import EdgeExample
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.scoring import ListScores, score_edge_batch
from mmdd_stage1.training import (
    _student_edge_losses,
    checkpoint,
    student_gradient_norms,
    student_projection_references,
)
from run_stage1_bridge import ROOT, R25, sha256, stable_sha, write_json
from run_stage1_r12_task_c import _optimizer, _ranking_scores


OUT = ROOT / "work/stage1_r30_c1_et_20260916"
SEEDS = (13, 29)
PARENT_ROOT = ROOT / "work/stage1_bridge_20260915/training/B4"
SCHEDULE_ROOT = ROOT / "work/stage1_bridge_20260915/schedules"
TEACHER_ROOT = ROOT / "work/stage1_optimization_r25_final_20260914/common"
FEATURE_ROOT = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
SAVE_STEPS = (356, 357, 428, 500, 580, 659)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _fingerprint(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
        return digest.hexdigest()
    if isinstance(value, dict):
        return stable_sha({str(k): _fingerprint(v) if isinstance(v, torch.Tensor) else v for k, v in value.items()})
    return stable_sha(value)


def _load_schedule(path: Path) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            payload = json.loads(line)
            if int(payload.get("step", line_no)) != line_no:
                raise ValueError(f"{path}: non-contiguous schedule step at line {line_no}")
            batches.append(payload["examples"])
    if len(batches) != 659:
        raise ValueError(f"{path}: expected 659 batches, got {len(batches)}")
    return batches


def _teacher_scores(path: Path) -> list[dict[str, Any]]:
    # The historical cache is occurrence-aligned with closure_full.  A
    # query/relation/candidate key can legitimately occur several times and
    # may differ by a few float32 rounding ulps, so do not collapse it into a
    # last-write dictionary.
    result: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids = [str(value) for value in row["candidate_ids"]]
            scores = [float(value) for value in row["scores"]]
            if len(ids) != len(scores):
                raise ValueError(f"Teacher candidate/score mismatch for occurrence {len(result)}")
            result.append({"query_id": str(row["query_id"]), "relation": str(row["relation"]), "candidate_ids": ids, "scores": scores})
    return result


def _edge_batches(schedule: list[list[dict[str, Any]]], teacher: list[dict[str, Any]]) -> list[list[EdgeExample]]:
    converted: list[list[EdgeExample]] = []
    occurrence = 0
    for raw_batch in schedule:
        batch: list[EdgeExample] = []
        for row in raw_batch:
            query_id = str(row["query_id"])
            relation = f"{row['source_type']}->{row['destination_type']}"
            candidate_ids = tuple(str(value) for value in row["candidate_ids"])
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"duplicate candidate in {query_id}/{relation}")
            if occurrence >= len(teacher):
                raise ValueError("Teacher cache ended before closure_full")
            teacher_row = teacher[occurrence]
            occurrence += 1
            if (
                teacher_row["query_id"] != query_id
                or teacher_row["relation"] != relation
                or teacher_row["candidate_ids"] != list(candidate_ids)
            ):
                raise ValueError(f"Teacher occurrence {occurrence} does not align with closure_full ({query_id}, {relation})")
            positive_id = str(row["positive_id"])
            if positive_id not in candidate_ids:
                raise ValueError(f"positive not in candidate list: {query_id}/{relation}")
            batch.append(
                EdgeExample(
                    query_id=query_id,
                    candidate_ids=candidate_ids,
                    positive_index=candidate_ids.index(positive_id),
                    dataset=str(row.get("dataset", "default")),
                    split="train",
                    teacher_logits=tuple(teacher_row["scores"]),
                    teacher_checkpoint_sha256="historical_C1_T_core",
                    teacher_logit_mode="historical_C1_T_core",
                    source_type=str(row["source_type"]),
                    destination_type=str(row["destination_type"]),
                    positive_ids=tuple(str(value) for value in row.get("positive_ids", ())),
                    confirmed_labels=None
                    if row.get("confirmed_labels") is None
                    else tuple(None if value is None else int(value) for value in row["confirmed_labels"]),
                )
            )
        converted.append(batch)
    if occurrence != len(teacher):
        raise ValueError(f"Teacher cache has {len(teacher) - occurrence} trailing occurrences")
    return converted


def _parameter_names(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[int, str]:
    by_object = {id(parameter): name for name, parameter in model.named_parameters()}
    result: dict[int, str] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = by_object.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains an unnamed model parameter")
            result[id(parameter)] = name
    return result


def _optimizer_receipt(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    names = _parameter_names(model, optimizer)
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        groups.append(
            {
                "index": index,
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": [float(value) for value in group["betas"]],
                "eps": float(group["eps"]),
                "parameter_names": [names[id(parameter)] for parameter in group["params"]],
            }
        )
    state = {}
    for parameter, values in optimizer.state.items():
        name = names[id(parameter)]
        state[name] = {
            key: (_fingerprint(value) if isinstance(value, torch.Tensor) else value)
            for key, value in values.items()
        }
    return {"groups": groups, "state": state, "state_count": len(state)}


def _projection_names(model: torch.nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if name.startswith("projections.") or name.startswith("projection_residual_")]


def _state_fingerprints(model: torch.nn.Module) -> dict[str, str]:
    return {name: _fingerprint(value) for name, value in model.state_dict().items()}


def _loss(model: torch.nn.Module, batch: list[EdgeExample], store: FeatureStore, device: torch.device) -> tuple[dict[str, torch.Tensor], ListScores]:
    raw = score_edge_batch(model, batch, store, device, student_score_space="raw_logit")
    teacher_values = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(example.teacher_logits, device=device) for example in batch], batch_first=True
    )
    teacher = ListScores(teacher_values, raw.candidate_mask, raw.positive_indices, raw.positive_mask)
    terms = _student_edge_losses(
        model,
        batch,
        raw,
        teacher,
        _ranking_scores(raw),
        None,
        ranking_weight=1.0,
        temperature=1.0,
        distillation_weight=0.3,
        edge_bce_weight=0.0,
        anchor_weight=0.1,
        anchor_weight_evidence=0.1,
        positive_loss_mode="sum_probability",
    )
    return terms, raw


def _first_batch_probe(parent: Path, batch: list[EdgeExample], device: torch.device) -> dict[str, Any]:
    """Compare the unfrozen control and frozen-P first update on one batch."""
    store = FeatureStore.from_path(FEATURE_ROOT, cache_size=4096)
    results: dict[str, Any] = {}
    for label, freeze in (("control", False), ("frozen_P", True)):
        model = load_student(parent, device)
        optimizer = _optimizer(model)
        payload = torch.load(parent, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if freeze:
            for name, parameter in model.named_parameters():
                if name.startswith("projections.") or name.startswith("projection_residual_"):
                    parameter.requires_grad_(False)
        optimizer.zero_grad(set_to_none=True)
        terms, _ = _loss(model, batch, store, device)
        terms["loss"].backward()
        relation_grads = {
            name: _fingerprint(parameter.grad)
            for name, parameter in model.named_parameters()
            if not name.startswith("projections.") and not name.startswith("projection_residual_")
        }
        before = {name: _fingerprint(parameter) for name, parameter in model.named_parameters() if name.startswith("relations.")}
        optimizer.step()
        after = {name: _fingerprint(parameter) for name, parameter in model.named_parameters() if name.startswith("relations.")}
        results[label] = {
            "losses": {key: float(value.detach().cpu()) for key, value in terms.items()},
            "relation_gradient_fingerprints": relation_grads,
            "relation_before": before,
            "relation_after": after,
            "frozen_projection_grads_none": all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if name.startswith("projections.") or name.startswith("projection_residual_")
            ),
        }
        del model, optimizer, payload
        torch.cuda.empty_cache()
    results["loss_equal"] = results["control"]["losses"] == results["frozen_P"]["losses"]
    results["relation_gradients_equal"] = results["control"]["relation_gradient_fingerprints"] == results["frozen_P"]["relation_gradient_fingerprints"]
    results["relation_updates_equal"] = results["control"]["relation_after"] == results["frozen_P"]["relation_after"]
    results["control_is_reference_only"] = True
    return results


def _save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, stage_dir: Path, step: int) -> dict[str, Any]:
    path = stage_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint(model, "student-edge")
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["bridge"] = {"stage": "R30", "training_stage": "C1", "recipe": "F-P", "step": step}
    payload["r30"] = {"freeze_projections": True, "freeze_method": "requires_grad_only", "model_flag_preserved": False, "projection_names": _projection_names(model), "optimizer_groups_preserved": True}
    torch.save(payload, path)
    return {
        "step": step,
        "checkpoint": {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size},
        "state_fingerprints": _state_fingerprints(model),
        "projection_references": student_projection_references(model),
    }


def run_seed(seed: int, device_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; rerun in an environment with the requested GPU")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)

    parent = PARENT_ROOT / f"seed{seed}/C1/checkpoints/step_000356.pt"
    schedule_path = SCHEDULE_ROOT / f"seed{seed}_steps659/closure_full.jsonl.gz"
    teacher_path = TEACHER_ROOT / f"teacher_edge_cache_seed{seed}.jsonl.gz"
    stage_dir = OUT / "C1" / "F-P" / f"seed{seed}"
    receipt_path = stage_dir / "EXECUTION.json"
    if receipt_path.is_file():
        existing = _json(receipt_path)
        if existing.get("status") == "completed":
            return existing
        raise FileExistsError(f"incomplete R30 job requires manual audit: {stage_dir}")
    for path in (parent, schedule_path, teacher_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    schedule = _load_schedule(schedule_path)
    teacher = _teacher_scores(teacher_path)
    batches = _edge_batches(schedule, teacher)
    active_batches = batches[356:]
    if len(active_batches) != 303:
        raise ValueError(f"expected 303 active batches, got {len(active_batches)}")

    model = load_student(parent, device)
    optimizer = _optimizer(model)
    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(parent_payload["optimizer_state_dict"])
    before_freeze = _optimizer_receipt(model, optimizer)
    parent_state = _state_fingerprints(model)
    projection_names = _projection_names(model)
    if len(projection_names) == 0:
        raise ValueError("no projection parameters found")
    if any(float(group["lr"]) != expected for group, expected in zip(optimizer.param_groups, (1e-5, 1e-6))):
        raise ValueError("R30 optimizer LR differs from the locked C1 optimizer")
    if any(float(group["weight_decay"]) != 0.01 for group in optimizer.param_groups):
        raise ValueError("R30 optimizer weight decay differs from the locked C1 optimizer")
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError("optimizer state is not complete at C1 step356")
    if any(float(values.get("step", -1)) != 356.0 for values in optimizer.state.values()):
        raise ValueError("optimizer moments do not all resume at step356")

    stage_dir.mkdir(parents=True, exist_ok=True)
    _write(stage_dir / "parent_receipt.json", {
        "parent": {"path": str(parent.resolve()), "sha256": sha256(parent), "step": 356},
        "schedule": {"path": str(schedule_path.resolve()), "sha256": sha256(schedule_path), "batches": len(schedule)},
        "teacher": {"path": str(teacher_path.resolve()), "sha256": sha256(teacher_path), "pairs": len(teacher), "role": "historical_C1_T_core"},
        "model_config": parent_payload["config"],
        "projection_names": projection_names,
        "parent_state_fingerprints": parent_state,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device), "pid": os.getpid(), "command": [sys.executable, *sys.argv]},
    })
    probe = _first_batch_probe(parent, active_batches[0], device)
    _write(stage_dir / "optimizer_named_state_checks.json", {
        "before_freeze": before_freeze,
        "first_batch_control_parity": probe,
        "required_checks": {"same_optimizer_groups": True, "same_step356": True, "projection_group_retained": True},
    })

    # Preserve the checkpoint's full-chain anchor semantics.  The model flag
    # remains false; freezing is expressed by requires_grad after complete
    # optimizer restoration, while the original P/R reference is retained.
    for parameter_name, parameter in model.named_parameters():
        if parameter_name in projection_names:
            parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name in projection_names and parameter.requires_grad:
            raise ValueError(f"projection parameter remained trainable: {name}")
        if name in projection_names:
            parameter.grad = None
    frozen_projection_state = {name: _fingerprint(parameter) for name, parameter in model.named_parameters() if name in projection_names}
    frozen_optimizer_state = _optimizer_receipt(model, optimizer)
    if before_freeze["groups"] != frozen_optimizer_state["groups"] or before_freeze["state"] != frozen_optimizer_state["state"]:
        raise ValueError("freezing changed optimizer groups or moments")
    _write(stage_dir / "candidate_manifest.json", {
        "recipe": "F-P", "candidate_policy": "unchanged Bridge closure_full", "schedule_sha256": sha256(schedule_path),
        "batch_count": 303, "source_batches": [357, 659],
        "candidate_ids_sha256": stable_sha([[[e.query_id, f"{e.source_type}->{e.destination_type}", list(e.candidate_ids)] for e in batch] for batch in active_batches]),
        "positive_ids_sha256": stable_sha([[[e.query_id, list(e.positive_ids)] for e in batch] for batch in active_batches]),
        "teacher_logits_sha256": stable_sha([[[e.query_id, list(e.teacher_logits or ())] for e in batch] for batch in active_batches]),
        "frozen_projection_state": frozen_projection_state,
        "only_allowed_change": "projection trainability; all IDs/order/labels/teacher logits unchanged",
    })

    store = FeatureStore.from_path(FEATURE_ROOT, cache_size=60_000)
    execution: dict[str, Any] = {
        "status": "running", "recipe": "F-P", "seed": seed, "device": str(device), "start_step": 356, "updates_planned": 303,
        "parent": {"path": str(parent.resolve()), "sha256": sha256(parent)},
        "schedule": {"path": str(schedule_path.resolve()), "sha256": sha256(schedule_path)},
        "teacher": {"path": str(teacher_path.resolve()), "sha256": sha256(teacher_path), "kind": "historical_C1_T_core"},
        "freeze": {"projection_names": projection_names, "projection_state_at_freeze": frozen_projection_state, "optimizer_groups_retained": True, "freeze_method": "requires_grad_only", "model_flag_preserved": False, "anchor_reset": False},
        "loss_config": {"ranking_weight": 1.0, "temperature": 1.0, "distillation_weight": 0.3, "edge_bce_weight": 0.0, "anchor_weight": 0.1, "anchor_weight_evidence": 0.1, "positive_loss_mode": "sum_probability"},
        "checkpoints": {"356": {"alias_parent": True, "checkpoint": {"path": str(parent.resolve()), "sha256": sha256(parent)}}},
    }
    _write(receipt_path, execution)
    consumed_path = stage_dir / "consumed_batches.jsonl.gz"
    trace_path = stage_dir / "step_trace.jsonl.gz"
    started = time.monotonic()
    with gzip.open(consumed_path, "wt", encoding="utf-8") as consumed, gzip.open(trace_path, "wt", encoding="utf-8") as traces:
        for offset, batch in enumerate(active_batches, 1):
            step = 356 + offset
            optimizer.zero_grad(set_to_none=True)
            terms, _ = _loss(model, batch, store, device)
            terms["loss"].backward()
            if any(parameter.grad is not None for name, parameter in model.named_parameters() if name in projection_names):
                raise RuntimeError(f"projection gradient appeared at step {step}")
            before_projection = {name: _fingerprint(parameter) for name, parameter in model.named_parameters() if name in projection_names}
            terms_loss = {key: float(value.detach().cpu()) for key, value in terms.items()}
            gradients = student_gradient_norms(model)
            optimizer.step()
            after_projection = {name: _fingerprint(parameter) for name, parameter in model.named_parameters() if name in projection_names}
            if before_projection != after_projection:
                raise RuntimeError(f"projection changed at step {step}")
            consumed_row = {"step": step, "examples": [asdict(example) for example in batch]}
            consumed.write(json.dumps(consumed_row, ensure_ascii=False) + "\n")
            traces.write(json.dumps({"step": step, "batch_sha256": stable_sha(consumed_row), "losses": terms_loss, "gradient_norms": gradients, "candidate_ids_sha256": stable_sha([[e.query_id, f"{e.source_type}->{e.destination_type}", list(e.candidate_ids)] for e in batch]), "teacher_targets_sha256": stable_sha([list(e.teacher_logits or ()) for e in batch]), "relations": dict(Counter(f"{e.source_type}->{e.destination_type}" for e in batch)), "projection_unchanged": before_projection == after_projection}, ensure_ascii=False) + "\n")
            execution["updates"] = step
            if step in SAVE_STEPS[1:]:
                execution["checkpoints"][str(step)] = _save_checkpoint(model, optimizer, stage_dir, step)
                _write(receipt_path, execution)
            if step % 25 == 0 or step in SAVE_STEPS[1:]:
                consumed.flush(); traces.flush()
                print(json.dumps({"recipe": "F-P", "seed": seed, "step": step, "loss": terms_loss["loss"], "elapsed": time.monotonic() - started}), flush=True)

    final_projection = {name: _fingerprint(parameter) for name, parameter in model.named_parameters() if name in projection_names}
    if final_projection != frozen_projection_state:
        raise RuntimeError("final projection fingerprint differs from freeze boundary")
    execution.update({"status": "completed", "elapsed_seconds": time.monotonic() - started, "consumption": {"path": str(consumed_path.resolve()), "sha256": sha256(consumed_path)}, "traces": {"path": str(trace_path.resolve()), "sha256": sha256(trace_path)}, "actual_consumed_batch_ids_sha256": sha256(consumed_path), "final_projection_state": final_projection, "optimizer_after": _optimizer_receipt(model, optimizer)})
    _write(receipt_path, execution)
    _write(stage_dir / "G_F.json", {"status": "pending_evaluation", "recipe": "F-P", "seed": seed, "note": "Training correctness complete; G-F requires repaired retrieval evaluation for both seeds."})
    return execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for seed in args.seed or list(SEEDS):
        print(json.dumps(run_seed(seed, args.device), ensure_ascii=False))


if __name__ == "__main__":
    main()
