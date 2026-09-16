"""Run the single preregistered R30 historical-C2 check."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.scoring import score_target_batch
from mmdd_stage1.training import (
    _student_path_losses,
    _target_teacher_scores,
    checkpoint,
    student_gradient_norms,
    student_projection_references,
)
from prepare_stage1_r27 import R12, R13, ROOT, read_json
from run_stage1_bridge import sha256, stable_sha, write_json
from run_stage1_r12_task_c import _optimizer
from run_stage1_r13 import _merge_witness_metadata


OUT = ROOT / "work/stage1_r30_c1_et_20260916"
FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"


def tensor_sha(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_fingerprints(model: torch.nn.Module) -> dict[str, str]:
    return {name: tensor_sha(value) for name, value in model.state_dict().items()}


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    aggregator: PathAggregator,
    recipe: str,
    step: int,
    directory: Path,
) -> dict[str, Any]:
    path = directory / f"checkpoints/step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint(model, "student-path", aggregator)
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["bridge"] = {"stage": "R30", "training_stage": "C2-CHECK", "recipe": recipe, "step": step}
    payload["r30"] = {"C1_freeze_released": True, "all_parameters_trainable": True}
    torch.save(payload, path)
    return {
        "step": step,
        "checkpoint": {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size},
        "state_fingerprints": state_fingerprints(model),
        "projection_references": student_projection_references(model),
    }


def require_selection(recipe: str) -> dict[str, Any]:
    path = OUT / "C1_SELECTION.json"
    if not path.is_file():
        raise RuntimeError("C1 selection has not been evaluated")
    selection = json.loads(path.read_text())
    if selection.get("status") != "selected" or selection.get("recipe") != recipe:
        raise RuntimeError(f"requested recipe {recipe} is not the preregistered selected recipe")
    return {"path": str(path.resolve()), "sha256": sha256(path), **selection}


def run(seed: int, recipe: str, device_name: str) -> dict[str, Any]:
    selection = require_selection(recipe)
    directory = OUT / f"C2-CHECK/{recipe}/seed{seed}"
    receipt_path = directory / "EXECUTION.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text())
        if existing.get("status") == "completed":
            return existing
        raise FileExistsError(f"inspect incomplete R30 C2 job: {directory}")
    parent = OUT / f"C1/{recipe}/seed{seed}/checkpoints/step_000659.pt"
    if not parent.is_file():
        raise FileNotFoundError(parent)
    device = torch.device(device_name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    model = load_student(parent, device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("C1 frozen parameters were not fully restored for C2")
    before_anchor = state_fingerprints(model)
    model.reset_projection_anchors()
    after_anchor = state_fingerprints(model)
    optimizer = _optimizer(model)
    if optimizer.state:
        raise ValueError("C2 optimizer must start fresh")

    examples = _merge_witness_metadata(ROOT)
    order_path = R13 / "taskD_witness_supervision/schedule_order.json"
    order = read_json(order_path)["indices"]
    ordered = [examples[index] for index in order]
    batches = [ordered[start:start + 64] for start in range(0, len(ordered), 64)]
    graph = R12 / "taskC_training/c2_candidates_seed13/path_hard.jsonl"
    if len(ordered) != 11_390 or len(batches) != 178:
        raise ValueError("historical C2 graph/order contract changed")
    control = ROOT / f"work/stage1_bridge_20260915/training/B5/seed{seed}/C2/consumed_order.jsonl.gz"
    with gzip.open(control, "rt", encoding="utf-8") as handle:
        control_first = json.loads(next(handle))["examples"]
    current_first = json.loads(json.dumps([asdict(example) for example in batches[0]]))
    contract_keys = (
        "query_id", "positive_target_ids", "teacher_direct_logits", "teacher_evidence_logits",
    )
    control_contract = [
        {key: row[key] for key in contract_keys} | {
            "candidate_ids": [candidate["target_id"] for candidate in row["candidates"]],
            "evidence_ids": [candidate["evidence_ids"] for candidate in row["candidates"]],
        }
        for row in control_first
    ]
    current_contract = [
        {key: row[key] for key in contract_keys} | {
            "candidate_ids": [candidate["target_id"] for candidate in row["candidates"]],
            "evidence_ids": [candidate["evidence_ids"] for candidate in row["candidates"]],
        }
        for row in current_first
    ]
    if current_contract != control_contract:
        raise ValueError("R30 C2 first-batch graph/Teacher contract differs from Bridge B5")

    directory.mkdir(parents=True, exist_ok=True)
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    execution: dict[str, Any] = {
        "status": "running",
        "stage": "R30",
        "training_stage": "C2-CHECK",
        "recipe": recipe,
        "seed": seed,
        "selection": selection,
        "parent": {"path": str(parent.resolve()), "sha256": sha256(parent)},
        "optimizer_initial_state": "fresh",
        "all_parameters_trainable": True,
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "graph": {"path": str(graph.resolve()), "sha256": sha256(graph)},
        "order": {"path": str(order_path.resolve()), "sha256": sha256(order_path)},
        "first_batch_contract": {"matches_B5": True, "sha256": stable_sha(current_contract)},
        "teacher_target_source": "embedded historical full-bag D/E tensors",
        "evidence_bag_policy": "full historical evidence_ids",
        "anchor_boundary": {"before": stable_sha(before_anchor), "after": stable_sha(after_anchor)},
        "updates": 0,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(device), "pid": os.getpid(), "command": [sys.executable, *sys.argv]},
    }
    write_json(receipt_path, execution)
    store = FeatureStore.from_path(FEATURES, cache_size=60_000)
    execution["checkpoints"] = {"0": save_checkpoint(model, optimizer, aggregator, recipe, 0, directory)}
    consumed_path = directory / "consumed_order.jsonl.gz"
    trace_path = directory / "step_traces.jsonl.gz"
    started = time.monotonic()
    with gzip.open(consumed_path, "wt", encoding="utf-8") as consumed, gzip.open(trace_path, "wt", encoding="utf-8") as traces:
        for step, batch in enumerate(batches, 1):
            scores = score_target_batch(model, batch, store, device, aggregator)
            teacher = _target_teacher_scores(batch, device)
            terms = _student_path_losses(
                model, scores, teacher, None,
                temperature=1.0,
                distillation_weight=0.3,
                anchor_weight=0.1,
                anchor_weight_evidence=0.1,
                distillation_rows=None,
                positive_loss_mode="sum_probability",
            )
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            gradients = student_gradient_norms(model)
            optimizer.step()
            consumed_row = {"step": step, "examples": [asdict(example) for example in batch]}
            consumed.write(json.dumps(consumed_row, ensure_ascii=False) + "\n")
            traces.write(json.dumps({
                "step": step,
                "batch_sha256": stable_sha(consumed_row),
                "losses": {key: float(value.detach().cpu()) for key, value in terms.items()},
                "gradient_norms": gradients,
                "candidate_ids_sha256": stable_sha([[example.query_id, [candidate.target_id for candidate in example.candidates]] for example in batch]),
                "evidence_ids_sha256": stable_sha([[[candidate.target_id, list(candidate.evidence_ids)] for candidate in example.candidates] for example in batch]),
                "teacher_direct_sha256": stable_sha([list(example.teacher_direct_logits or ()) for example in batch]),
                "teacher_evidence_sha256": stable_sha([list(example.teacher_evidence_logits or ()) for example in batch]),
                "direct_positive_mask_sha256": tensor_sha(scores.direct.positive_mask),
                "evidence_positive_mask_sha256": tensor_sha(scores.evidence.positive_mask),
            }, ensure_ascii=False) + "\n")
            execution["updates"] = step
            if step in {89, 178}:
                execution["checkpoints"][str(step)] = save_checkpoint(model, optimizer, aggregator, recipe, step, directory)
                write_json(receipt_path, execution)
            if step % 25 == 0 or step in {89, 178}:
                consumed.flush()
                traces.flush()
                print(json.dumps({"recipe": recipe, "seed": seed, "stage": "C2-CHECK", "step": step,
                                  "loss": float(terms["loss"].detach()), "elapsed": time.monotonic() - started}), flush=True)
    execution.update({
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "consumption": {"path": str(consumed_path.resolve()), "sha256": sha256(consumed_path)},
        "traces": {"path": str(trace_path.resolve()), "sha256": sha256(trace_path)},
        "actual_consumed_order_sha256": stable_sha([example.query_id for example in ordered]),
    })
    write_json(receipt_path, execution)
    return execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("F-P", "F-P-ETNAT"), required=True)
    parser.add_argument("--seed", type=int, choices=(13, 29), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.recipe, args.device), ensure_ascii=False))


if __name__ == "__main__":
    main()
