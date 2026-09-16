"""Train the preregistered R29 freeze-P/freeze-R Student arms.

This runner keeps R28's graph, objective, order, optimizer rates and anchor
semantics.  Its only per-arm change is which projection or relation tensors
have ``requires_grad`` and enter the optimizer.  The device is explicit in
every run identity so CPU smoke runs cannot be mistaken for the formal GPU
protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.r26_training import graph_edges
from mmdd_stage1.r28_objectives import backward_batch
from mmdd_stage1.features import FeatureStore
from prepare_stage1_r28 import ROOT as REPO_ROOT, registry, feature_store
from run_stage1_r13 import _merge_witness_metadata
from prepare_stage1_r28 import inputs


ROOT = Path(REPO_ROOT)
R29 = ROOT / "work/stage1_optimization_r29_candidate_vs_drift_20260915"
PARENT = Path(inputs()["student_parent"]["path"])
EPOCHS = 3
BATCH_SIZE = 64
MICROBATCH = 8


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, str]:
    return {name: tensor_hash(value) for name, value in model.state_dict().items() if name.startswith(prefixes)}


def geometry(model: torch.nn.Module) -> dict[str, Any]:
    out: dict[str, Any] = {"projections": {}, "relations": {}}
    for name, value in model.named_parameters():
        if name.startswith("projections."):
            out["projections"][name] = {"frobenius": float(value.detach().norm()), "spectral": float(torch.linalg.matrix_norm(value.detach(), ord=2)), "sha256": tensor_hash(value.detach())}
        elif name.startswith("relations."):
            out["relations"][name] = {"frobenius": float(value.detach().norm()), "spectral": float(torch.linalg.matrix_norm(value.detach(), ord=2)), "sha256": tensor_hash(value.detach())}
    return out


def make_optimizer(model: torch.nn.Module, arm: str) -> torch.optim.AdamW:
    relation = [p for n, p in model.named_parameters() if n.startswith(("relations.", "relation_as.", "relation_bs.")) and p.requires_grad]
    projection = [p for n, p in model.named_parameters() if n.startswith("projections.") and p.requires_grad]
    groups = []
    if relation:
        groups.append({"params": relation, "lr": 1e-5, "name": "relations"})
    if projection:
        groups.append({"params": projection, "lr": 1e-6, "name": "projections"})
    if not groups:
        raise RuntimeError(f"{arm}: no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=0.01)


def configure(model: torch.nn.Module, arm: str) -> tuple[list[str], list[str]]:
    if arm not in {"S-EDGE-FREEZE-P", "S-EDGE-FREEZE-R"}:
        raise ValueError(arm)
    for name, parameter in model.named_parameters():
        if name.startswith("projections."):
            parameter.requires_grad = arm == "S-EDGE-FREEZE-R"
        elif name.startswith(("relations.", "relation_as.", "relation_bs.")):
            parameter.requires_grad = arm == "S-EDGE-FREEZE-P"
        else:
            parameter.requires_grad = False
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    return trainable, frozen


def relation_activity(batch: list[Any], known: dict, store: FeatureStore) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n_total": 0, "n_active": 0})
    for example in batch:
        for edge in graph_edges(example, known, lambda oid: store.embedding_features(oid).object_type):
            key = f"{edge.source_type}->{edge.destination_type}"
            positive = bool(edge.positive_ids)
            unknown = bool(set(edge.candidate_ids) - set(edge.positive_ids))
            counts[key]["n_total"] += 1
            counts[key]["n_active"] += int(positive and unknown)
    return dict(counts)


def checkpoint(path: Path, model: Any, optimizer: Any, arm: str, seed: int, epoch: float, step: int, parent_hashes: dict[str, str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "model_kind": "student", "completed_stage": "r29", "r29_arm": arm, "continuation_seed": seed, "optimizer_step": step, "epoch": epoch, "config": model.config(), "state_dict": {n: p.detach().cpu() for n, p in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "r29_parent": str(PARENT), "r29_parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(), "r29_parent_tensor_hashes": parent_hashes, "r29_frozen_parameters": [n for n, p in model.named_parameters() if not p.requires_grad]}
    torch.save(payload, path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "epoch": epoch, "step": step, "geometry": geometry(model)}


def train(arm: str, seed: int = 13, device_name: str = "cpu", epochs: int = EPOCHS) -> dict[str, Any]:
    if seed != 13:
        raise ValueError("R29 first pass is seed13 only")
    if epochs > EPOCHS:
        raise ValueError("R29 budget is at most three epochs")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    model = load_student(PARENT, device).train()
    parent_projection = state_hash(model, ("projections.",))
    parent_relation = state_hash(model, ("relations.", "relation_as.", "relation_bs."))
    trainable, frozen = configure(model, arm)
    optimizer = make_optimizer(model, arm)
    store = feature_store(False)
    examples = _merge_witness_metadata(ROOT)
    known = registry()
    orders = json.loads((R29.parent / "common/orders.json").read_text()) if (R29 / "common/orders.json").exists() else json.loads((ROOT / "work/stage1_optimization_r28_split_path_20260915/common/orders.json").read_text())
    job = R29 / "training" / arm / f"seed{seed}"
    job.mkdir(parents=True, exist_ok=True)
    identity = {"arm": arm, "seed": seed, "device": device_name, "parent_path": str(PARENT.resolve()), "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(), "feature_manifest_sha256": hashlib.sha256((inputs()["feature_manifest"]["path"] and Path(inputs()["feature_manifest"]["path"])).read_bytes()).hexdigest(), "trainable_names": trainable, "frozen_names": frozen, "optimizer_groups": [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups], "r28_order_hashes": [hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest() for order in orders["13"]], "logical_batch": BATCH_SIZE, "microbatch": MICROBATCH, "epochs": epochs, "candidate_policy": "R12 graph unchanged", "teacher": "not_used"}
    (job / "candidate_manifest.json").write_text(json.dumps({"policy": "R12 graph unchanged", "source": str(inputs()["target_path_graph"]["path"]), "r12_graph_sha256": hashlib.sha256(Path(inputs()["target_path_graph"]["path"]).read_bytes()).hexdigest(), "positive_registry_sha256": hashlib.sha256(Path(inputs()["edge_positive_registry"]["path"]).read_bytes()).hexdigest(), "only_variable": arm, "status": "pass"}, indent=2) + "\n")
    (job / "order_hashes.json").write_text(json.dumps({"seed13": identity["r28_order_hashes"]}, indent=2) + "\n")
    (job / "RUN_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    fixed_hashes = parent_projection if arm == "S-EDGE-FREEZE-P" else parent_relation
    history: list[dict[str, Any]] = []
    step = 0
    started = time.monotonic()
    checkpoint(job / "checkpoints/step_000000.pt", model, optimizer, arm, seed, 0.0, 0, fixed_hashes)
    with (job / "history.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            order = list(orders["13"][epoch - 1])
            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start:start + BATCH_SIZE]
                batch = [examples[i] for i in indices]
                optimizer.zero_grad(set_to_none=True)
                terms = backward_batch(model, batch, store, device, "EDGE", known, student=True, microbatch=MICROBATCH)
                if not math.isfinite(float(terms["loss"])):
                    raise FloatingPointError(f"nonfinite loss at {step}")
                grad_norms = {name: float(parameter.grad.detach().norm()) for name, parameter in model.named_parameters() if parameter.grad is not None}
                optimizer.step(); step += 1
                trace = {"step": step, "epoch": epoch, "batch_start": start, "losses": {key: value for key, value in terms.items() if not isinstance(value, dict)}, "gradient_norms": grad_norms, "active_relations": relation_activity(batch, known, store), "elapsed_seconds": time.monotonic() - started}
                log.write(json.dumps(trace, ensure_ascii=False) + "\n")
                if step % 10 == 0:
                    log.flush()
            checkpoint(job / "checkpoints" / f"step_{step:06d}.pt", model, optimizer, arm, seed, float(epoch), step, fixed_hashes)
    final_projection = state_hash(model, ("projections.",)); final_relation = state_hash(model, ("relations.", "relation_as.", "relation_bs."))
    if arm == "S-EDGE-FREEZE-P" and final_projection != parent_projection:
        raise AssertionError("freeze-P projection tensor changed")
    if arm == "S-EDGE-FREEZE-R" and final_relation != parent_relation:
        raise AssertionError("freeze-R relation tensor changed")
    result = {"status": "completed", "arm": arm, "seed": seed, "epochs": epochs, "updates": step, "device": device_name, "parent_projection_hashes": parent_projection, "final_projection_hashes": final_projection, "parent_relation_hashes": parent_relation, "final_relation_hashes": final_relation, "elapsed_seconds": time.monotonic() - started}
    (job / "EXECUTION.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("S-EDGE-FREEZE-P", "S-EDGE-FREEZE-R"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(train(args.arm, args.seed, args.device, args.epochs), ensure_ascii=False))


if __name__ == "__main__":
    main()
