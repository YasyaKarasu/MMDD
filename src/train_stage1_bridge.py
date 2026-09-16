"""Train the first causal bridge step from B13 exact: B1 modern init only.

B1 keeps the frozen R27 historical C1 schedule/logits and historical C2
graph/logits/order.  Its only recipe change is loading the actual R25 modern
step-0 Student state.  The two stages use fresh optimizers and write complete
consumption/lineage receipts under the bridge namespace.
"""
from __future__ import annotations

import argparse
import gzip
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
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.scoring import score_edge_batch, score_target_batch
from mmdd_stage1.training import (
    _student_edge_losses,
    _student_path_losses,
    _target_teacher_scores,
    checkpoint,
    student_gradient_norms,
    student_projection_references,
)
from prepare_stage1_r27 import R12, R13, read_json
from run_stage1_bridge import OUT, ROOT, record, sha256, stable_sha, write_json
from run_stage1_r12_task_c import (
    _optimizer,
    _ranking_scores,
    _schedule_batches,
    _score_payload,
    _teacher_list_scores,
)
from run_stage1_r13 import _merge_witness_metadata


def _tensor_sha(tensor: torch.Tensor) -> str:
    digest = __import__("hashlib").sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_fingerprints(model: torch.nn.Module) -> dict[str, str]:
    return {name: _tensor_sha(value) for name, value in model.state_dict().items()}


def _modern_init(seed: int) -> Path:
    return ROOT / f"work/stage1_optimization_r25_final_20260914/training/C1/seed{seed}/checkpoints/step_000000.pt"


def _job(seed: int, stage: str) -> Path:
    return OUT / "training/B1" / f"seed{seed}" / stage


def _save(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    aggregator: PathAggregator | None,
    stage: str,
    step: int,
    directory: Path,
) -> dict[str, Any]:
    path = directory / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint(model, "student-edge" if stage == "C1" else "student-path", aggregator)
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["bridge"] = {"stage": "B1", "training_stage": stage, "step": step}
    torch.save(payload, path)
    receipt = {
        "stage": "B1",
        "training_stage": stage,
        "step": step,
        "checkpoint": record(path),
        "state_fingerprints": _state_fingerprints(model),
        "projection_references": student_projection_references(model),
    }
    write_json(path.with_suffix(".json"), receipt)
    return receipt


def _runtime(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "cpu_threads": torch.get_num_threads(),
    }


def train_c1(seed: int, device_name: str) -> dict[str, Any]:
    directory = _job(seed, "C1")
    receipt_path = directory / "EXECUTION.json"
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        if existing.get("status") == "completed":
            return existing
        raise FileExistsError(f"Inspect incomplete B1 C1 job: {directory}")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    parent = _modern_init(seed)
    if not parent.is_file():
        raise FileNotFoundError(parent)
    model = load_student(parent, device)
    optimizer = _optimizer(model)
    if optimizer.state:
        raise ValueError("B1 C1 optimizer must start fresh")
    scores, teacher_manifest = _score_payload(R12)
    schedule = R12 / "taskC_training/candidates_seed13_steps356/candidates.jsonl.gz"
    batches = list(_schedule_batches(schedule, scores, str(teacher_manifest["teacher_checkpoint_sha256"])))
    if len(batches) != 356 or [step for step, _ in batches] != list(range(1, 357)):
        raise ValueError("Historical C1 schedule is not the exact registered 356-step order")
    directory.mkdir(parents=True, exist_ok=True)
    execution: dict[str, Any] = {
        "status": "running",
        "stage": "B1",
        "training_stage": "C1",
        "unique_factor": "historical PCA/init -> actual R25 modern seed-specific step0",
        "parent": record(parent),
        "optimizer_initial_state": "fresh",
        "schedule": record(schedule),
        "teacher_scores": record(Path(teacher_manifest["scores"])),
        "teacher_checkpoint_sha256": teacher_manifest["teacher_checkpoint_sha256"],
        "updates": 0,
        "runtime": _runtime(device),
    }
    write_json(receipt_path, execution)
    store = FeatureStore.from_path(ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=60_000)
    nodes = {0, 178, 356}
    execution["checkpoints"] = {"0": _save(model, optimizer, None, "C1", 0, directory)}
    started = time.monotonic()
    consumed_path = directory / "consumed_batches.jsonl.gz"
    trace_path = directory / "step_traces.jsonl.gz"
    with gzip.open(consumed_path, "wt", encoding="utf-8") as consumed, gzip.open(trace_path, "wt", encoding="utf-8") as traces:
        for step, batch in batches:
            model.train()
            raw = score_edge_batch(model, batch, store, device, student_score_space="raw_logit")
            teacher = _teacher_list_scores(batch, raw, device)
            terms = _student_edge_losses(
                model, batch, raw, teacher, _ranking_scores(raw), None,
                ranking_weight=1.0, temperature=1.0, distillation_weight=0.3,
                edge_bce_weight=0.0, anchor_weight=0.1,
                anchor_weight_evidence=0.1, positive_loss_mode="sum_probability",
            )
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            gradients = student_gradient_norms(model)
            optimizer.step()
            consumed_row = {"step": step, "examples": [asdict(example) for example in batch]}
            consumed.write(json.dumps(consumed_row) + "\n")
            traces.write(json.dumps({
                "step": step,
                "batch_sha256": stable_sha(consumed_row),
                "losses": {key: float(value.detach()) for key, value in terms.items()},
                "gradient_norms": gradients,
                "candidate_ids_sha256": stable_sha([[example.query_id, list(example.candidate_ids)] for example in batch]),
                "teacher_targets_sha256": stable_sha([list(example.teacher_logits or ()) for example in batch]),
                "positive_masks_sha256": _tensor_sha(raw.positive_mask),
                "relations": dict(Counter(f"{example.source_type}->{example.destination_type}" for example in batch)),
            }) + "\n")
            execution["updates"] = step
            if step in nodes:
                execution["checkpoints"][str(step)] = _save(model, optimizer, None, "C1", step, directory)
                write_json(receipt_path, execution)
            if step % 25 == 0 or step in nodes:
                consumed.flush(); traces.flush()
                print(json.dumps({"bridge": "B1", "stage": "C1", "seed": seed, "step": step, "loss": float(terms["loss"].detach()), "elapsed": time.monotonic() - started}), flush=True)
    execution.update(
        status="completed",
        elapsed_seconds=time.monotonic() - started,
        consumption=record(consumed_path),
        traces=record(trace_path),
        actual_consumed_batch_ids_sha256=sha256(consumed_path),
    )
    write_json(receipt_path, execution)
    return execution


def train_c2(seed: int, device_name: str) -> dict[str, Any]:
    directory = _job(seed, "C2")
    receipt_path = directory / "EXECUTION.json"
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        if existing.get("status") == "completed":
            return existing
        raise FileExistsError(f"Inspect incomplete B1 C2 job: {directory}")
    c1_receipt = read_json(_job(seed, "C1") / "EXECUTION.json")
    if c1_receipt.get("status") != "completed" or c1_receipt.get("updates") != 356:
        raise RuntimeError("B1 C2 requires the completed B1 C1 step356 parent")
    parent = Path(c1_receipt["checkpoints"]["356"]["checkpoint"]["path"])
    if sha256(parent) != c1_receipt["checkpoints"]["356"]["checkpoint"]["sha256"]:
        raise ValueError("B1 C1 parent bytes changed")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    model = load_student(parent, device)
    before_anchor = _state_fingerprints(model)
    model.reset_projection_anchors()
    after_anchor = _state_fingerprints(model)
    optimizer = _optimizer(model)
    if optimizer.state:
        raise ValueError("B1 C2 optimizer must start fresh")
    examples = _merge_witness_metadata(ROOT)
    order_path = R13 / "taskD_witness_supervision/schedule_order.json"
    order = read_json(order_path)["indices"]
    ordered = [examples[index] for index in order]
    batches = [ordered[start : start + 64] for start in range(0, len(ordered), 64)]
    if len(ordered) != 11390 or len(batches) != 178:
        raise ValueError("Historical C2 graph/order does not match R27 B0")
    graph = R12 / "taskC_training/c2_candidates_seed13/path_hard.jsonl"
    directory.mkdir(parents=True, exist_ok=True)
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    execution: dict[str, Any] = {
        "status": "running",
        "stage": "B1",
        "training_stage": "C2",
        "unique_factor": "none at C2; historical full graph/logits/order retained",
        "parent": record(parent),
        "optimizer_initial_state": "fresh",
        "graph": record(graph),
        "order": record(order_path),
        "teacher_target_source": "embedded historical full-bag D/E tensors",
        "evidence_bag_policy": "full historical evidence_ids",
        "anchor_boundary": {"before": stable_sha(before_anchor), "after": stable_sha(after_anchor)},
        "updates": 0,
        "runtime": _runtime(device),
    }
    write_json(receipt_path, execution)
    store = FeatureStore.from_path(ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=60_000)
    nodes = {0, 89, 178}
    execution["checkpoints"] = {"0": _save(model, optimizer, aggregator, "C2", 0, directory)}
    started = time.monotonic()
    consumed_path = directory / "consumed_order.jsonl.gz"
    trace_path = directory / "step_traces.jsonl.gz"
    with gzip.open(consumed_path, "wt", encoding="utf-8") as consumed, gzip.open(trace_path, "wt", encoding="utf-8") as traces:
        for step, batch in enumerate(batches, 1):
            model.train()
            scores = score_target_batch(model, batch, store, device, aggregator)
            teacher = _target_teacher_scores(batch, device)
            terms = _student_path_losses(
                model, scores, teacher, None, temperature=1.0,
                distillation_weight=0.3, anchor_weight=0.1,
                anchor_weight_evidence=0.1, distillation_rows=None,
                positive_loss_mode="sum_probability",
            )
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            gradients = student_gradient_norms(model)
            optimizer.step()
            consumed_row = {"step": step, "examples": [asdict(example) for example in batch]}
            consumed.write(json.dumps(consumed_row) + "\n")
            traces.write(json.dumps({
                "step": step,
                "batch_sha256": stable_sha(consumed_row),
                "losses": {key: float(value.detach()) for key, value in terms.items()},
                "gradient_norms": gradients,
                "candidate_ids_sha256": stable_sha([[example.query_id, [candidate.target_id for candidate in example.candidates]] for example in batch]),
                "evidence_ids_sha256": stable_sha([[[candidate.target_id, list(candidate.evidence_ids)] for candidate in example.candidates] for example in batch]),
                "teacher_direct_sha256": stable_sha([list(example.teacher_direct_logits or ()) for example in batch]),
                "teacher_evidence_sha256": stable_sha([list(example.teacher_evidence_logits or ()) for example in batch]),
                "direct_positive_mask_sha256": _tensor_sha(scores.direct.positive_mask),
                "evidence_positive_mask_sha256": _tensor_sha(scores.evidence.positive_mask),
            }) + "\n")
            execution["updates"] = step
            if step in nodes:
                execution["checkpoints"][str(step)] = _save(model, optimizer, aggregator, "C2", step, directory)
                write_json(receipt_path, execution)
            if step % 25 == 0 or step in nodes:
                consumed.flush(); traces.flush()
                print(json.dumps({"bridge": "B1", "stage": "C2", "seed": seed, "step": step, "loss": float(terms["loss"].detach()), "elapsed": time.monotonic() - started}), flush=True)
    execution.update(
        status="completed",
        elapsed_seconds=time.monotonic() - started,
        consumption=record(consumed_path),
        traces=record(trace_path),
        actual_consumed_order_sha256=stable_sha([example.query_id for example in ordered]),
    )
    write_json(receipt_path, execution)
    return execution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("C1", "C2"), required=True)
    parser.add_argument("--seed", type=int, choices=(13, 29), default=13)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = train_c1(args.seed, args.device) if args.stage == "C1" else train_c2(args.seed, args.device)
    print(json.dumps(result, ensure_ascii=False))
