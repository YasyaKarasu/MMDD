"""Train the B2--B7 cumulative bridge stages after B1 confirmation.

The runner keeps the B1 modern initialization fixed while changing exactly one
registered C1/C2 component per stage.  It writes input-consumption and
checkpoint lineage receipts beside the B1 receipts.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import EdgeExample, TargetCandidate, TargetExample
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
from prepare_stage1_r27 import R13, read_json
from run_stage1_bridge import OUT, R12, R24, R25, ROOT, sha256, stable_sha, write_json
from run_stage1_r13 import _merge_witness_metadata
from run_stage1_r12_task_c import _optimizer, _ranking_scores


SEEDS = (13, 29)
HIST_TEACHER = "historical_tcore_cache"
MODERN_TEACHER = "modern"


def job(stage: str, seed: int, component: str) -> Path:
    return OUT / "training" / stage / f"seed{seed}" / component


def modern_init(seed: int) -> Path:
    return ROOT / f"work/stage1_optimization_r25_final_20260914/training/C1/seed{seed}/checkpoints/step_000000.pt"


def schedule_path(seed: int, variant: str) -> Path:
    return OUT / f"schedules/seed{seed}_steps659/{variant}.jsonl.gz"


def read_schedule(path: Path) -> list[list[dict[str, Any]]]:
    batches = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            batches.append(payload["examples"])
    return batches


def read_edge_teacher(path: Path) -> dict[tuple[str, str, str], float]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id, relation = str(row["query_id"]), str(row["relation"])
            for target_id, score in zip(row["candidate_ids"], row["scores"]):
                result[(query_id, relation, str(target_id))] = float(score)
    return result


def edge_batches(schedule: Path, teacher_path: Path, teacher_name: str) -> list[list[EdgeExample]]:
    cache = read_edge_teacher(teacher_path)
    batches = []
    for raw_batch in read_schedule(schedule):
        converted = []
        for row in raw_batch:
            candidates = tuple(map(str, row["candidate_ids"]))
            relation = f"{row['source_type']}->{row['destination_type']}"
            keys = [(str(row["query_id"]), relation, candidate) for candidate in candidates]
            if any(key not in cache for key in keys):
                missing = next(key for key in keys if key not in cache)
                raise ValueError(f"Teacher cache miss: {missing[:2]}")
            values = [cache[key] for key in keys]
            positive_id = str(row["positive_id"])
            converted.append(EdgeExample(
                query_id=str(row["query_id"]), candidate_ids=candidates,
                positive_index=candidates.index(positive_id), dataset=str(row.get("dataset", "default")), split="train",
                teacher_logits=tuple(values), teacher_checkpoint_sha256=teacher_name,
                teacher_logit_mode=teacher_name, source_type=str(row["source_type"]), destination_type=str(row["destination_type"]),
                positive_ids=tuple(map(str, row.get("positive_ids", ()))),
                confirmed_labels=None if row.get("confirmed_labels") is None else tuple(row["confirmed_labels"])))
        batches.append(converted)
    return batches


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, stage: str, component: str,
                    step: int, directory: Path, aggregator: PathAggregator | None = None) -> dict[str, Any]:
    path = directory / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint(model, "student-edge" if component == "C1" else "student-path", aggregator)
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["bridge"] = {"stage": stage, "training_stage": component, "step": step}
    torch.save(payload, path)
    return {"stage": stage, "training_stage": component, "step": step,
            "checkpoint": {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size},
            "state_fingerprints": {name: stable_sha(value.detach().cpu().tolist()) for name, value in model.state_dict().items()},
            "projection_references": student_projection_references(model)}


def runtime(device: torch.device) -> dict[str, Any]:
    return {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device), "pid": os.getpid(), "command": [sys.executable, *sys.argv]}


def train_c1(stage: str, seed: int, schedule_variant: str, teacher_kind: str, *, continuation: bool = False,
             device_name: str = "cuda:0") -> dict[str, Any]:
    component = "C1"
    directory = job(stage, seed, component)
    receipt_path = directory / "EXECUTION.json"
    resume_step: int | None = None
    previous_execution: dict[str, Any] | None = None
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        if existing.get("status") == "completed":
            return existing
        previous_execution = existing
        checkpoints = sorted(directory.glob("checkpoints/step_*.pt"))
        if checkpoints:
            resume_step = max(int(path.stem.split("_")[-1]) for path in checkpoints)
        else:
            raise FileExistsError(f"Incomplete bridge job has no safe checkpoint: {directory}")
    device = torch.device(device_name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(device); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.set_num_threads(2)
    schedule = schedule_path(seed, schedule_variant)
    teacher = (R25 / "common" / f"teacher_edge_cache_seed{seed}.jsonl.gz" if teacher_kind == HIST_TEACHER
               else R25 / "common" / f"teacher_edge_cache_seed{seed}.jsonl.gz")
    batches = edge_batches(schedule, teacher, teacher_kind)
    if resume_step is not None:
        parent = directory / "checkpoints" / f"step_{resume_step:06d}.pt"
        start_step = resume_step
        model = load_student(parent, device)
        optimizer = _optimizer(model)
        payload = torch.load(parent, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        active_batches = batches[resume_step:]
    elif continuation:
        parent_dir = job("B4", seed, "C1")
        parent = parent_dir / "checkpoints/step_000356.pt"
        if not parent.is_file():
            raise FileNotFoundError(parent)
        start_step = 356
        model = load_student(parent, device)
        optimizer = _optimizer(model)
        payload = torch.load(parent, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if len(batches) != 659:
            raise ValueError("B5 continuation requires the 659-step schedule")
        active_batches = batches[356:]
    else:
        parent = modern_init(seed)
        start_step = 0
        model = load_student(parent, device)
        optimizer = _optimizer(model)
        active_batches = batches
    expected_steps = 659 if continuation else 356
    if len(batches) != expected_steps:
        raise ValueError(f"expected {expected_steps} C1 batches, got {len(batches)}")
    directory.mkdir(parents=True, exist_ok=True)
    execution = {"status": "running", "stage": stage, "training_stage": component,
                 "unique_factor": {"B2": "modern base C1 schedule, closure off; historical T_core logits",
                                    "B3": "positive closure only on shared modern base; historical T_core logits",
                                    "B4": "modern C1 Teacher logits on B3 closure schedule",
                                    "B5": "continue B4 C1 from step356 to659"}[stage],
                 "seed": seed, "parent": {"path": str(parent.resolve()), "sha256": sha256(parent)},
                 "schedule": {"path": str(schedule.resolve()), "sha256": sha256(schedule), "variant": schedule_variant},
                 "teacher": {"path": str(teacher.resolve()), "sha256": sha256(teacher), "kind": teacher_kind},
                 "optimizer_initial_state": "resumed" if resume_step is not None else ("continued" if continuation else "fresh"), "start_step": start_step,
                 "updates": start_step, "runtime": runtime(device)}
    if resume_step is not None:
        execution["resumed_from"] = {"step": resume_step, "checkpoint": {"path": str(parent.resolve()), "sha256": sha256(parent)}}
    write_json(receipt_path, execution)
    store = FeatureStore.from_path(ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=60_000)
    nodes = {356} if continuation else {0, 178, 356}
    execution["checkpoints"] = {}
    if resume_step is None and not continuation:
        execution["checkpoints"]["0"] = save_checkpoint(model, optimizer, stage, component, 0, directory)
    elif resume_step is None:
        execution["checkpoints"]["356"] = save_checkpoint(model, optimizer, stage, component, 356, directory)
    else:
        execution["checkpoints"][str(resume_step)] = {"stage": stage, "training_stage": component, "step": resume_step,
                                                        "checkpoint": {"path": str(parent.resolve()), "sha256": sha256(parent), "bytes": parent.stat().st_size}}
    consumed_path = directory / "consumed_batches.jsonl.gz"
    trace_path = directory / "step_traces.jsonl.gz"
    started = time.monotonic()
    file_mode = "at" if resume_step is not None else "wt"
    with gzip.open(consumed_path, file_mode, encoding="utf-8") as consumed, gzip.open(trace_path, file_mode, encoding="utf-8") as traces:
        for offset, batch in enumerate(active_batches, 1):
            step = start_step + offset
            raw = score_edge_batch(model, batch, store, device, student_score_space="raw_logit")
            teacher_scores = torch.nn.utils.rnn.pad_sequence([torch.tensor(e.teacher_logits, device=device) for e in batch], batch_first=True)
            from mmdd_stage1.scoring import ListScores
            teacher_list = ListScores(teacher_scores, raw.candidate_mask, raw.positive_indices, raw.positive_mask)
            terms = _student_edge_losses(model, batch, raw, teacher_list, _ranking_scores(raw), None,
                                         ranking_weight=1.0, temperature=1.0, distillation_weight=0.3,
                                         edge_bce_weight=0.0, anchor_weight=0.1, anchor_weight_evidence=0.1,
                                         positive_loss_mode="sum_probability")
            optimizer.zero_grad(set_to_none=True); terms["loss"].backward(); gradients = student_gradient_norms(model); optimizer.step()
            consumed_row = {"step": step, "examples": [asdict(example) for example in batch]}
            consumed.write(json.dumps(consumed_row) + "\n")
            traces.write(json.dumps({"step": step, "batch_sha256": stable_sha(consumed_row),
                                     "losses": {key: float(value.detach()) for key, value in terms.items()},
                                     "candidate_ids_sha256": stable_sha([[e.query_id, list(e.candidate_ids)] for e in batch]),
                                     "teacher_targets_sha256": stable_sha([list(e.teacher_logits or ()) for e in batch]),
                                     "relations": dict(Counter(f"{e.source_type}->{e.destination_type}" for e in batch))}) + "\n")
            execution["updates"] = step
            if step in nodes or step in {500, 659}:
                execution["checkpoints"][str(step)] = save_checkpoint(model, optimizer, stage, component, step, directory)
                write_json(receipt_path, execution)
            if offset % 25 == 0 or step in nodes or step in {500, 659}:
                consumed.flush(); traces.flush()
                print(json.dumps({"stage": stage, "seed": seed, "component": component, "step": step,
                                  "loss": float(terms["loss"].detach()), "elapsed": time.monotonic() - started}), flush=True)
    execution.update(status="completed", elapsed_seconds=time.monotonic() - started,
                     consumption={"path": str(consumed_path.resolve()), "sha256": sha256(consumed_path)},
                     traces={"path": str(trace_path.resolve()), "sha256": sha256(trace_path)},
                     actual_consumed_batch_ids_sha256=sha256(consumed_path))
    write_json(receipt_path, execution)
    return execution


def path_examples(mode: str) -> list[TargetExample]:
    examples = _merge_witness_metadata(ROOT)
    if mode in ("first8", "modern_first8"):
        examples = [replace(example, candidates=tuple(replace(candidate, evidence_ids=candidate.evidence_ids[:8])
                                                       for candidate in example.candidates)) for example in examples]
    if mode == "modern_first8":
        cache_path = R25 / "common/teacher_native_path_cache.jsonl.gz"
        cache = {}
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line); cache[row["query_id"]] = row
        converted = []
        for example in examples:
            row = cache.get(example.query_id)
            if row is None or tuple(row["candidate_ids"]) != tuple(candidate.target_id for candidate in example.candidates):
                raise ValueError(f"modern path Teacher cache mismatch: {example.query_id}")
            converted.append(replace(example, teacher_direct_logits=tuple(map(float, row["direct_logits"])),
                                     teacher_evidence_logits=tuple(map(float, row["evidence_logits"])),
                                     teacher_logit_mode="modern_native_path_cache"))
        examples = converted
    return examples


def train_c2(stage: str, seed: int, parent_step: int, graph_mode: str, teacher_mode: str,
             device_name: str = "cuda:0") -> dict[str, Any]:
    component = "C2"
    directory = job(stage, seed, component)
    receipt_path = directory / "EXECUTION.json"
    resume_step: int | None = None
    previous_execution: dict[str, Any] | None = None
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        if existing.get("status") == "completed":
            return existing
        previous_execution = existing
        checkpoints = sorted(directory.glob("checkpoints/step_*.pt"))
        if checkpoints:
            resume_step = max(int(path.stem.split("_")[-1]) for path in checkpoints)
        else:
            raise FileExistsError(f"Incomplete bridge job has no safe checkpoint: {directory}")
    parent = (directory / "checkpoints" / f"step_{resume_step:06d}.pt" if resume_step is not None
              else job(stage, seed, "C1") / "checkpoints" / f"step_{parent_step:06d}.pt")
    if not parent.is_file():
        raise FileNotFoundError(parent)
    device = torch.device(device_name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(device); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.set_num_threads(2)
    model = load_student(parent, device)
    if resume_step is None:
        before_anchor = student_projection_references(model); model.reset_projection_anchors(); after_anchor = student_projection_references(model)
    else:
        before_anchor = student_projection_references(model); after_anchor = before_anchor
    optimizer = _optimizer(model)
    if resume_step is not None:
        payload = torch.load(parent, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    examples = path_examples(graph_mode)
    order_path = R13 / "taskD_witness_supervision/schedule_order.json"
    order = read_json(order_path)["indices"]
    ordered = [examples[index] for index in order]
    batches = [ordered[start:start + 64] for start in range(0, len(ordered), 64)]
    if len(ordered) != 11390 or len(batches) != 178:
        raise ValueError("C2 historical order must contain 11,390 examples and 178 batches")
    # ``path_examples`` starts from the historical path_hard candidate graph
    # for every bridge stage; first8/modern_first8 only transform each
    # evidence_ids bag and (for B7) replace the Teacher tensors.  Keep the
    # receipt pointed at the graph actually consumed, rather than the modern
    # comparison graph used by the descriptive audit.
    graph = R12 / "taskC_training/c2_candidates_seed13/path_hard.jsonl"
    graph_policy = "historical full evidence_ids" if graph_mode == "full" else "historical candidate graph with evidence_ids[:8]"
    directory.mkdir(parents=True, exist_ok=True)
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    execution = previous_execution or {"status": "running", "stage": stage, "training_stage": component, "seed": seed,
                 "unique_factor": {"B2": "historical full graph and logits", "B3": "historical full graph and logits",
                                    "B4": "historical full graph and logits", "B5": "historical full graph and logits",
                                    "B6": "Student first8 graph; historical full-bag Teacher logits",
                                    "B7": "first8 graph and modern native path Teacher logits"}[stage],
                 "parent": {"path": str(parent.resolve()), "sha256": sha256(parent)},
                 "graph": {"path": str(graph.resolve()), "sha256": sha256(graph), "mode": graph_mode, "policy": graph_policy},
                 "order": {"path": str(order_path.resolve()), "sha256": sha256(order_path)},
                 "teacher_mode": teacher_mode, "optimizer_initial_state": "fresh",
                 "anchor_boundary": {"before": stable_sha(before_anchor), "after": stable_sha(after_anchor)},
                 "updates": resume_step or 0, "runtime": runtime(device)}
    execution.update(status="running", runtime=runtime(device), updates=resume_step or 0)
    if resume_step is not None:
        execution["resumed_from"] = {"step": resume_step, "checkpoint": {"path": str(parent.resolve()), "sha256": sha256(parent)}}
    write_json(receipt_path, execution)
    store = FeatureStore.from_path(ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=60_000)
    if resume_step is None:
        execution["checkpoints"] = {"0": save_checkpoint(model, optimizer, stage, component, 0, directory, aggregator)}
    else:
        execution.setdefault("checkpoints", {})
        execution["checkpoints"][str(resume_step)] = {"stage": stage, "training_stage": component, "step": resume_step,
                                                        "checkpoint": {"path": str(parent.resolve()), "sha256": sha256(parent), "bytes": parent.stat().st_size}}
    consumed_path = directory / "consumed_order.jsonl.gz"; trace_path = directory / "step_traces.jsonl.gz"; started = time.monotonic()
    file_mode = "at" if resume_step is not None else "wt"
    active_batches = batches[resume_step:] if resume_step is not None else batches
    start_step = resume_step or 0
    with gzip.open(consumed_path, file_mode, encoding="utf-8") as consumed, gzip.open(trace_path, file_mode, encoding="utf-8") as traces:
        for offset, batch in enumerate(active_batches, 1):
            step = start_step + offset
            scores = score_target_batch(model, batch, store, device, aggregator)
            teacher_scores = _target_teacher_scores(batch, device)
            terms = _student_path_losses(model, scores, teacher_scores, None, temperature=1.0, distillation_weight=0.3,
                                         anchor_weight=0.1, anchor_weight_evidence=0.1, distillation_rows=None,
                                         positive_loss_mode="sum_probability")
            optimizer.zero_grad(set_to_none=True); terms["loss"].backward(); gradients = student_gradient_norms(model); optimizer.step()
            consumed_row = {"step": step, "examples": [asdict(example) for example in batch]}; consumed.write(json.dumps(consumed_row) + "\n")
            traces.write(json.dumps({"step": step, "batch_sha256": stable_sha(consumed_row),
                                     "losses": {key: float(value.detach()) for key, value in terms.items()},
                                     "candidate_ids_sha256": stable_sha([[e.query_id, [c.target_id for c in e.candidates]] for e in batch]),
                                     "evidence_ids_sha256": stable_sha([[[c.target_id, list(c.evidence_ids)] for c in e.candidates] for e in batch]),
                                     "teacher_direct_sha256": stable_sha([list(e.teacher_direct_logits or ()) for e in batch]),
                                     "teacher_evidence_sha256": stable_sha([list(e.teacher_evidence_logits or ()) for e in batch])}) + "\n")
            execution["updates"] = step
            if step in {89, 178}:
                execution["checkpoints"][str(step)] = save_checkpoint(model, optimizer, stage, component, step, directory, aggregator); write_json(receipt_path, execution)
            if step % 25 == 0 or step in {89, 178}:
                consumed.flush(); traces.flush(); print(json.dumps({"stage": stage, "seed": seed, "component": component, "step": step,
                                                                       "loss": float(terms["loss"].detach()), "elapsed": time.monotonic() - started}), flush=True)
    execution.update(status="completed", elapsed_seconds=time.monotonic() - started,
                     consumption={"path": str(consumed_path.resolve()), "sha256": sha256(consumed_path)},
                     traces={"path": str(trace_path.resolve()), "sha256": sha256(trace_path)},
                     actual_consumed_order_sha256=stable_sha([example.query_id for example in ordered]))
    write_json(receipt_path, execution)
    return execution


def inherit_c1(stage: str, seed: int, *, source_stage: str = "B5") -> dict[str, Any]:
    """Carry the identical B5 C1 step659 node into B6/B7 without retraining."""
    directory = job(stage, seed, "C1")
    receipt_path = directory / "EXECUTION.json"
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        if existing.get("status") == "completed":
            return existing
        raise FileExistsError(directory)
    source = job(source_stage, seed, "C1") / "checkpoints/step_000659.pt"
    source_receipt = job(source_stage, seed, "C1") / "EXECUTION.json"
    if not source.is_file() or not source_receipt.is_file():
        raise FileNotFoundError(source)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "checkpoints/step_000659.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    payload = {"status": "completed", "stage": stage, "training_stage": "C1", "seed": seed,
               "unique_factor": "none at C1; exact B5 step659 parent reused",
               "parent": {"path": str(source.resolve()), "sha256": sha256(source)},
               "source_execution": {"path": str(source_receipt.resolve()), "sha256": sha256(source_receipt)},
               "updates": 659, "optimizer_initial_state": "inherited_node_only",
               "checkpoints": {"659": {"stage": stage, "training_stage": "C1", "step": 659,
                                        "checkpoint": {"path": str(destination.resolve()), "sha256": sha256(destination), "bytes": destination.stat().st_size}}}}
    write_json(receipt_path, payload)
    return payload


def run(stage: str, seed: int, device_name: str) -> dict[str, Any]:
    configs = {
        "B2": ("base_first356", HIST_TEACHER, False, 356, "full", HIST_TEACHER),
        "B3": ("closure_first356", HIST_TEACHER, False, 356, "full", HIST_TEACHER),
        "B4": ("closure_first356", MODERN_TEACHER, False, 356, "full", HIST_TEACHER),
        "B5": ("closure_full", MODERN_TEACHER, True, 659, "full", HIST_TEACHER),
        "B6": ("closure_full", MODERN_TEACHER, True, 659, "first8", HIST_TEACHER),
        "B7": ("closure_full", MODERN_TEACHER, True, 659, "modern_first8", MODERN_TEACHER),
    }
    schedule_variant, teacher_kind, continuation, parent_step, graph_mode, teacher_mode = configs[stage]
    c1 = inherit_c1(stage, seed) if stage in {"B6", "B7"} else train_c1(stage, seed, schedule_variant, teacher_kind, continuation=continuation, device_name=device_name)
    c2 = train_c2(stage, seed, parent_step, graph_mode, teacher_mode, device_name=device_name)
    return {"C1": c1, "C2": c2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("B2", "B3", "B4", "B5", "B6", "B7"), required=True)
    parser.add_argument("--seed", action="append", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for seed in args.seed or list(SEEDS):
        print(json.dumps(run(args.stage, seed, args.device), ensure_ascii=False))
