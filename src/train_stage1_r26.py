"""Run a full R26 C2 from the tensor-audited matching R25 C1 endpoint."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from mmdd_stage1.b13_recipe import path_objective
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.r25_objectives import split_objective
from mmdd_stage1.r26_training import edge_query_loss, graph_edges
from mmdd_stage1.scoring import score_edge_batch, score_target_batch
from mmdd_stage1.training import _anchor_losses, student_gradient_norms
from prepare_stage1_r26 import ROOT, OUT, file_record, parameter_sha, stable_sha
from run_stage1_r21 import paths, read_rows, write_rows
from run_stage1_r22_f0 import _optimizer
from run_stage1_r25 import _json, _load_target_cache, _r25_path_pool, _target_cache_scores, out as r25_out, sha256


def known_edges() -> dict:
    registry = defaultdict(set)
    for row in read_rows(paths(ROOT)["train"]):
        positives = set(row.get("positive_ids", []))
        if row.get("positive_id"):
            positives.add(row["positive_id"])
        registry[(row["query_id"], row["source_type"], row["destination_type"])].update(positives)
    return registry


def path_terms(model, scores, arm: str, teacher=None) -> dict:
    """The exact R25 factories, with only example consumption order changed."""
    if arm == "O-NATIVE":
        return path_objective(model, scores, teacher)
    if arm != "O-SUP":
        raise ValueError(arm)
    _, anchor = _anchor_losses(model, .1, .1)
    return split_objective(scores, arm="SPLIT-SUP", anchor_loss=anchor)


def train(arm: str, seed: int, device_name: str) -> dict:
    torch.set_num_threads(3)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    audit = json.loads((OUT / "acceptance/C1_TENSOR_AUDIT.json").read_text())
    if not audit["c1"][str(seed)]["reuse_valid"]:
        raise ValueError("C1 tensor/coverage audit did not permit reuse")
    parent = r25_out(ROOT) / f"training/C1/seed{seed}/checkpoints/step_000659.pt"
    graph = _r25_path_pool(ROOT)
    order_path = OUT / "common/c2_order.jsonl"
    native_cache_path = r25_out(ROOT) / "common/teacher_native_path_cache.jsonl.gz"
    signature = {"arm": arm, "seed": seed, "parent": file_record(parent), "graph_sha256": sha256(graph),
                 "order_sha256": sha256(order_path), "protocol_sha256": sha256(OUT / "PROTOCOL.json"),
                 "registry_sha256": sha256(paths(ROOT)["train"]),
                 "feature_sha256": sha256(paths(ROOT)["features"] / "manifest.jsonl"),
                 "teacher_native_cache": file_record(native_cache_path) if arm == "O-NATIVE" else None,
                 "code_sha256": {name: sha256(ROOT / "src" / name) for name in (
                     "train_stage1_r26.py", "mmdd_stage1/r26_training.py", "mmdd_stage1/r25_objectives.py",
                     "mmdd_stage1/b13_recipe.py", "mmdd_stage1/scoring.py", "mmdd_stage1/models.py",
                     "mmdd_stage1/objectives.py", "mmdd_stage1/training.py", "run_stage1_r22_f0.py")}}
    job = OUT / "training" / arm / f"seed{seed}"
    job.mkdir(parents=True, exist_ok=True)
    receipt_path = job / "C2_COMPLETION_RECEIPT.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous["signature"] != signature:
            raise ValueError("Training identity differs from existing receipt")
        return previous
    if (job / "RUN_IDENTITY.json").exists() and json.loads((job / "RUN_IDENTITY.json").read_text()) != signature:
        raise ValueError("Incomplete run identity differs; do not silently overwrite")
    _json(job / "RUN_IDENTITY.json", signature)
    graph_examples = load_target_examples(graph, split="train")
    order = list(read_rows(order_path))
    examples = [graph_examples[row["source_row"]] for row in order]
    if any(example.query_id != row["query_id"] for example, row in zip(examples, order)):
        raise ValueError("Frozen query order does not match graph")
    model = load_student(parent, device).train()
    optimizer = _optimizer(model)
    store = FeatureStore.from_path(paths(ROOT)["features"], cache_size=120000)
    native_cache = _load_target_cache(native_cache_path) if arm == "O-NATIVE" else None
    registry = known_edges()
    closure = []
    for example in examples:
        ids = {c.target_id for c in example.candidates}
        missing = (registry[(example.query_id, "table", "table")] & ids) - set(example.positive_target_ids)
        if missing:
            closure.append({"q": example.query_id, "missing": sorted(missing)})
    _json(job / "TARGET_CLOSURE.json", {"checked_queries": len(examples), "violations": closure})
    if closure:
        raise ValueError("Graph has known positive target mislabeled; only-order requires graph correction disclosure")
    aux = [graph_edges(example, registry, lambda oid: store.embedding_features(oid).object_type) for example in examples]
    # Both path and Edge arms freeze this same q-bound graph, even when path SUP does not consume auxiliary labels.
    write_rows(job / "query_graph_edges.jsonl.gz", ({"query_id": example.query_id, "lists": [asdict(e) for e in rows]} for example, rows in zip(examples, aux)))
    steps = math.ceil(len(examples) / 64)
    (job / "checkpoints").mkdir(exist_ok=True)
    initial_hash = parameter_sha(model)
    start_time = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    def save(step: int) -> dict:
        path = job / "checkpoints" / f"step_{step:06d}.pt"
        torch.save({"format_version": 1, "model_kind": "student", "completed_stage": "r26-c2",
                    "arm": arm, "seed": seed, "step": step, "config": model.config(),
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(), "r26_signature": signature}, path)
        return {"step": step, "parameter_sha256": parameter_sha(model), **file_record(path)}

    checkpoints = [save(0)]
    history = []
    with (job / "train_history.jsonl").open("w") as log:
        for step, start in enumerate(range(0, len(examples), 64), 1):
            batch = examples[start:start+64]
            optimizer.zero_grad(set_to_none=True)
            if arm == "E-GRAPH":
                edges, owners = [], []
                for owner, rows in enumerate(aux[start:start+len(batch)]):
                    edges.extend(rows)
                    owners.extend([owner] * len(rows))
                edge_scores = score_edge_batch(model, edges, store, device, student_score_space="raw_logit")
                _, anchor = _anchor_losses(model, .1, .1)
                terms = edge_query_loss(edge_scores, edges, owners, len(batch), anchor)
                e_active = None
            else:
                scores = score_target_batch(model, batch, store, device,
                    PathAggregator("logsumexp", 4 if arm == "O-NATIVE" else 8, path_combination="sum"), student_score_space="raw_logit")
                if arm == "O-NATIVE":
                    teacher = _target_cache_scores(batch, native_cache, device, qt_evidence=False)
                else:
                    teacher = None
                terms = path_terms(model, scores, arm, teacher)
                e_active = int((scores.evidence.positive_mask & scores.evidence.candidate_mask).any(-1).sum())
            terms["loss"].backward()
            gradients = student_gradient_norms(model)
            optimizer.step()
            row = {"step": step, "query_ids": [e.query_id for e in batch], "E_positive_active": e_active,
                   "terms": {k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                   "gradient": gradients, "elapsed_seconds": time.monotonic() - start_time}
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step in (math.ceil(steps / 2), steps):
                checkpoints.append(save(step))
            if step % 10 == 0:
                print(json.dumps({"arm": arm, "seed": seed, "step": step, "total": steps, "loss": row["terms"]["loss"]}), flush=True)
    receipt = {"signature": signature, "execution_status": "ran", "scientific_validity": "valid",
               "optimizer_initial_state": "fresh", "optimizer_updates": steps, "coverage_lists": len(examples),
               "initial_parameter_sha256": initial_hash, "final_parameter_sha256": parameter_sha(model),
               "checkpoints": checkpoints, "elapsed_seconds": time.monotonic() - start_time,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "graph_edges": file_record(job / "query_graph_edges.jsonl.gz"),
               "history": file_record(job / "train_history.jsonl")}
    _json(receipt_path, receipt)
    return {"arm": arm, "seed": seed, "updates": steps, "elapsed_seconds": receipt["elapsed_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("O-NATIVE", "O-SUP", "E-GRAPH"), required=True)
    parser.add_argument("--seed", type=int, choices=(13, 29), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(train(args.arm, args.seed, args.device)))
