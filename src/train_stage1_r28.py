"""Execute exactly one of the twelve R28 continuations, with fixed epoch receipts."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.r28_objectives import backward_batch
from prepare_stage1_r26 import parameter_sha
from prepare_stage1_r27 import record, sha, stable_sha, write_json
from prepare_stage1_r28 import ROOT, OUT, FAMILIES, inputs, registry, feature_store
from run_stage1_r13 import _merge_witness_metadata, _optimizer
from run_stage1_r19 import load_r19_checkpoint


def setup(arm: str, device: torch.device):
    student = arm.startswith("S-")
    parent = Path(inputs()["student_parent" if student else "teacher_parent"]["path"])
    model = load_student(parent, device) if student else load_r19_checkpoint(parent, device)[3]
    if student:
        model.reset_projection_anchors()
    manifest = json.loads((OUT / "R28_PREPARED_MANIFEST.json").read_text())
    optimizer = _optimizer(model) if student else torch.optim.AdamW(model.parameters(),
        lr=manifest["teacher"]["lr"], weight_decay=manifest["teacher"]["weight_decay"])
    assert not optimizer.state
    return model.train(), optimizer, parent


def train(arm: str, seed: int, device_name: str, *, smoke: bool = False) -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    student = arm.startswith("S-")
    kind = "student" if student else "teacher"
    prepared = json.loads((OUT / "R28_PREPARED_MANIFEST.json").read_text())
    if not smoke:
        audit = json.loads((OUT / "R28_OBJECTIVE_AUDIT.json").read_text())
        assert audit["status"] == "pass"
        for name, rec in audit["source_identity"].items():
            assert sha(Path(rec["path"])) == rec["sha256"], name
    frozen = json.loads((OUT / "INPUT_HASHES.json").read_text())
    for name, rec in frozen.items():
        assert sha(Path(rec["path"])) == rec["sha256"], name
    batch_size = prepared[kind]["batch"]
    microbatch = 8 if student else 1
    job = OUT / ("smoke" if smoke else kind) / arm / f"seed{seed}"
    if (job / "EXECUTION.json").exists():
        raise FileExistsError("Inspect existing job; never silently restart or overwrite")
    model, optimizer, parent = setup(arm, device)
    trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    assert trainable == prepared["trainable"][kind]["names"]
    store = feature_store(not student)
    examples, known = _merge_witness_metadata(ROOT), registry()
    orders = json.loads((OUT / "common/orders.json").read_text())[str(seed)]
    names = {id(p): n for n,p in model.named_parameters()}
    signature = {"arm": arm, "seed": seed, "parent": record(parent),
        "graph": frozen["target_path_graph"], "order_hashes": [stable_sha(o) for o in orders],
        "objective_family": FAMILIES[arm], "split": FAMILIES[arm] != "EDGE",
        "aggregator": None if FAMILIES[arm] == "EDGE" else FAMILIES[arm],
        "trainable_names": trainable, "trainable_names_hash": stable_sha(trainable),
        "optimizer": [{**{k:v for k,v in g.items() if k != "params"},
                       "parameters": [names[id(p)] for p in g["params"]]} for g in optimizer.param_groups],
        "optimizer_initial_state": "fresh", "logical_batch": batch_size, "microbatch": microbatch,
        "optional_evidence_reduction": "usable-E-list mean across entire logical batch",
        "code": {str(p.relative_to(ROOT)): record(p) for p in (Path(__file__), ROOT/"src/mmdd_stage1/r28_objectives.py")},
        "dtype": "float32", "device": device_name, "smoke_no_optimizer_updates": smoke}
    write_json(job / "RUN_IDENTITY.json", signature)
    execution = {"status": "running", "pid": os.getpid(), "updates": 0, "checkpoints": [], "signature": signature}
    write_json(job / "EXECUTION.json", execution)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    def save(step: int, epoch: float) -> dict:
        path = job / "checkpoints" / f"step_{step:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "model_kind": "student" if student else "teacher_r19_global",
                   "completed_stage": "r28", "r19_arm": arm, "continuation_seed": seed,
                   "optimizer_step": step, "epoch": epoch, "config": model.config(),
                   "state_dict": {n:p.detach().cpu() for n,p in model.state_dict().items()},
                   "optimizer_state_dict": optimizer.state_dict(), "r28_signature": signature}
        torch.save(payload, path)
        rec = {"epoch": epoch, "updates": step, "parameter_sha256": parameter_sha(model), **record(path)}
        write_json(path.with_suffix(".json"), rec)
        return rec
    if not smoke:
        execution["checkpoints"].append(save(0, 0))
    try:
        with (job / "history.jsonl").open("w") as log:
            for epoch, order in enumerate(orders, 1):
                for start in range(0, len(order), batch_size):
                    indices = order[start:start+batch_size]
                    batch = [examples[i] for i in indices]
                    optimizer.zero_grad(set_to_none=True)
                    terms = backward_batch(model, batch, store, device, FAMILIES[arm], known,
                                           student=student, microbatch=microbatch)
                    assert math.isfinite(terms["loss"]), "Nonfinite training loss"
                    gradients = {n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
                    assert gradients and all(math.isfinite(v) for v in gradients.values())
                    if not smoke:
                        optimizer.step()
                        execution["updates"] += 1
                    step = execution["updates"]
                    trace = {"step": step, "epoch": epoch, "batch_start": start, "source_indices": indices,
                             "order_hash": stable_sha(indices), "losses": terms,
                             "gradient_norms": gradients if smoke or step == 1 or step % 100 == 0 else None,
                             "elapsed_seconds": time.monotonic()-started}
                    log.write(json.dumps(trace) + "\n")
                    log.flush()
                    if smoke:
                        execution.update({"status": "completed", "smoke": trace,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
                        write_json(job / "EXECUTION.json", execution)
                        return execution
                    local_step = start // batch_size + 1
                    checkpoint_epoch = (.5 if epoch == 1 and local_step == math.ceil(prepared[kind]["updates_per_epoch"] / 2)
                                        else epoch if local_step == prepared[kind]["updates_per_epoch"] and epoch in (1,2,3,5) else None)
                    if checkpoint_epoch is not None:
                        execution["checkpoints"].append(save(step, checkpoint_epoch))
                    if step % 10 == 0 or checkpoint_epoch is not None:
                        execution.update({"elapsed_seconds": time.monotonic()-started, "epoch": epoch,
                                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
                        write_json(job / "EXECUTION.json", execution)
                        print(json.dumps({"arm": arm, "seed": seed, "step": step, "epoch": epoch,
                                          "loss": terms["loss"], "elapsed": execution["elapsed_seconds"]}), flush=True)
        execution["status"] = "completed"
    except Exception as exc:
        execution.update({"status": "failed", "exception_type": type(exc).__name__})
        write_json(job / "EXECUTION.json", execution)
        raise
    execution.update({"elapsed_seconds": time.monotonic()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
    write_json(job / "EXECUTION.json", execution)
    return {"arm": arm, "seed": seed, "updates": execution["updates"], "status": execution["status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=FAMILIES)
    parser.add_argument("--seed", required=True, type=int, choices=(13,29))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.arm, args.seed, args.device, smoke=args.smoke)))
