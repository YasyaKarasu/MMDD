"""Continue fixed T0 on one of the three preregistered five-relation controls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch

from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore
from prepare_stage1_r26 import ROOT, OUT, file_record, stable_sha
from prepare_stage1_r26_refinement import T0
from run_stage1_r19 import load_r19_checkpoint, backward_logical_batch, _checkpoint_payload
from run_stage1_r21 import paths, read_rows
from run_stage1_r25 import _json, _r25_teacher_feature_paths, sha256


def train(arm: str, seed: int, device_name: str) -> dict:
    torch.set_num_threads(2)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    job = OUT / "feedback/refinement" / arm / f"seed{seed}"
    protocol_path = OUT / "feedback/REFINEMENT_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    gate = json.loads((OUT / "feedback/REFINEMENT_GATE_FROZEN.json").read_text())
    if gate["status"] != "triggered":
        raise ValueError("Refinement budget gate did not trigger")
    lists_receipt = json.loads((job / "LISTS_RECEIPT.json").read_text())
    for key in ("lists", "order", "protocol", "gate", "hard_source"):
        record = lists_receipt[key]
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Refinement input changed: {key}")
    examples = load_edge_examples(Path(lists_receipt["lists"]["path"]), split="train")
    order_path = Path(lists_receipt["order"]["path"])
    schedule = list(read_rows(order_path))
    if len(schedule) != protocol["total_updates"] or len(examples) != protocol["list_budget"]:
        raise ValueError("Refinement budget changed")
    for epoch in (1, 2):
        consumed = [i for row in schedule if row["epoch"] == epoch for i in row["source_rows"]]
        if sorted(consumed) != list(range(len(examples))):
            raise ValueError("Each epoch must consume every original relation list exactly once")
    signature = {"arm":arm, "seed":seed, "protocol":file_record(protocol_path),
                 "parent":file_record(T0), "lists_receipt":file_record(job / "LISTS_RECEIPT.json"),
                 "evaluation_protocol":file_record(OUT / "feedback/REFINEMENT_EVALUATION_PROTOCOL.json"),
                 "code":{name:sha256(ROOT / "src" / name) for name in (
                     "train_stage1_r26_refinement.py", "mmdd_stage1/scoring.py", "run_stage1_r19.py",
                     "mmdd_stage1/models.py", "mmdd_stage1/features.py")}}
    if not signature["evaluation_protocol"]["exists"]:
        raise ValueError("Freeze common evaluation pools before continuation")
    receipt_path = job / "TRAINING_RECEIPT.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["signature"] != signature or sha256(Path(receipt["final_checkpoint"]["path"])) != receipt["final_checkpoint"]["sha256"]:
            raise ValueError("Existing refinement training identity changed")
        return {"arm":arm, "seed":seed, "status":"verified_cached"}
    _, _, _, model, parent = load_r19_checkpoint(T0, device)
    model.train()
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("Refinement must retain every Teacher trainable component")
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["optimizer"]["lr"],
                                 weight_decay=protocol["optimizer"]["weight_decay"])
    if optimizer.state or {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in model.parameters()}:
        raise ValueError("All controls require fresh AdamW over all parameters")
    checkpoints = job / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    history_path = job / "steps.jsonl"
    if history_path.exists():
        raise ValueError("Partial training exists; audit its last checkpoint before resuming")
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if any(not torch.equal(v, parent["state_dict"][k].cpu()) for k,v in initial.items()):
        raise ValueError("Continuation initialization differs from fixed T0")
    torch.save({**_checkpoint_payload(model, arm, seed, 0, optimizer=optimizer), "r26_signature":signature},
               checkpoints / "step_000000.pt")
    _json(job / "INITIALIZATION_AUDIT.json", {"signature":signature, "exact_parent_tensors":len(initial),
          "all_parameters_trainable":True, "optimizer_initial_states":len(optimizer.state),
          "trainable_parameter_names":[n for n,_ in model.named_parameters()],
          "initial_checkpoint":file_record(checkpoints / "step_000000.pt")})
    store = FeatureStore.from_path(paths(ROOT)["features"], cache_size=8192, cache_bytes=2*1024**3,
                                   teacher_paths=_r25_teacher_feature_paths(ROOT))
    started = time.monotonic()
    gradients_seen = Counter()
    total_slots = 0
    with history_path.open("w") as handle:
        for expected_step, row in enumerate(schedule, 1):
            if row["step"] != expected_step:
                raise ValueError("Order step sequence changed")
            batch = [examples[i] for i in row["source_rows"]]
            optimizer.zero_grad(set_to_none=True)
            loss, slots = backward_logical_batch(model, batch, store, device,
                                                 microbatch_lists=protocol["microbatch_lists"])
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    value = float(parameter.grad.detach().norm())
                    if not torch.isfinite(parameter.grad).all():
                        raise ValueError("Nonfinite refinement gradient")
                    gradients[name] = value
                    gradients_seen[name] += value > 0
            optimizer.step()
            total_slots += slots
            record = {**row, "loss":loss, "pair_slots":slots, "gradient_norms":gradients,
                      "query_ids":[e.query_id for e in batch],
                      "relations":[f"{e.source_type}->{e.destination_type}" for e in batch]}
            handle.write(json.dumps(record)+"\n")
            if expected_step % 500 == 0:
                handle.flush()
                print(json.dumps({"arm":arm,"seed":seed,"step":expected_step,"loss":loss,
                                  "elapsed_seconds":time.monotonic()-started}), flush=True)
            if expected_step % protocol["updates_per_epoch"] == 0:
                torch.save({**_checkpoint_payload(model,arm,seed,expected_step,optimizer=optimizer),
                            "r26_signature":signature}, checkpoints / f"step_{expected_step:06d}.pt")
    final = checkpoints / f"step_{protocol['total_updates']:06d}.pt"
    result = {"execution_status":"ran", "scientific_validity":"valid", "signature":signature,
              "optimizer_updates":len(schedule), "pair_slots":total_slots,
              "nonzero_gradient_steps":dict(gradients_seen),
              "parameter_delta_norms":{n:float((p.detach().cpu()-initial[n]).norm()) for n,p in model.named_parameters()},
              "epochs":protocol["epochs"], "elapsed_seconds":time.monotonic()-started, "device":device_name,
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
              "history":file_record(history_path), "final_checkpoint":file_record(final),
              "training_run_id":stable_sha(signature)}
    _json(receipt_path,result)
    return {"arm":arm,"seed":seed,"updates":len(schedule),"final_checkpoint":str(final)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=("Tcont","Told","Tnew"),required=True)
    parser.add_argument("--seed",type=int,choices=(13,29),required=True)
    parser.add_argument("--device",required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.arm,args.seed,args.device)))
