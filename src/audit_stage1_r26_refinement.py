"""Audit actual continuation histories, checkpoint tensors and optimizer states."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import torch

from package_stage1_r26 import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/stage1_optimization_r26_20260914"


def audit_history(history: list[dict], order: list[dict], examples: list[dict]) -> dict:
    """Reconstruct consumed query/relation/candidate counts from the actual lists."""
    if len(history) != len(order):
        raise ValueError("History does not cover the full frozen update budget")
    present, nonzero, relations = Counter(), Counter(), Counter()
    slots = 0
    for step, (observed, expected) in enumerate(zip(history, order), 1):
        if expected["step"] != step or any(observed[k] != v for k, v in expected.items()):
            raise ValueError("Actual consumed update order differs")
        batch = [examples[i] for i in expected["source_rows"]]
        expected_relations = [f"{r['source_type']}->{r['destination_type']}" for r in batch]
        expected_slots = sum(len(r["candidate_ids"]) for r in batch)
        if observed["query_ids"] != [r["query_id"] for r in batch] or observed["relations"] != expected_relations:
            raise ValueError("Actual queries or relations differ from the frozen lists")
        if observed["pair_slots"] != expected_slots or not math.isfinite(observed["loss"]):
            raise ValueError("Actual pair slots or loss invalid")
        for name, norm in observed["gradient_norms"].items():
            if not math.isfinite(norm) or norm < 0:
                raise ValueError("Invalid actual gradient norm")
            present[name] += 1
            nonzero[name] += norm > 0
        relations.update(expected_relations)
        slots += expected_slots
    for epoch in {r["epoch"] for r in order}:
        consumed = [i for r in order if r["epoch"] == epoch for i in r["source_rows"]]
        if sorted(consumed) != list(range(len(examples))):
            raise ValueError("Epoch did not consume every relation list exactly once")
    return {"updates": len(history), "pair_slots": slots, "relations": dict(relations),
            "gradient_present_steps": dict(present), "nonzero_gradient_steps": dict(nonzero)}


def run() -> dict:
    torch.set_num_threads(2)
    verified = {}

    def check(record):
        path = Path(record["path"])
        if str(path) not in verified:
            verified[str(path)] = digest(path)
        if verified[str(path)] != record["sha256"]:
            raise ValueError(f"Refinement artifact changed: {path}")
        return path

    def read_rows(path):
        with path.open() as stream:
            return [json.loads(line) for line in stream]

    protocol_path = OUT / "feedback/REFINEMENT_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    if (protocol["list_budget"], protocol["total_updates"], protocol["updates_per_epoch"], protocol["epochs"]) != (42143,10536,5268,2):
        raise ValueError("Frozen continuation budget differs from contract")
    rows, pending, orders = [], [], {}
    for seed in (13,29):
        for arm in ("Tcont","Told","Tnew"):
            job = OUT / f"feedback/refinement/{arm}/seed{seed}"
            path = job / "TRAINING_RECEIPT.json"
            if not path.exists():
                pending.append(str(path))
                continue
            receipt = json.loads(path.read_text())
            signature = receipt["signature"]
            for key in ("parent","protocol","lists_receipt","evaluation_protocol"):
                check(signature[key])
            for name, sha in signature["code"].items():
                check({"path": str(ROOT / "src" / name), "sha256": sha})
            inputs = json.loads((job / "LISTS_RECEIPT.json").read_text())
            for key in ("lists","order","protocol","gate","hard_source"):
                check(inputs[key])
            order = read_rows(Path(inputs["order"]["path"]))
            order_hash = inputs["order"]["sha256"]
            if seed in orders and orders[seed] != order_hash:
                raise ValueError("Teacher controls did not use identical per-seed order")
            orders[seed] = order_hash
            examples = read_rows(Path(inputs["lists"]["path"]))
            actual = audit_history(read_rows(check(receipt["history"])), order, examples)
            if len(examples) != 42143 or actual["updates"] != 10536 or receipt["optimizer_updates"] != 10536:
                raise ValueError("Incomplete actual continuation coverage")
            if actual["pair_slots"] != receipt["pair_slots"] or actual["nonzero_gradient_steps"] != receipt["nonzero_gradient_steps"]:
                raise ValueError("Reported gradient or pair counts differ from actual history")
            init = json.loads((job / "INITIALIZATION_AUDIT.json").read_text())
            start = torch.load(check(init["initial_checkpoint"]), map_location="cpu", weights_only=True)
            parent = torch.load(check(signature["parent"]), map_location="cpu", weights_only=True)
            if start["state_dict"].keys() != parent["state_dict"].keys() or any(
                    not torch.equal(v,parent["state_dict"][k]) for k,v in start["state_dict"].items()):
                raise ValueError("Actual starting tensors differ from common T0")
            if start["optimizer_state_dict"]["state"] or not init["all_parameters_trainable"]:
                raise ValueError("Continuation must start with fresh AdamW and all trainable components")
            del parent
            checkpoint_records = []
            for step in (0,5268,10536):
                checkpoint = job / f"checkpoints/step_{step:06d}.pt"
                payload = start if step == 0 else torch.load(checkpoint, map_location="cpu", weights_only=True)
                if payload["optimizer_step"] != step or payload["r26_signature"] != signature:
                    raise ValueError("Checkpoint step or input identity differs")
                optimizer = payload["optimizer_state_dict"]
                ids = [i for group in optimizer["param_groups"] for i in group["params"]]
                names = init["trainable_parameter_names"]
                if len(ids) != len(names) or len(set(ids)) != len(ids):
                    raise ValueError("Optimizer does not cover all trainable parameters")
                for group in optimizer["param_groups"]:
                    if (group["lr"],group["weight_decay"]) != (protocol["optimizer"]["lr"],protocol["optimizer"]["weight_decay"]):
                        raise ValueError("Actual optimizer hyperparameters differ")
                if step == 10536:
                    check(receipt["final_checkpoint"])
                    for parameter_id, name in zip(ids,names):
                        observed = optimizer["state"].get(parameter_id, {}).get("step",0)
                        if int(observed) != actual["gradient_present_steps"].get(name,0):
                            raise ValueError("Actual optimizer update counts differ from gradients")
                        delta = float((payload["state_dict"][name]-start["state_dict"][name]).norm())
                        if not math.isclose(delta,receipt["parameter_delta_norms"][name],rel_tol=1e-5,abs_tol=1e-7):
                            raise ValueError("Reported parameter delta differs from actual tensors")
                checkpoint_records.append({"path":str(checkpoint),"bytes":checkpoint.stat().st_size,"sha256":digest(checkpoint)})
                if step:
                    del payload
            del start
            rows.append({"arm":arm,"seed":seed,"actual":actual,"checkpoints":checkpoint_records,
                         "training_receipt":{"path":str(path),"sha256":digest(path)}})
            print(json.dumps({"audited_teacher":arm,"seed":seed}),flush=True)
    result = {"execution_status":"ran" if len(rows)==6 else "in_progress",
              "scientific_validity":"valid" if len(rows)==6 else "partial",
              "jobs":rows,"pending":pending,"verified_artifact_hashes":verified,
              "code":{"path":str(Path(__file__)),"sha256":digest(Path(__file__))},
              "scope":"Full recorded update/list/gradient counts, common-parent tensors, all checkpoint identities, final optimizer update counts and actual parameter deltas. Does not independently replay every backward operation."}
    (OUT / "acceptance/REFINEMENT_TRAINING_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
    return {"jobs":len(rows),"pending":len(pending),"execution_status":result["execution_status"]}


if __name__ == "__main__":
    print(json.dumps(run()))
