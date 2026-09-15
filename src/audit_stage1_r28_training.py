"""Verify saved R28 tensors, optimizer groups, batches, and loss decomposition."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from prepare_stage1_r27 import record, rows, sha, stable_sha, write_json
from prepare_stage1_r28 import OUT, FAMILIES, inputs


def tensor_sha(state: dict, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_job(arm: str, seed: int) -> dict:
    kind = "student" if arm.startswith("S-") else "teacher"
    prepared = json.loads((OUT / "R28_PREPARED_MANIFEST.json").read_text())
    orders = json.loads((OUT / "common/orders.json").read_text())[str(seed)]
    assert sha(OUT / "common/orders.json") == prepared["orders"]["sha256"]
    directory = OUT / kind / arm / f"seed{seed}"
    execution = json.loads((directory / "EXECUTION.json").read_text())
    signature = execution["signature"]
    assert signature == json.loads((directory / "RUN_IDENTITY.json").read_text())
    assert execution["status"] == "completed"
    assert execution["updates"] == prepared[kind]["total_updates"]
    assert (signature["arm"], signature["seed"]) == (arm, seed)
    assert signature["parent"]["sha256"] == inputs()[f"{kind}_parent"]["sha256"]
    assert signature["graph"]["sha256"] == inputs()["target_path_graph"]["sha256"]
    assert signature["order_hashes"] == [stable_sha(o) for o in orders]
    names = prepared["trainable"][kind]["names"]
    assert signature["trainable_names"] == names
    assert signature["trainable_names_hash"] == stable_sha(names)
    assert signature["objective_family"] == FAMILIES[arm]
    assert signature["split"] == (FAMILIES[arm] != "EDGE")
    assert signature["aggregator"] == (None if FAMILIES[arm] == "EDGE" else FAMILIES[arm])
    assert signature["optimizer_initial_state"] == "fresh"
    batch = prepared[kind]["batch"]
    assert signature["logical_batch"] == batch
    for rec in signature["code"].values():
        assert sha(Path(rec["path"])) == rec["sha256"]
    groups = signature["optimizer"]
    assert sorted(n for g in groups for n in g["parameters"]) == sorted(names)
    for group in groups:
        assert group["weight_decay"] == .01
        for name in group["parameters"]:
            expected_lr = (1e-6 if name.startswith("projections.") else 1e-5) if kind == "student" else prepared[kind]["lr"]
            assert group["lr"] == expected_lr
    history = list(rows(directory / "history.jsonl"))
    expected_batches = [(e, start, order[start:start+batch]) for e, order in enumerate(orders, 1)
                        for start in range(0, len(order), batch)]
    assert len(history) == len(expected_batches) == execution["updates"]
    max_loss_error = 0.
    for step, (trace, (epoch, start, indices)) in enumerate(zip(history, expected_batches), 1):
        assert (trace["step"], trace["epoch"], trace["batch_start"]) == (step, epoch, start)
        assert trace["source_indices"] == indices and trace["order_hash"] == stable_sha(indices)
        terms = trace["losses"]
        assert all(np.isfinite(v) for v in terms.values())
        expected = terms["direct_supervised_loss"] + terms["weighted_anchor_loss"]
        if FAMILIES[arm] == "EDGE":
            assert "evidence_supervised_loss" not in terms
            expected += .5 * (terms["qe_supervised_loss"] + terms["et_supervised_loss"])
        else:
            expected += terms["evidence_supervised_loss"]
        if kind == "teacher":
            assert terms["weighted_anchor_loss"] == 0
        max_loss_error = max(max_loss_error, abs(expected - terms["loss"]))
    assert max_loss_error < 1e-5
    parent = torch.load(signature["parent"]["path"], map_location="cpu", weights_only=True)
    parent_hash = tensor_sha(parent["state_dict"], names)
    checks = []
    assert [c["epoch"] for c in execution["checkpoints"]] == [0, .5, 1, 2, 3, 5]
    for rec in execution["checkpoints"]:
        path = Path(rec["path"])
        assert sha(path) == rec["sha256"]
        assert json.loads(path.with_suffix(".json").read_text()) == rec
        payload = torch.load(path, map_location="cpu", weights_only=True)
        step = math.ceil(prepared[kind]["updates_per_epoch"] * rec["epoch"])
        assert payload["optimizer_step"] == rec["updates"] == step
        assert payload["epoch"] == rec["epoch"]
        assert json.loads(json.dumps(payload["r28_signature"])) == signature
        actual_hash = tensor_sha(payload["state_dict"], names)
        assert actual_hash == rec["parameter_sha256"]
        optimizer = payload["optimizer_state_dict"]
        assert len(optimizer["param_groups"]) == len(groups)
        for saved, group in zip(optimizer["param_groups"], groups):
            assert len(saved["params"]) == len(group["parameters"])
            for key, value in group.items():
                if key != "parameters":
                    assert json.loads(json.dumps(saved[key])) == value
        if step == 0:
            assert actual_hash == parent_hash and not optimizer["state"]
        else:
            assert actual_hash != parent_hash and optimizer["state"]
            assert all(0 < float(s["step"]) <= step for s in optimizer["state"].values())
        if kind == "student":
            state = payload["state_dict"]
            assert torch.equal(state["initial_projection_weights"], parent["state_dict"]["initial_projection_weights"])
            expected_anchor = torch.stack([parent["state_dict"][f"projections.{modality}.weight"] for modality in ("table", "text", "image")])
            assert torch.equal(state["stage_initial_projection_weights"], expected_anchor)
        checks.append({"epoch": rec["epoch"], "updates": step, "actual_parameter_sha256": actual_hash,
                       "checkpoint": rec})
        del payload
    return {"kind": kind, "arm": arm, "seed": seed, "status": "pass", "updates": len(history),
            "max_loss_reconstruction_error": max_loss_error, "parent_parameter_sha256": parent_hash,
            "exact_batches_and_order_verified": True, "checkpoint_tensors": checks,
            "execution": record(directory / "EXECUTION.json"), "history": record(directory / "history.jsonl")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("student", "teacher", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = []
    for arm in FAMILIES:
        kind = "student" if arm.startswith("S-") else "teacher"
        if args.kind not in (kind, "all"):
            continue
        for seed in (13, 29):
            result = audit_job(arm, seed)
            results.append(result)
            print(json.dumps({"arm": arm, "seed": seed, "status": "pass"}), flush=True)
    write_json(OUT / f"TRAINING_TENSOR_AUDIT_{args.kind}.json", {"status": "pass", "jobs": results})
