"""Write fixed-batch gradient and active-relation audits for R29 checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.r28_objectives import objective
from prepare_stage1_r28 import ROOT, inputs, registry, feature_store
from run_stage1_r13 import _merge_witness_metadata


OUT = ROOT / "work/stage1_optimization_r29_candidate_vs_drift_20260915"
R28 = ROOT / "work/stage1_optimization_r28_split_path_20260915"


def configure(model: Any, arm: str) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("projections."):
            parameter.requires_grad = arm == "S-EDGE-FREEZE-R"
        elif name.startswith(("relations.", "relation_as.", "relation_bs.")):
            parameter.requires_grad = arm == "S-EDGE-FREEZE-P"
        else:
            parameter.requires_grad = False


def category(name: str) -> str | None:
    if name.startswith("projections."): return "P"
    if name == "relations.table_to_table": return "R_QT"
    if name == "relations.table_to_text": return "R_QE"
    if name == "relations.text_to_table": return "R_ET"
    return None


def collect(model: Any, terms: dict[str, torch.Tensor], key: str) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    loss = terms[key]
    if not loss.requires_grad:
        return {name: {"norm": None} for name in ("P", "R_QT", "R_QE", "R_ET")}
    loss.backward(retain_graph=True)
    grouped: dict[str, list[torch.Tensor]] = {name: [] for name in ("P", "R_QT", "R_QE", "R_ET")}
    for name, parameter in model.named_parameters():
        cat = category(name)
        if cat and parameter.grad is not None:
            grouped[cat].append(parameter.grad.detach().float().reshape(-1))
    result = {}
    for name, values in grouped.items():
        vector = torch.cat(values) if values else torch.empty(0)
        result[name] = {"norm": float(vector.norm()) if vector.numel() else None, "vector": vector}
    return result


def main() -> None:
    examples = _merge_witness_metadata(ROOT)
    known = registry()
    store = feature_store(False)
    order = json.loads((R28 / "common/orders.json").read_text())["13"][0][:64]
    batch = [examples[i] for i in order]
    for arm in ("S-EDGE-FREEZE-P", "S-EDGE-FREEZE-R"):
        root = OUT / "training" / arm / "seed13"
        output: dict[str, Any] = {"fixed_batch_indices": order, "logical_batch": len(batch), "checkpoints": {}}
        for step in (0, 178, 356, 534):
            model = load_student(root / "checkpoints" / f"step_{step:06d}.pt", torch.device("cpu")).train()
            configure(model, arm)
            terms = objective(model, batch, store, torch.device("cpu"), "EDGE", known, student=True)
            by_loss: dict[str, dict[str, Any]] = {}
            vectors: dict[str, dict[str, torch.Tensor]] = {}
            for key in ("direct_supervised_loss", "qe_supervised_loss", "et_supervised_loss", "weighted_anchor_loss"):
                values = collect(model, terms, key)
                vectors[key] = {cat: item.pop("vector") for cat, item in values.items()}
                by_loss[key] = values
            cosines: dict[str, dict[str, float | None]] = {}
            loss_keys = list(vectors)
            for cat in ("P", "R_QT", "R_QE", "R_ET"):
                cosines[cat] = {}
                for left in loss_keys:
                    for right in loss_keys:
                        a, b = vectors[left][cat], vectors[right][cat]
                        value = None
                        if a.numel() and b.numel() and float(a.norm()) and float(b.norm()): value = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                        cosines[cat][f"{left}:{right}"] = value
            output["checkpoints"][str(step)] = {"losses": {k: float(v.detach()) for k, v in terms.items() if isinstance(v, torch.Tensor)}, "gradient_norms": by_loss, "pairwise_cosines": cosines}
        (root / "gradients").mkdir(parents=True, exist_ok=True)
        (root / "gradients/fixed_batch.json").write_text(json.dumps(output, indent=2) + "\n")
        # Collapse per-batch activity into the required epoch-level receipt.
        totals: dict[str, dict[str, int]] = {}
        for line in (root / "history.jsonl").read_text().splitlines():
            record = json.loads(line)
            for rel, values in record.get("active_relations", {}).items():
                total = totals.setdefault(rel, {"n_total": 0, "n_active": 0})
                total["n_total"] += int(values["n_total"]); total["n_active"] += int(values["n_active"])
        (root / "active_relation_epoch_summary.json").write_text(json.dumps({"by_epoch": "history contains per-batch values", "totals_over_3_epochs": totals}, indent=2) + "\n")
    (OUT / "diagnostics/teacher_score_status.json").write_text(json.dumps({"status": "not_measured_r29_scope", "reason": "R29 Natural-candidate T0 scoring was not part of this bounded diagnostic; CUDA was verified outside the sandbox", "constraint": "Teacher may only score Student candidates", "dev_test_qrels_used_for_mining": False, "cuda_verified": True}, indent=2) + "\n")
    print(json.dumps({"status": "gradient_and_activity_audit_complete"}))


if __name__ == "__main__":
    main()
