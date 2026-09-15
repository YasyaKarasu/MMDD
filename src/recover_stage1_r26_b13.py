"""Recover pruned B13 only if replayed checkpoint bytes match archived SHA256."""
from __future__ import annotations

import json
import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.scoring import score_target_batch
from mmdd_stage1.training import checkpoint, _student_path_losses, _target_teacher_scores
from prepare_stage1_r26 import ROOT, OUT, file_record
from run_stage1_r13 import _merge_witness_metadata, _optimizer
from run_stage1_r21 import paths
from run_stage1_r25 import _json


def recover() -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    device = torch.device("cuda:1")
    source = ROOT / "work/stage1_optimization_r12_20260908/taskC_training/c2_path_only_seed13/student_path.steps/step_000000.pt"
    directory = OUT / "recovered/B13"
    directory.mkdir(parents=True, exist_ok=True)
    model = load_student(source, device)
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    archive = ROOT / "work/stage1_optimization_r13_20260909/taskD_witness_supervision/p_s_target_only/checkpoints"
    records = []

    def save(step: int) -> bool:
        target = directory / f"step_{step:06d}.pt"
        payload = checkpoint(model, "student-path", aggregator)
        # R13 predates residual projection adapters. These default-only config
        # additions change pickle bytes although all P/R tensors are identical.
        for key in ("projection_adapter", "projection_hidden_dim", "projection_scales"):
            payload["config"].pop(key, None)
        for key in ("projection_residual_inputs", "projection_residual_outputs"):
            payload["state_dict"]._metadata.pop(key, None)
        torch.save(payload, target)
        expected = json.loads((archive / f"step_{step:06d}.json").read_text())["checkpoint_sha256"]
        record = {"step": step, **file_record(target), "expected_sha256": expected}
        record["exact_archived_file_match"] = record["sha256"] == expected
        records.append(record)
        _json(directory / "RECOVERY_AUDIT.json", {"source": file_record(source), "checkpoints": records,
             "valid_historical_B13": step == 178 and all(r["exact_archived_file_match"] for r in records)})
        print(json.dumps(record), flush=True)
        return record["exact_archived_file_match"]

    if not save(0):
        return {"status": "blocked_start_identity", "records": records}
    examples = _merge_witness_metadata(ROOT)
    order_path = archive.parent.parent / "schedule_order.json"
    order = json.loads(order_path.read_text())["indices"]
    ordered = [examples[i] for i in order]
    optimizer = _optimizer(model)
    store = FeatureStore.from_path(paths(ROOT)["features"], cache_size=120000)
    for step, start in enumerate(range(0, len(ordered), 64), 1):
        batch = ordered[start:start+64]
        model.train()
        scores = score_target_batch(model, batch, store, device, aggregator)
        teacher = _target_teacher_scores(batch, device)
        terms = _student_path_losses(model, scores, teacher, None, temperature=1.0,
            distillation_weight=.3, anchor_weight=.1, anchor_weight_evidence=.1,
            distillation_rows=None, positive_loss_mode="sum_probability")
        optimizer.zero_grad()
        terms["loss"].backward()
        optimizer.step()
        if step in (45, 89, 178) and not save(step):
            return {"status": "blocked_replay_identity", "records": records}
    return {"status": "exact_historical_checkpoint_recovered", "records": records}


if __name__ == "__main__":
    print(json.dumps(recover()))
