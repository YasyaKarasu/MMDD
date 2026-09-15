"""R27 H: two fresh historical stages, with frozen inputs and full state traces."""
from __future__ import annotations

import argparse
import copy
import difflib
import gzip
import json
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.scoring import score_edge_batch, score_target_batch
from mmdd_stage1.training import _student_edge_losses, _student_path_losses, _target_teacher_scores, checkpoint, student_gradient_norms, student_projection_references
from prepare_stage1_r27 import ROOT, OUT, R12, R13, R26, read_json, write_json, record, rows, sha, stable_sha
from run_stage1_r12_task_c import _initialize_student, _optimizer, _ranking_scores, _schedule_batches, _score_payload, _teacher_list_scores
from run_stage1_r13 import _merge_witness_metadata


def tensor_sha(tensor: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_fingerprints(model) -> dict:
    return {name: tensor_sha(tensor) for name, tensor in model.state_dict().items()}


def compare_state(current: dict, reference: dict) -> list[dict]:
    records = []
    for key in sorted(set(current) | set(reference)):
        if key not in current or key not in reference:
            records.append({"key": key, "status": "missing_key", "current": key in current, "reference": key in reference})
            continue
        a, b = current[key].detach().cpu(), reference[key].detach().cpu()
        same_shape = a.shape == b.shape
        diff = a.double() - b.double() if same_shape else None
        records.append({"key": key, "shape": list(a.shape), "reference_shape": list(b.shape), "dtype": str(a.dtype), "reference_dtype": str(b.dtype), "torch_equal": same_shape and a.dtype == b.dtype and torch.equal(a, b), "max_abs": float(diff.abs().max()) if diff is not None else None, "relative_l2": float(diff.norm() / b.double().norm().clamp_min(1e-30)) if diff is not None else None, "allclose": same_shape and torch.allclose(a, b, atol=1e-6, rtol=1e-5)})
    return records


def save_node(model, aggregator, stage: str, step: int, directory: Path) -> dict:
    node = directory / "checkpoints" / f"step_{step:06d}.pt"
    node.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint(model, "student-edge" if stage == "C1" else "student-path", aggregator)
    torch.save(payload, node)
    # Preserve original output; export only historical default metadata separately.
    export = copy.deepcopy(payload)
    removed = {key: export["config"].pop(key) for key in ("projection_adapter", "projection_hidden_dim", "projection_scales")}
    if stage == "C1":
        removed["projection_mode"] = export["config"].pop("projection_mode")
    for key in ("projection_residual_inputs", "projection_residual_outputs"):
        export["state_dict"]._metadata.pop(key, None)
    export_path = directory / "historical_metadata_export" / node.name
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, export_path)
    if stage == "C1":
        ref = R12 / "taskC_training/c_candidates_seed13/checkpoints" / node.name
        expected = read_json(R12 / "taskC_training/c_candidates_seed13/manifest.json")["checkpoints"][str(step)]["checkpoint_sha256"]
    else:
        ref = R26 / "recovered/B13" / node.name
        expected = read_json(R13 / "taskD_witness_supervision/p_s_target_only/checkpoints" / node.with_suffix(".json").name)["checkpoint_sha256"]
    comparison = []
    if ref.exists():
        old = torch.load(ref, map_location="cpu", weights_only=True)
        comparison = compare_state(payload["state_dict"], old["state_dict"])
    result = {"stage": stage, "step": step, "checkpoint": record(node), "reference": record(ref), "expected_archived_sha256": expected, "metadata_export": record(export_path), "removed_default_metadata": removed, "original_file_exact": sha(node) == expected, "historical_export_file_exact": sha(export_path) == expected, "state_comparison": comparison, "state_fingerprints": state_fingerprints(model), "projection_references": student_projection_references(model)}
    result["state_parity_level"] = ("tensor_exact" if result["historical_export_file_exact"] or (comparison and all(r.get("torch_equal") for r in comparison)) else "tensor_divergence" if comparison else "state_parity_unverifiable")
    write_json(node.with_suffix(".json"), result)
    parity = OUT / "historical_replay/parity"
    parity.mkdir(parents=True, exist_ok=True)
    with gzip.open(parity / "state_comparison.jsonl.gz", "at") as handle:
        handle.write(json.dumps(result) + "\n")
    return result


def run(stage: str, device_name: str) -> dict:
    hist = OUT / "historical_replay"
    gates = read_json(hist / "H_EXECUTION_LEDGER.json")
    assert gates["GH0"] == "pass_with_historical_runtime_unknown" and gates["GH1"] == "pass"
    frozen = read_json(hist / "H_RESOLVED_INPUTS.json")
    for rec in frozen:
        if rec["required_for_training"]:
            assert sha(Path(rec["path"])) == rec["sha256"], rec["logical_id"]
    assert (hist / "H_RECIPE_DIFF.json").exists()
    directory = hist / stage / "seed13"
    if (directory / "EXECUTION.json").exists():
        raise FileExistsError("Existing stage must be inspected; never start another training job implicitly")
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    store = FeatureStore.from_path(ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=120000)
    if stage == "C1":
        model = _initialize_student(ROOT, device)
        scores, teacher_manifest = _score_payload(R12)
        batches = list(_schedule_batches(R12 / "taskC_training/candidates_seed13_steps356/candidates.jsonl.gz", scores, teacher_manifest["teacher_checkpoint_sha256"]))
        aggregator = None
        nodes = (0, 45, 89, 178, 267, 356)
        parent = record(ROOT / "work/stage1_optimization_r11_20260908/taskA_protocol/baselines/pca_init.pt")
    else:
        parent_path = hist / "C1/seed13/checkpoints/step_000356.pt"
        c1_execution = read_json(hist / "C1/seed13/EXECUTION.json")
        assert c1_execution["status"] == "completed" and c1_execution["updates"] == 356
        assert sha(parent_path) == c1_execution["last_checkpoint"]["sha256"]
        model = load_student(parent_path, device)
        before = state_fingerprints(model)
        model.reset_projection_anchors()
        write_json(directory / "ANCHOR_BOUNDARY.json", {"before": before, "after": state_fingerprints(model), "parent": record(parent_path), "fresh_parent_run": str(hist / "C1/seed13")})
        parent = record(parent_path)
        examples = _merge_witness_metadata(ROOT)
        order = read_json(R13 / "taskD_witness_supervision/schedule_order.json")["indices"]
        ordered = [examples[i] for i in order]
        batches = [(step, ordered[start:start+64]) for step, start in enumerate(range(0, len(ordered), 64), 1)]
        assert len(ordered) == 11390 and len(batches[-1][1]) == 62
        aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
        nodes = (0, 45, 89, 178)
    assert len(batches) == nodes[-1]
    optimizer = _optimizer(model)
    assert not optimizer.state
    names = {id(p): name for name, p in model.named_parameters()}
    runtime = {"stage": stage, "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(device), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "cpu_threads": 2, "optimizer_groups": [{**{k: v for k, v in group.items() if k != "params"}, "parameters": [names[id(p)] for p in group["params"]]} for group in optimizer.param_groups], "fresh_optimizer_empty": True, "seed": 13, "historical_runtime": "unknown; do not claim E0 bitwise recipe identity solely from current runtime"}
    write_json(directory / "RUNTIME_LOCK.json", runtime)
    # Actual imported local dependency closure, frozen before the first update.
    closure = [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(ROOT / "src"))]
    write_json(directory / "DEPENDENCY_CLOSURE.json", [record(p) for p in sorted(set(closure))])
    source = ROOT / "src" / ("run_stage1_r12_task_c.py" if stage == "C1" else "run_stage1_r13.py")
    diff = "".join(difflib.unified_diff(source.read_text().splitlines(True), Path(__file__).read_text().splitlines(True), fromfile=str(source), tofile=str(Path(__file__))))
    (directory / "H_SOURCE_DIFF.patch").write_text(diff)
    execution = {"stage": stage, "planned": True, "implemented": True, "executed": True, "evaluated": False, "status": "running", "pid": os.getpid(), "updates": 0, "parent": parent, "command": [sys.executable, *sys.argv]}
    write_json(directory / "EXECUTION.json", execution)
    start_node = save_node(model, aggregator, stage, 0, directory)
    if stage == "C1" and not start_node["historical_export_file_exact"]:
        raise ValueError("Fresh C1 initialization differs from archived step0; inspect before updates")
    write_json(directory / "GH2.json", {"status": "pass_with_parity_reported", "initial_state": start_node["state_parity_level"], "optimizer": runtime, "supervision": "sum_probability", "KD": 0.3, "anchor": 0.1, "batch_size": 64, "aggregation": "all_original_paths" if stage == "C2" else None})
    traces = directory / "step_traces.jsonl.gz"
    consumption = directory / ("consumed_batches.jsonl.gz" if stage == "C1" else "consumed_order.jsonl.gz")
    started = time.monotonic()
    history = []
    historical_manifest = read_json(R12 / "taskC_training/c_candidates_seed13/manifest.json" if stage == "C1" else R13 / "taskD_witness_supervision/p_s_target_only/manifest.json")
    with gzip.open(traces, "wt") as trace_file, gzip.open(consumption, "wt") as consumed_file:
        for step, batch in batches:
            assert step == execution["updates"] + 1
            model.train()
            if stage == "C1":
                raw = score_edge_batch(model, batch, store, device, student_score_space="raw_logit")
                teacher = _teacher_list_scores(batch, raw, device)
                terms = _student_edge_losses(model, batch, raw, teacher, _ranking_scores(raw), None, ranking_weight=1., temperature=1., distillation_weight=.3, edge_bce_weight=0., anchor_weight=.1, anchor_weight_evidence=.1, positive_loss_mode="sum_probability")
                terms["loss"] = terms["loss"] + 0.1 * raw.logits.new_zeros(())
                channels = {"edge": raw}
            else:
                raw = score_target_batch(model, batch, store, device, aggregator)
                teacher = _target_teacher_scores(batch, device)
                terms = _student_path_losses(model, raw, teacher, None, temperature=1., distillation_weight=.3, anchor_weight=.1, anchor_weight_evidence=.1, distillation_rows=None, positive_loss_mode="sum_probability")
                channels = {"direct": raw.direct, "evidence": raw.evidence}
            if not torch.isfinite(terms["loss"]):
                raise ValueError(f"Nonfinite loss at {stage}/{step}")
            optimizer.zero_grad()
            terms["loss"].backward()
            gradient = student_gradient_norms(model)
            optimizer.step()
            consumed = {"step": step, "examples": [asdict(e) for e in batch]}
            consumed_file.write(json.dumps(consumed) + "\n")
            trace = {"step": step, "batch_sha256": stable_sha(consumed), "batch_size": len(batch), "losses": {k: float(v.detach()) for k, v in terms.items()}, "gradient_norms": gradient, "state_fingerprints_after_update": state_fingerprints(model), "relations": dict(Counter(f"{e.source_type}_to_{e.destination_type}" for e in batch)) if stage == "C1" else {"native_direct_evidence_lists": len(batch)}, "score_mask_fingerprints": {name: {"raw_logits": tensor_sha(s.logits), "candidate_mask": tensor_sha(s.candidate_mask), "positive_mask": tensor_sha(s.positive_mask)} for name, s in channels.items()}, "elapsed_seconds": time.monotonic()-started}
            ref_loss = historical_manifest["history"][step-1]
            trace["historical_loss_comparison"] = {k: {"reference": ref_loss[k], "current": trace["losses"][k], "abs_difference": abs(trace["losses"][k]-ref_loss[k])} for k in ("loss", "supervised_loss", "distillation_loss")}
            if step == 1 or step in nodes:
                diagnostic = {name: {"raw_logits": s.logits.detach().cpu(), "candidate_mask": s.candidate_mask.detach().cpu(), "positive_mask": s.positive_mask.detach().cpu()} for name,s in channels.items()}
                torch.save(diagnostic, directory / f"batch_scores_before_update_{step:06d}.pt")
            trace_file.write(json.dumps(trace) + "\n")
            history.append({"step": step, "losses": trace["losses"]})
            execution["updates"] = step
            if step in nodes:
                node = save_node(model, aggregator, stage, step, directory)
                execution["last_checkpoint"] = node["checkpoint"]
                write_json(directory / "EXECUTION.json", execution)
            if step % 10 == 0 or step in nodes:
                trace_file.flush()
                consumed_file.flush()
                print(json.dumps({"stage": stage, "step": step, "loss": trace["losses"]["loss"], "seconds": trace["elapsed_seconds"]}), flush=True)
    execution.update(status="completed", elapsed_seconds=time.monotonic()-started, traces=record(traces), consumption=record(consumption))
    write_json(directory / "EXECUTION.json", execution)
    write_json(directory / "loss_history.json", history)
    return execution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("C1", "C2"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.stage, args.device)))
