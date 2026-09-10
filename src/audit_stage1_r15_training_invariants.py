#!/usr/bin/env python
"""Audit the frozen R15 training inputs, initialization, optimizer and updates."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_checkpoint, load_student
from mmdd_stage1.scoring import target_positive_mask
from run_stage1_r13 import _paths, freeze_path_schedule
from run_stage1_r14 import _optimizer, _pair_counts, _schedule, _sha256_text
from run_stage1_r15 import ARM_SPECS, arm_directory, optimizer_audit, train_arm


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_check(record: dict[str, Any]) -> dict[str, Any]:
    actual = checkpoint_fingerprint(Path(record["path"]))
    return {"path": record["path"], "frozen_sha256": record["sha256"], "actual_sha256": actual,
            "matches": actual == record["sha256"]}


def schedule_audit(root: Path) -> tuple[dict[str, Any], list[Any]]:
    frozen = freeze_path_schedule(root)
    batches, order = _schedule(root, 13)
    examples = [example for batch in batches for example in batch]
    canonical_batches = []
    for step, batch in enumerate(batches, 1):
        width = max(len(example.candidates) for example in batch)
        direct_positive = target_positive_mask(batch, width, torch.device("cpu"), channel="direct")
        evidence_available = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, example in enumerate(batch):
            evidence_available[row, :len(example.candidates)] = torch.tensor([bool(candidate.evidence_ids) for candidate in example.candidates])
        evidence_positive = target_positive_mask(batch, width, torch.device("cpu"), channel="evidence") & evidence_available
        canonical_batches.append({
            "step": step,
            "queries": [{"query_id": example.query_id, "positive_target_ids": list(example.positive_target_ids),
                         "candidates": [{"target_id": candidate.target_id, "evidence_ids": list(candidate.evidence_ids)} for candidate in example.candidates]}
                        for example in batch],
            "direct_positive_mask": direct_positive.tolist(),
            "evidence_available_mask": evidence_available.tolist(),
            "evidence_positive_mask": evidence_positive.tolist(),
        })
    teacher_logits = [{"query_id": example.query_id, "direct": example.teacher_direct_logits,
                       "evidence": example.teacher_evidence_logits,
                       "teacher_checkpoint_sha256": example.teacher_checkpoint_sha256}
                      for example in examples]
    mask_rows = [{key: row[key] for key in ("step", "direct_positive_mask", "evidence_available_mask", "evidence_positive_mask")} for row in canonical_batches]
    output = root / "work/stage1_optimization_r15_20260909/stageI_interaction/statistics"
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "training_schedule_replay.json"
    write_json(schedule_path, {"canonicalization": "sorted-key compact UTF-8 JSON hashes; current shared loader replay on frozen files",
                               "batches": canonical_batches})
    return {
        "queries": len(examples), "batches": len(batches), "batch_sizes": [len(batch) for batch in batches],
        "order_query_ids_sha256": _sha256_text([example.query_id for example in examples]),
        "order_indices_canonical_sha256": canonical_hash(order),
        "ordered_candidates_and_masks_canonical_sha256": canonical_hash(canonical_batches),
        "positive_and_availability_masks_canonical_sha256": canonical_hash(mask_rows),
        "embedded_teacher_logits_canonical_sha256": canonical_hash(teacher_logits),
        "every_query_has_both_cached_teacher_channels": all(example.teacher_direct_logits is not None and example.teacher_evidence_logits is not None for example in examples),
        "processed": _pair_counts(examples),
        "schedule_replay": {"path": str(schedule_path), "sha256": checkpoint_fingerprint(schedule_path)},
        "actual_inputs": {
            "order": file_check({"path": frozen["order"], "sha256": frozen["order_sha256"]}),
            "path_hard_with_embedded_teacher_logits": file_check(frozen["path_hard"]),
            "witness_source": file_check(frozen["witness_source"]),
            "path_hard_metadata": file_check({"path": str(_paths(root)["path_hard_metadata"]), "sha256": frozen["path_hard"]["metadata_sha256"]}),
        },
        "historical_record_limit": "R15 manifests did not store order/mask hashes during execution; current replay matches historical R14 query-order hash and unchanged frozen inputs/shared loader. Replayed masks are not a newly discovered historical trace.",
    }, examples


def audit(root: Path) -> dict[str, Any]:
    torch.set_num_threads(2)
    output = root / "work/stage1_optimization_r15_20260909"
    frozen = json.loads((output / "PLAN_FROZEN.json").read_text())
    recovery_path = output / "RECOVERED_TRAINING_SOURCE.json"
    recovery = json.loads(recovery_path.read_text())
    recovered_source = recovery["recovered_source"]
    if not file_check(recovered_source)["matches"] or not recovery["both_training_manifest_hashes_match"]:
        raise ValueError("Recovered R15 training entrypoint fingerprint differs")
    r14_recovery = recovery["r14_entrypoint"]
    recovered_r14 = r14_recovery["recovered_source"]
    if not file_check(recovered_r14)["matches"] or not r14_recovery["all_seven_training_manifest_entrypoints_match"]:
        raise ValueError("Recovered R14 training entrypoint fingerprint differs")
    schedule, examples = schedule_audit(root)
    inputs = {key: file_check(frozen["r14_inputs"][key]) for key in ("s0", "schedule", "teacher_scores")}
    s0 = load_checkpoint(Path(frozen["r14_inputs"]["s0"]["path"]))
    arms = {}
    training_source = root / "src/run_stage1_r15.py"
    shared_source = root / "src/run_stage1_r14.py"
    for arm, specification in ARM_SPECS.items():
        directory = arm_directory(root, arm)
        manifest = json.loads((directory / "manifest.json").read_text())
        r14_directory = root / "work/stage1_optimization_r14_20260909/stage1_M_projection_capacity" / specification["r14_arm_id"]
        r14_manifest = json.loads((r14_directory / "manifest.json").read_text())
        initial_path = directory / "checkpoints/step_000000.pt"
        final_path = directory / "checkpoints/step_000178.pt"
        reference_path = Path(frozen["r14_checkpoints"][arm]["0"]["path"])
        initial = load_checkpoint(initial_path)
        reference = load_checkpoint(reference_path)
        final = load_checkpoint(final_path)
        state = initial["state_dict"]
        base_parameter_names = [name for name in state if name.startswith(("projections.", "relations."))]
        model = load_student(initial_path, torch.device("cpu"))
        optimizer = _optimizer(model)
        rebuilt = optimizer_audit(optimizer, model)
        checkpoints = {str(step): json.loads((directory / "checkpoints" / f"step_{step:06d}.json").read_text()) for step in (0, 45, 89, 178)}
        actual_names = [name for group in checkpoints["0"]["optimizer"]["groups"] for name in group["parameter_names"]]
        trainable = {name: {"requires_grad": parameter.requires_grad, "elements": parameter.numel(), "dtype": str(parameter.dtype)} for name, parameter in model.named_parameters()}
        grouped_names = set(actual_names)
        needed_names = {name for name in trainable if name.startswith(("projections.", "projection_residual_inputs.", "projection_residual_outputs.", "relations."))}
        state_groups = {step: checkpoint["optimizer"]["groups"] for step, checkpoint in checkpoints.items()}
        history = manifest["history"]
        initialized_same = all(torch.equal(state[name], reference["state_dict"][name]) for name in state)
        base_same = all(torch.equal(state[name], s0["state_dict"][name]) for name in base_parameter_names)
        initial_b_zero = {kind: int(torch.count_nonzero(state[f"projection_residual_outputs.{kind}.weight"])) == 0 for kind in ("table", "text", "image")}
        initial_a_nonzero = {kind: int(torch.count_nonzero(state[f"projection_residual_inputs.{kind}.weight"])) > 0 for kind in ("table", "text", "image")}
        checkpoint_file_checks = {step: file_check({"path": value["checkpoint"], "sha256": value["checkpoint_sha256"]}) for step, value in checkpoints.items()}
        fresh_state = all(group["state_entries"] == 0 and group["state_step_min"] is None and group["state_step_max"] is None for group in state_groups["0"])
        state_steps_match = all(all(group["state_step_min"] == int(step) and group["state_step_max"] == int(step) for group in groups) for step, groups in state_groups.items() if step != "0")
        scales = {step: load_checkpoint(Path(checkpoints[step]["checkpoint"]))["config"]["projection_scales"] for step in checkpoints}
        loss_max_error = max(abs(row["loss"] - row["direct_loss"] - row["weighted_anchor_loss"]) for row in history)
        checks = {
            "initial_checkpoint_reference_hash_matches_frozen": file_check(manifest["initial_checkpoint"])["matches"] and manifest["initial_checkpoint"] == frozen["r14_checkpoints"][arm]["0"],
            "saved_step0_all_state_tensors_equal_R14": initialized_same,
            "saved_step0_config_equal_R14": initial["config"] == reference["config"],
            "step0_base_P_R_equal_S0": base_same,
            "step0_B_all_zero": all(initial_b_zero.values()), "step0_A_all_nonzero": all(initial_a_nonzero.values()),
            "c_tau_preserved_all_checkpoints": all(value == reference["config"]["projection_scales"] for value in scales.values()),
            "actual_optimizer_group_membership_matches_shared_builder": all(a["parameter_names"] == b["parameter_names"] and a["lr"] == b["lr"] and a["weight_decay"] == b["weight_decay"] for a, b in zip(state_groups["0"], rebuilt["groups"])),
            "P_A_B_R_all_trainable_and_in_optimizer_once": needed_names == grouped_names and len(actual_names) == len(grouped_names) and all(trainable[name]["requires_grad"] for name in needed_names),
            "fresh_Adam_state": fresh_state, "recorded_Adam_steps_match_checkpoints": state_steps_match,
            "checkpoint_file_hashes_match_recorded": all(value["matches"] for value in checkpoint_file_checks.values()),
            "query_order_matches_R14_historical_hash": schedule["order_query_ids_sha256"] == r14_manifest["schedule"]["order_query_ids_sha256"],
            "queries_11390_updates178": len(examples) == manifest["schedule"]["queries"] == 11390 and manifest["optimizer_updates"] == manifest["schedule"]["batches"] == len(history) == 178,
            "history_steps_are_1_to_178": [row["optimizer_updates"] for row in history] == list(range(1, 179)),
            "all_weighted_evidence_losses_zero": all(row["weighted_evidence_loss"] == 0 for row in history) and manifest["evidence_loss_weight"] == 0,
            "unweighted_evidence_computed_finite": all(math.isfinite(row["evidence_loss_unweighted"]) and row["evidence_loss_unweighted"] > 0 for row in history) and manifest["full_evidence_forward_computed"],
            "loss_matches_D_plus_anchor_FP32": loss_max_error < 1e-6,
            "same_candidate_and_teacher_dependencies_as_R14": manifest["schedule"]["candidate_content"] == r14_manifest["schedule"]["candidate_content"] and manifest["teacher_scores"] == r14_manifest["teacher_scores"],
            "processed_pair_counts_match_schedule_and_R14": manifest["processed"] == r14_manifest["processed"] == schedule["processed"],
            "zero_new_teacher_inference_recorded": manifest["cost"]["new_teacher_inference"] == 0,
            "single_object_1024_projection_backbone_absent": model.student_dim == 1024 and model.input_dim == 4096 and grouped_names == needed_names,
            "recovered_R15_entrypoint_matches_training_hash": recovered_source["sha256"] == manifest["code_sha256"],
            "recovered_R14_entrypoint_matches_parent_training_hash": recovered_r14["sha256"] == r14_manifest["code_sha256"],
            "recovered_R14_optimizer_definition_AST_unchanged": r14_recovery["optimizer_definition_AST_unchanged"],
        }
        arms[arm] = {
            "checks": checks, "all_checks_pass": all(checks.values()),
            "initial_R14_file_sha256": checkpoint_fingerprint(reference_path),
            "saved_R15_step0_file_sha256": checkpoint_fingerprint(initial_path),
            "step0_files_byte_identical": checkpoint_fingerprint(reference_path) == checkpoint_fingerprint(initial_path),
            "base_P_R_parameter_names": base_parameter_names,
            "step0_A_nonzero": initial_a_nonzero, "step0_B_zero": initial_b_zero,
            "c_tau_by_checkpoint": scales,
            "parameter_trainability_loaded_from_saved_checkpoint": trainable,
            "actual_optimizer_groups_by_checkpoint": {step: value["optimizer"] for step, value in checkpoints.items()},
            "optimizer_state_evidence": "actual in-process summaries at save time; moment tensors were not serialized and cannot be independently reconstructed without replaying training",
            "checkpoint_hashes": checkpoint_file_checks,
            "loss": {"count": len(history), "max_abs_total_minus_D_minus_anchor": loss_max_error,
                     "first": history[0], "last": history[-1]},
            "source": {"executed_training_entrypoint_sha256": manifest["code_sha256"],
                       "current_entrypoint_sha256": checkpoint_fingerprint(training_source),
                       "current_file_matches_executed_file": checkpoint_fingerprint(training_source) == manifest["code_sha256"],
                       "current_train_arm_function_sha256": hashlib.sha256(inspect.getsource(train_arm).encode()).hexdigest(),
                       "recovered_historical_entrypoint": recovered_source,
                       "historical_entrypoint_hash_identity_verified": recovered_source["sha256"] == manifest["code_sha256"],
                       "limitation": "The current R15 entrypoint includes later evaluation code. Removing that addition recovered bytes whose SHA256 exactly matches both training manifests; training/shared function ASTs match the current file. This verifies the recorded entrypoint identity, not imported-module history, optimizer tensors, or the full historical execution environment.",
                       "shared_R14_helper_file_sha256": checkpoint_fingerprint(shared_source)},
            "manifest": {"path": str(directory / "manifest.json"), "sha256": checkpoint_fingerprint(directory / "manifest.json")},
            "optimizer_updates": manifest["optimizer_updates"],
            "new_teacher_inference": manifest["cost"]["new_teacher_inference"],
        }
        arms[arm]["source"].update({
            "executed_R14_training_entrypoint_sha256": r14_manifest["code_sha256"],
            "shared_full_file_matches_R14_execution": checkpoint_fingerprint(shared_source) == r14_manifest["code_sha256"],
            "recovered_R14_entrypoint": recovered_r14,
            "historical_R14_entrypoint_hash_identity_verified": recovered_r14["sha256"] == r14_manifest["code_sha256"],
            "historical_R14_optimizer_definition_AST_unchanged": r14_recovery["optimizer_definition_AST_unchanged"],
            "shared_helper_source_limit": "The current whole-file R14 hash differs from the full-parent training manifests because evaluation was added. Post hoc reconstruction recovers their exact recorded entrypoint hash, with the optimizer and 13 other shared/training function ASTs unchanged. This establishes optimizer-definition source identity, not historical optimizer moment tensors or the full imported-module runtime.",
        })
    invariants_pass = all(value["all_checks_pass"] for value in arms.values()) and all(value["matches"] for value in inputs.values()) and all(value["matches"] for value in schedule["actual_inputs"].values())
    result = {
        "status": "passed_with_archival_limitations" if invariants_pass else "failed",
        "training_invariants_status": "passed" if invariants_pass else "failed",
        "historical_source_and_full_optimizer_archival_status": "R14_R15_entrypoints_recovered_other_history_incomplete",
        "R15_entrypoint_recovery": {"receipt_path": str(recovery_path), "receipt_sha256": checkpoint_fingerprint(recovery_path),
                                    "recovered_source": recovered_source,
                                    "recorded_entrypoint_identity_verified": True,
                                    "imported_module_historical_identity_verified": False},
        "R14_entrypoint_recovery": {"receipt_path": str(recovery_path), "receipt_sha256": checkpoint_fingerprint(recovery_path),
                                    "recovered_source": recovered_r14,
                                    "recorded_entrypoint_identity_verified": True,
                                    "optimizer_definition_source_identity_verified": True,
                                    "imported_module_historical_identity_verified": False},
        "schedule": schedule, "frozen_R14_inputs": inputs, "arms": arms,
        "total_new_student_updates": sum(value["optimizer_updates"] for value in arms.values()),
        "new_teacher_inference": sum(value["new_teacher_inference"] for value in arms.values()),
        "teacher_execution_evidence": "Recorded run costs are zero; inspected training source only materializes teacher logits embedded in TargetExample, no Teacher checkpoint loading or forward call.",
        "runtime_archival_limitations": ["optimizer moment tensors absent", "historical imported-module runtime identities not established beyond the recovered R14/R15 entrypoints", "positive-mask hash replayed after training rather than logged during execution"],
        "code": {"path": str(Path(__file__).resolve()), "sha256": checkpoint_fingerprint(Path(__file__))},
    }
    destination = output / "stageI_interaction/statistics/training_invariants.json"
    write_json(destination, result)
    print(json.dumps({"status": result["status"], "output": str(destination), "failed_checks": {arm: [key for key, value in data["checks"].items() if not value] for arm, data in arms.items()}}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    audit(parser.parse_args().root.resolve())
