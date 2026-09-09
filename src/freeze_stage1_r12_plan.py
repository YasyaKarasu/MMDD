#!/usr/bin/env python
"""Freeze the R12 protocol identity before running the new experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


def run(root: Path, output: Path) -> None:
    plan = root / "stage1_optimization_r12_plan_20260908.md"
    if (output / "PLAN_FROZEN.json").exists():
        raise FileExistsError("R12 protocol is already frozen")
    for name in (
        "taskA_correctness", "taskB_attribute_audit", "taskC_training",
        "taskD_retention", "taskE_admission", "taskF_end_to_end",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)
    write_json(output / "PLAN_FROZEN.json", {
        "format_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_plan": str(plan),
        "source_plan_sha256": checkpoint_fingerprint(plan),
        "scope": "All Tasks A-F and conditional extensions in the source plan",
        "independent_confirmation_available": False,
        "transductive_unlabeled_lake": True,
        "protected_environment_file_access": "prohibited; no configured remote client authorized",
        "seeds": {"screen": 13, "conditional_student_variance": [17, 23]},
        "splits": {"train_fit": 11390, "cal_fit": 624, "cal_check": 616,
                   "dev": 1198, "r10_test_regression": 1166},
        "retrieval_budget": {"direct_k": 100, "text_k": 20, "image_k": 20,
                             "single_modality_k": 40, "targets_per_evidence": 20,
                             "top_l": 20, "evidence_budget": 4, "recall_ks": [10, 20, 50]},
        "training": {
            "arms": ["C-base", "C-candidates", "C-function"],
            "initialization": "PCA-1024, identity R, frozen encoder, trainable P/R",
            "batch_size": 64, "projection_lr": 1e-6, "relation_lr": 1e-5,
            "weight_decay": 0.01, "supervised_score": "sigmoid(raw)/0.1",
            "kd_score": "raw_logit", "kd_weight": 0.3, "kd_temperature": 1,
            "bce_weight": 0, "parameter_anchor_weight": 0.1,
            "parameter_anchor_reference": "P_PCA and identity R",
            "checkpoints": [0, 45, 89, 178, 267, 356],
            "full_dev_steps": [0, 178, 356], "extension_steps": [659, 1318],
            "extension_max_new_directions": 1, "health_validpool_ratio": 0.8,
            "stop_consecutive_unhealthy_full_evals": 2,
            "candidate_replacement_fraction": 0.5,
            "kd_coverage": "complete materialized candidate mask for all three arms",
            "function_regularizer": {"weight": 0.1, "pairs_per_relation": 256,
                                     "reference": "fixed train-visible PCA score, relation std normalization"},
            "lr_rescue": "one 0.1x P/R LR run only when all screen arms degrade",
            "path": {"updates": 356, "arms": ["path-only", "path+0.1edge"],
                     "condition": "edge start ValidPool >= 80% of own initialization"},
            "checkpoint_order": ["ValidPool_count", "RowB", "ValidB_count", "direct_R@10", "earlier_step"],
        },
        "attribute_audit": {
            "requested_rows": 256, "buckets": ["train_fit/text", "train_fit/image", "dev/text", "dev/image"],
            "per_bucket_strata": {"recovery_recheck": 16, "retrieved_high_score_unknown": 24,
                                  "same_entity_candidate_wrong_attribute": 16, "candidate_conflict": 8},
            "max_per_source_group": 4, "independent_double_review_fraction": 0.2,
            "support_model_gate": {"positive": 64, "negative": 64, "groups_each": 20},
            "unknown_is_negative": False,
        },
        "retention": ["D0 exact-content top-B", "D1 soft row coverage", "D2 unique argmax routing"],
        "admission": ["F1 union-direct", "F3 equal RRF", "F5 half quota"],
        "quality_max_drop": {"overall_R@10": 0.02, "implicit_R@10": 0.02, "explicit_R@10": 0.02},
        "numeric_tolerance": 1e-12,
        "end_to_end": {"queries": 128, "implicit": 96, "explicit": 32,
                       "sample_lock_required_before_new_outputs": True,
                       "column_permutation_seed": 13,
                       "conditions": ["no_evidence_same_generator", "retrieved", "wrong_attribute", "oracle_evidence"],
                       "full_pipeline_queues": ["F1", "frozen_fusion"],
                       "human_audit": "all new successes and fixed failure fraction"},
        "primary_comparisons": ["selected C arm minus C-base endpoint ValidPool",
                                "D2 minus D1 actual RoutedSupport",
                                "frozen fusion minus F1 EvidenceEnabledJoin/OutsideDirectFinalJoin"],
        "bootstrap": {"unit": "source_table_id", "paired": True, "iterations": 10000, "seed": 13},
        "hardware": {"gpus": ["RTX 4090 24GB", "RTX 4090 24GB"],
                     "concurrency": "several lightweight processes per GPU; bounded disk for temporary indices"},
        "completion": "Source-plan requirements remain authoritative; unknown or pending is not complete",
    })
    print(output / "PLAN_FROZEN.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output_dir.resolve())
