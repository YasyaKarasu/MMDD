#!/usr/bin/env python
"""Write the auditable R12 completion, latency, and final-status reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing(values: Iterable[float]) -> dict[str, Any]:
    observed = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "observations": len(observed),
        "seconds_p50": _percentile(observed, 0.50),
        "seconds_p95": _percentile(observed, 0.95),
        "seconds_total": sum(observed) if observed else None,
    }


def build_latency(output_root: Path) -> dict[str, Any]:
    fusion_root = output_root / "taskA_correctness" / "fusion"
    task_a = {}
    for path in sorted(fusion_root.glob("*/*/metrics.json")):
        payload = _load(path)
        cost = payload.get("cost", {})
        task_a[str(path.parent.relative_to(fusion_root))] = {
            "queries": cost.get("queries"),
            "seconds_p50": cost.get("query_processing_seconds_p50"),
            "seconds_p95": cost.get("query_processing_seconds_p95"),
            "seconds_total": cost.get("elapsed_seconds"),
            "scope": cost.get("timing_scope"),
            "device": cost.get("device"),
        }

    task_f_root = output_root / "taskF_end_to_end"
    fa_rows = _load_jsonl(task_f_root / "f_a_predictions.jsonl")
    by_condition: dict[str, list[float]] = {}
    for row in fa_rows:
        for condition, result in row["conditions"].items():
            elapsed = result.get("elapsed_seconds")
            if elapsed is not None:
                by_condition.setdefault(str(condition), []).append(float(elapsed))

    full_rows = _load_jsonl(task_f_root / "full_chain_predictions.jsonl")
    full_non_reused: dict[str, list[float]] = {}
    reused: dict[str, int] = {}
    for row in full_rows:
        system = str(row["system"])
        if row.get("identical_queue_result_reused"):
            reused[system] = reused.get(system, 0) + 1
            continue
        full_non_reused.setdefault(system, []).append(float(row["elapsed_seconds"]))

    payload = {
        "format_version": 1,
        "measurement_policy": (
            "Only recorded elapsed values are summarized. Missing stage-specific "
            "measurements are null and are not interpreted as zero cost."
        ),
        "hardware": {
            "declared": ["RTX 4090 24GB", "RTX 4090 24GB"],
            "task_f_device": "cuda:0",
            "repeat_condition": "single recorded experimental pass",
        },
        "task_a_combined_direct_retention_fusion": task_a,
        "task_d_retention": {
            "seconds_p50": None,
            "seconds_p95": None,
            "reason": "The Task D runner did not record elapsed time.",
        },
        "task_e_admission": {
            "seconds_p50": None,
            "seconds_p95": None,
            "reason": "The Task E runner did not record elapsed time.",
        },
        "task_f": {
            "f_a_by_condition": {
                name: _timing(values) for name, values in sorted(by_condition.items())
            },
            "full_chain_non_reused_by_system": {
                name: _timing(values)
                for name, values in sorted(full_non_reused.items())
            },
            "identical_queue_results_reused": dict(sorted(reused.items())),
            "reader": {"seconds_p50": None, "seconds_p95": None},
            "localization": {"seconds_p50": None, "seconds_p95": None},
            "generation": {"seconds_p50": None, "seconds_p95": None},
            "stage_breakdown_reason": (
                "Reader, localization, and generation were timed only as a combined "
                "condition or full-chain call."
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_root / "LATENCY.json", payload)
    return payload


def _review_state(output_root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    attribute_path = output_root / "taskB_attribute_audit" / "review_summary.json"
    task_f_path = (
        output_root / "taskF_end_to_end" / "human_audit" / "review_summary.json"
    )
    attribute = _load(attribute_path) if attribute_path.is_file() else None
    task_f = _load(task_f_path) if task_f_path.is_file() else None
    return attribute, task_f


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": checkpoint_fingerprint(path) if path.is_file() else None,
    }


def _require_automated_artifacts(output_root: Path) -> None:
    required = [
        output_root / "taskA_correctness/supervision_audit.json",
        output_root / "taskA_correctness/fusion/raw/original_mixed/metrics.json",
        output_root / "taskA_correctness/fusion/raw/remove_image/metrics.json",
        output_root / "taskA_correctness/fusion/d1_probe/original_mixed/metrics.json",
        output_root / "taskA_correctness/fusion/d1_probe/remove_image/metrics.json",
        output_root / "taskA_correctness/fusion/raw_text40/original_mixed/metrics.json",
        output_root / "taskA_correctness/fusion/raw_text40/remove_image/metrics.json",
        output_root / "taskA_correctness/fusion/raw_image40/original_mixed/metrics.json",
        output_root / "taskA_correctness/fusion/raw_image40/remove_image/metrics.json",
        output_root / "taskC_training/gradient_diagnostics/summary.json",
        output_root / "taskC_training/candidate_quality/summary.json",
        output_root / "taskC_training/full_lake_evaluations/r12_base/step356/metrics.json",
        output_root / "taskC_training/full_lake_evaluations/r12_candidates/step356/metrics.json",
        output_root / "taskC_training/full_lake_evaluations/r12_function/step356/metrics.json",
        output_root / "taskC_training/full_lake_evaluations/r12_base_extension/step1318/metrics.json",
        output_root / "taskC_training/full_lake_evaluations/r12_candidates_extension/step1318/metrics.json",
        output_root / "taskC_training/selected_c2_seed13/comparison.json",
        output_root / "taskD_retention/raw/original/metrics.json",
        output_root / "taskD_retention/raw/remove_image/metrics.json",
        output_root / "taskD_retention/student_selected/original/metrics.json",
        *[
            output_root / "taskE_admission" / name / "metrics.json"
            for name in (
                "raw_mixed_d1",
                "raw_text40_d1",
                "raw_image40_d1",
                "student_selected_mixed_d1",
                "student_selected_text40_d1",
                "student_selected_image40_d1",
            )
        ],
        output_root / "taskF_end_to_end/column_scorer/cal_check_metrics.json",
        output_root / "taskF_end_to_end/f_a_predictions.jsonl",
        output_root / "taskF_end_to_end/full_chain_predictions.jsonl",
        output_root / "taskF_end_to_end/metrics.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "R12 automated completion evidence is missing:\n" + "\n".join(missing)
        )


def build_completion(output_root: Path) -> dict[str, Any]:
    _require_automated_artifacts(output_root)
    summary_path = output_root / "statistics" / "summary.json"
    task_f_metrics_path = output_root / "taskF_end_to_end" / "metrics.json"
    summary = _load(summary_path)
    task_f_metrics = _load(task_f_metrics_path)
    attribute_review, task_f_review = _review_state(output_root)
    attribute_complete = bool(
        attribute_review and attribute_review.get("status") == "complete"
    )
    task_f_review_complete = bool(
        task_f_review and task_f_review.get("status") == "complete"
    )

    task_b_root = output_root / "taskB_attribute_audit"
    primary_packets = _load_jsonl(task_b_root / "review_packets.jsonl")
    second_packets = _load_jsonl(task_b_root / "second_review_packets.jsonl")
    task_f_packets = _load_jsonl(
        output_root / "taskF_end_to_end" / "human_audit" / "review_packets.jsonl"
    )
    support_gate = (
        attribute_review["support_model_gate"] if attribute_complete else None
    )
    intervention_material = (
        attribute_review["intervention_material"] if attribute_complete else None
    )

    support_model_path = task_b_root / "support_model" / "metrics.json"
    support_model = (
        "not_triggered_insufficient_confirmed_labels"
        if attribute_complete and not support_gate["eligible"]
        else "complete"
        if support_model_path.is_file()
        else "pending_execution"
        if attribute_complete
        else "blocked_by_task_b_human_review"
    )
    redundancy_path = output_root / "taskD_retention" / "confirmed_same_row_redundancy" / "metrics.json"
    wrong_attribute_path = output_root / "taskD_retention" / "confirmed_wrong_attribute" / "metrics.json"
    if not attribute_complete:
        redundancy = wrong_attribute = "blocked_by_task_b_human_review"
    else:
        redundancy = (
            "not_triggered_insufficient_confirmed_material"
            if intervention_material["same_row_multiple_confirmed_evidence_contexts"] == 0
            else "complete"
            if redundancy_path.is_file()
            else "pending_execution"
        )
        wrong_attribute = (
            "not_triggered_insufficient_confirmed_material"
            if intervention_material["confirmed_wrong_attribute_cases"] == 0
            else "complete"
            if wrong_attribute_path.is_file()
            else "pending_execution"
        )
    decomposition_path = output_root / "taskE_admission" / "confirmed_attribute_support.json"
    confirmed_decomposition = (
        "complete"
        if decomposition_path.is_file()
        else "pending_execution"
        if attribute_complete
        else "blocked_by_task_b_human_review"
    )

    task_b_status = "complete" if attribute_complete else "human_review_pending"
    task_f_status = "complete" if task_f_review_complete else "human_review_pending"
    downstream_states = (
        support_model,
        redundancy,
        wrong_attribute,
        confirmed_decomposition,
    )
    downstream_pending = any(state == "pending_execution" for state in downstream_states)
    overall = (
        "human_review_pending"
        if not attribute_complete or not task_f_review_complete
        else "conditional_followups_pending"
        if downstream_pending
        else "complete"
    )

    c_bootstrap = summary["bootstrap"][
        "c_candidates_minus_c_base_step356_valid_pool"
    ]
    f_system = task_f_metrics["by_system"]["f1_union_direct"]
    payload = {
        "format_version": 1,
        "status": overall,
        "completion_rule": "unknown or pending is not complete",
        "tasks": {
            "A": {
                "status": "complete",
                "requirements": {
                    "supervision_and_mask": "complete",
                    "checkpoint_and_training_observation": "complete",
                    "evidence_channel_and_behavior_tests": "complete",
                    "frozen_pool_recomputation": "complete",
                },
                "evidence": [
                    _artifact(output_root / "taskA_correctness" / "supervision_audit.json"),
                    _artifact(
                        output_root
                        / "taskA_correctness/fusion/raw/original_mixed/metrics.json"
                    ),
                ],
            },
            "B": {
                "status": task_b_status,
                "primary_cases": len(primary_packets),
                "second_review_cases": len(second_packets),
                "independent_reviews_completed": len(primary_packets)
                if attribute_complete
                else 0,
                "model_assistance_counts_as_human_review": False,
                "support_model_gate": support_gate,
                "review_results": (
                    {
                        "overall": attribute_review["overall"],
                        "by_bucket": attribute_review["by_bucket"],
                        "actual_routing": attribute_review["actual_routing"],
                        "double_review_disagreements": attribute_review[
                            "double_review_disagreements"
                        ],
                    }
                    if attribute_complete
                    else None
                ),
                "review_summary": _artifact(task_b_root / "review_summary.json"),
            },
            "C": {
                "status": "complete",
                "requirements": {
                    "C0_diagnostics": "complete",
                    "C1_screen": "complete",
                    "C1_extensions": "complete",
                    "C1_student_seeds_17_23": "complete",
                    "C2_path_only_and_path_plus_edge": "complete",
                    "lr_rescue": "not_triggered_not_all_screen_arms_degraded",
                    "dynamic_student_refresh": (
                        "not_triggered_static_path_training_did_not_improve"
                    ),
                },
                "selected_edge_direction": "C-candidates step356",
                "selected_path_checkpoint": "path_only step0",
                "path_training_gain_claimed": False,
                "c_candidates_minus_base": {
                    "left_numerator": c_bootstrap["left_numerator"],
                    "right_numerator": c_bootstrap["right_numerator"],
                    "denominator": c_bootstrap["denominator"],
                    "difference": c_bootstrap["difference"],
                    "ci95_low": c_bootstrap["ci95_low"],
                    "ci95_high": c_bootstrap["ci95_high"],
                },
            },
            "D": {
                "status": "complete" if all(
                    state == "complete" or state.startswith("not_triggered")
                    for state in (support_model, redundancy, wrong_attribute)
                ) else "human_labels_or_followup_pending",
                "raw_and_student_base_strategies": "complete",
                "remove_image": "complete",
                "different_content_same_row_redundancy": redundancy,
                "wrong_attribute_replacement": wrong_attribute,
                "conditional_support_model": support_model,
            },
            "E": {
                "status": "complete"
                if confirmed_decomposition == "complete"
                else "human_labels_or_followup_pending",
                "raw_and_student_modality_runs": "complete",
                "frozen_selection": "f1_union_direct",
                "confirmed_attribute_support_decomposition": confirmed_decomposition,
            },
            "F": {
                "status": task_f_status,
                "automated_chain": "complete",
                "human_review_cases": len(task_f_packets),
                "independent_reviews_completed": len(task_f_packets)
                if task_f_review_complete
                else 0,
                "global_assignment": (
                    "not_triggered_actual_routing_close_to_matching_upper_bound_and_"
                    "value_generation_failed_upstream"
                ),
                "correct_value_recovery": {
                    "numerator": f_system["correct_value_numerator"],
                    "denominator": f_system["correct_value_denominator"],
                },
                "human_review_summary": _artifact(
                    output_root
                    / "taskF_end_to_end/human_audit/review_summary.json"
                ),
                "human_review_results": (
                    task_f_review["counts"] if task_f_review_complete else None
                ),
            },
        },
        "blocking_requirements": [
            *(
                [
                    "Task B: 256 independent primary reviews and 52 independent "
                    "second reviews"
                ]
                if not attribute_complete
                else []
            ),
            *(
                ["Task F: independent review of 3 blinded audit cases"]
                if not task_f_review_complete
                else []
            ),
            *(
                ["Run newly enabled conditional D/E analyses"]
                if attribute_complete and downstream_pending
                else []
            ),
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_root / "COMPLETION.json", payload)
    return payload


def _provenance_records(root: Path, output_root: Path) -> list[dict[str, Any]]:
    python = str(root / "src")
    r12 = str(output_root)
    features = str(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b")
    content = str(root / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl")
    raw = str(root / "work/stage1_optimization_r11_20260908/taskE_fixed_pool")
    selected = str(output_root / "taskC_training/selected_c2_seed13/path_pools")
    common_d = ["--features", features, "--content-keys", content, "--protocol", "r12"]
    records = [
        {
            "record_id": "r12-d-raw-original",
            "task": "Task D raw original",
            "command": [
                "conda", "run", "-n", "MMDD", "python",
                f"{python}/run_stage1_r11_task_e.py", "--path-pool",
                f"{raw}/raw_dev.jsonl", *common_d, "--output-dir",
                f"{r12}/taskD_retention/raw/original",
            ],
            "output": f"{r12}/taskD_retention/raw/original/metrics.json",
        },
        {
            "record_id": "r12-d-raw-remove-image",
            "task": "Task D raw remove-image",
            "command": [
                "conda", "run", "-n", "MMDD", "python",
                f"{python}/run_stage1_r11_task_e.py", "--path-pool",
                f"{raw}/raw_dev.jsonl", *common_d, "--output-dir",
                f"{r12}/taskD_retention/raw/remove_image", "--intervention",
                "remove_image",
            ],
            "output": f"{r12}/taskD_retention/raw/remove_image/metrics.json",
        },
        {
            "record_id": "r12-d-student-original",
            "task": "Task D selected Student original",
            "command": [
                "conda", "run", "-n", "MMDD", "python",
                f"{python}/run_stage1_r11_task_e.py", "--path-pool",
                f"{selected}/student_mixed_dev.jsonl", *common_d, "--output-dir",
                f"{r12}/taskD_retention/student_selected/original",
                "--feature-cache-size", "60000",
            ],
            "output": f"{r12}/taskD_retention/student_selected/original/metrics.json",
        },
    ]
    for system, prefix, device in (
        ("raw", raw, "cpu"),
        ("student_selected", selected, "cuda:0"),
    ):
        for modality, pool_name in (
            ("mixed", "raw" if system == "raw" else "student_mixed"),
            ("text40", "raw_text40" if system == "raw" else "student_text40"),
            ("image40", "raw_image40" if system == "raw" else "student_image40"),
        ):
            output_name = f"{system}_{modality}_d1"
            command = [
                "conda", "run", "-n", "MMDD", "python",
                f"{python}/run_stage1_r11_task_f.py", "--dev-pool",
                f"{prefix}/{pool_name}_dev.jsonl", "--cal-fit-pool",
                f"{prefix}/{pool_name}_cal_fit.jsonl", "--raw-reference-pool",
                f"{raw}/raw_dev.jsonl", "--features", features, "--content-keys",
                content, "--output-dir", f"{r12}/taskE_admission/{output_name}",
                "--retention", "d1_soft_row_coverage", "--protocol", "r12",
                "--device", device,
            ]
            if system != "raw":
                command.extend(["--feature-cache-size", "60000"])
            if modality != "mixed":
                command.extend(
                    [
                        "--frozen-selection", "f1_union_direct",
                        "--frozen-selection-source",
                        f"{r12}/taskE_admission/raw_mixed_d1/metrics.json",
                    ]
                )
            records.append(
                {
                    "record_id": f"r12-e-{system}-{modality}",
                    "task": f"Task E {system} {modality}",
                    "command": command,
                    "output": f"{r12}/taskE_admission/{output_name}/metrics.json",
                }
            )
    records.extend(
        [
            {
                "record_id": "r12-f-cache-column-train-forward",
                "task": "Task F cache candidate-column train split (forward shards)",
                "command": [
                    "conda", "run", "-n", "MMDD", "python",
                    f"{python}/run_stage1_r12_task_f.py", "cache-column",
                    "--protocol-split", "train_fit", "--device", "cuda:0",
                    "--dtype", "bf16", "--shard-size", "32",
                ],
                "output": (
                    f"{r12}/taskF_end_to_end/column_scorer/reader_cache/"
                    "train_fit/manifest.json"
                ),
            },
            {
                "record_id": "r12-f-cache-column-train-reverse",
                "task": "Task F cache candidate-column train split (reverse shards)",
                "command": [
                    "conda", "run", "-n", "MMDD", "python",
                    f"{python}/run_stage1_r12_task_f.py", "cache-column",
                    "--protocol-split", "train_fit", "--device", "cuda:0",
                    "--dtype", "bf16", "--shard-size", "32", "--reverse-shards",
                ],
                "output": (
                    f"{r12}/taskF_end_to_end/column_scorer/reader_cache/"
                    "train_fit/manifest.json"
                ),
            },
            {
                "record_id": "r12-f-cache-column-calibration",
                "task": "Task F cache candidate-column calibration splits",
                "command": [
                    "conda", "run", "-n", "MMDD", "python",
                    f"{python}/run_stage1_r12_task_f.py", "cache-column",
                    "--protocol-split", "cal_fit", "cal_check", "--device",
                    "cuda:0", "--dtype", "bf16", "--shard-size", "32",
                ],
                "output": (
                    f"{r12}/taskF_end_to_end/column_scorer/reader_cache/"
                    "cal_check/manifest.json"
                ),
            },
            {
                "record_id": "r12-f-train-column",
                "task": "Task F train candidate-column scorer",
                "command": [
                    "conda", "run", "-n", "MMDD", "python",
                    f"{python}/run_stage1_r12_task_f.py", "train-column",
                ],
                "output": f"{r12}/taskF_end_to_end/column_scorer/seed_13/candidate.pt",
            },
            {
                "record_id": "r12-f-evaluate-column",
                "task": "Task F evaluate candidate-column scorer",
                "command": [
                    "conda", "run", "-n", "MMDD", "python",
                    f"{python}/run_stage1_r12_task_f.py", "evaluate-column",
                ],
                "output": f"{r12}/taskF_end_to_end/column_scorer/cal_check_metrics.json",
            },
            {
                "record_id": "r12-f-end-to-end",
                "task": "Task F end-to-end",
                "command": [
                    "conda", "run", "-n", "MMDD", "python",
                    f"{python}/run_stage1_r12_task_f.py", "run-end-to-end",
                    "--phase", "both", "--device", "cuda:0",
                ],
                "output": f"{r12}/taskF_end_to_end/metrics.json",
            },
        ]
    )
    for record in records:
        record.update(
            {
                "status": "complete_recovered_provenance",
                "output_exists": Path(record["output"]).exists(),
                "source": "recovered from Codex execution history",
            }
        )
    return records


def write_provenance(root: Path, output_root: Path) -> dict[str, Any]:
    records = _provenance_records(root, output_root)
    payload = {
        "format_version": 1,
        "records": records,
        "note": (
            "These canonical completed commands supplement runs.jsonl records that "
            "were not emitted by the original D/E/F runners."
        ),
    }
    write_json(output_root / "PROVENANCE_SUPPLEMENT.json", payload)
    runs_path = output_root / "runs.jsonl"
    existing = {
        str(row.get("record_id"))
        for row in _load_jsonl(runs_path)
        if row.get("record_id") is not None
    }
    with runs_path.open("a", encoding="utf-8") as handle:
        for record in records:
            if record["record_id"] not in existing:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return payload


def _markdown(completion: dict[str, Any], summary: dict[str, Any], latency: dict[str, Any]) -> str:
    c = completion["tasks"]["C"]["c_candidates_minus_base"]
    f = summary["task_f"]
    f1 = f["by_system"]["f1_union_direct"]
    task_rows = [
        (name, row["status"])
        for name, row in completion["tasks"].items()
    ]
    attribute_complete = completion["tasks"]["B"]["status"] == "complete"
    task_f_review_complete = completion["tasks"]["F"]["status"] == "complete"
    interpretation_tail = (
        "The independently reviewed attribute labels are summarized in "
        "`taskB_attribute_audit/review_summary.json`; label-dependent D/E claims "
        "remain limited until every triggered follow-up shown in `COMPLETION.json` "
        "is executed."
        if attribute_complete
        else "They remain unproven until independent labels are available."
    )
    support_gate_line = (
        f"- The support-model gate resolved to "
        f"`{completion['tasks']['D']['conditional_support_model']}`; the two "
        "label-backed Task D interventions follow their recorded material gates."
        if attribute_complete
        else "- The support-model gate and the two label-backed Task D interventions "
        "await Task B review labels."
    )
    lines = [
        "# Stage 1 R12 final status",
        "",
        f"Overall status: **{completion['status']}**.",
        "",
        "The frozen rule is `unknown or pending is not complete`. Automated work is "
        "finished, but independent human review and label-dependent analyses are not "
        "reported as complete.",
        "",
        "## Completion matrix",
        "",
        "| Task | Status |",
        "| --- | --- |",
        *[f"| {name} | `{status}` |" for name, status in task_rows],
        "",
        "## Main findings",
        "",
        f"- C-candidates reached {c['left_numerator']:.0f} versus "
        f"{c['right_numerator']:.0f} ValidPool pairs out of {c['denominator']:.0f} "
        f"at step 356: {c['difference']:+.2%}, grouped bootstrap 95% CI "
        f"[{c['ci95_low']:+.2%}, {c['ci95_high']:+.2%}].",
        "- Seeds 17 and 23 reproduced the conditional Student gain (+32 and +31 "
        "ValidPool), while both extended arms became unhealthy only at step 1318.",
        "- C2 selected `path_only` step 0; neither trained path objective established "
        "a path-training gain.",
        "- Raw D2 minus D1 actual RoutedSupport was -0.53 percentage points, with "
        "95% CI [-1.11, +0.03]; sparse argmax retention did not improve routing.",
        "- The quality gate selected F1 itself for both Raw and Student modality "
        "comparisons, so no fusion improvement is claimed.",
        f"- Task F produced {f1['true_final_joins']} true and "
        f"{f1['false_final_joins']} false final joins; correct value recovery was "
        f"{f1['correct_value_numerator']}/{f1['correct_value_denominator']}. Oracle "
        "evidence also recovered no correct values.",
        *(
            [
                "- Independent Task F review counts: "
                + ", ".join(
                    f"{name}={value}"
                    for name, value in completion["tasks"]["F"][
                        "human_review_results"
                    ].items()
                )
                + "."
            ]
            if task_f_review_complete
            else []
        ),
        "",
        "## Interpretation",
        "",
        "The controlled C1 result supports the hypothesis that candidate-distribution "
        "mismatch contributed to training damage. It does not establish successful "
        "path learning. The automated end-to-end result is a negative mechanism "
        "result: better candidate-column identification did not translate into value "
        "recovery, and verifier precision remained too low. Attribute support, "
        "multimodal complementarity, and evidence-enabled join claims require "
        "independent evidence. " + interpretation_tail,
        "",
        "## Conditional decisions",
        "",
        "- Learning-rate rescue was not triggered because not all C1 screen arms degraded.",
        "- Dynamic Student refresh was not triggered because static path training did not improve.",
        "- Global row assignment was not triggered because actual routing was close to "
        "its matching upper bound and value generation failed upstream.",
        support_gate_line,
        "",
        "## Latency",
        "",
        "Task A combined query-processing P50/P95 values and Task F combined "
        "condition/full-chain P50/P95 values are stored in `LATENCY.json`. Separate "
        "reader, localization, and generation measurements were not recorded and are "
        "therefore `null`, not zero.",
        "",
        "## Required human work",
        "",
        *(
            [f"- {item}" for item in completion["blocking_requirements"]]
            if completion["blocking_requirements"]
            else ["- No blocking requirements remain."]
        ),
        "",
        "Reviewers should edit `taskB_attribute_audit/primary_reviews.jsonl`, "
        "`taskB_attribute_audit/second_reviews.jsonl`, and "
        "`taskF_end_to_end/human_audit/human_reviews.jsonl`, then run "
        "`finalize_stage1_r12_reviews.py finalize` followed by this finalizer.",
        "",
        "## Artifacts",
        "",
        "- `COMPLETION.json`: machine-readable requirement matrix and blockers.",
        "- `LATENCY.json`: measured timing summaries and explicit missing fields.",
        "- `PROVENANCE_SUPPLEMENT.json`: recovered canonical D/E/F commands.",
        "- `statistics/summary.json`: detailed results and bootstrap records.",
        "",
    ]
    del latency
    return "\n".join(lines)


def run(root: Path, output_root: Path) -> dict[str, Any]:
    root = root.resolve()
    output_root = output_root.resolve()
    plan_path = output_root / "PLAN_FROZEN.json"
    if _load(plan_path).get("completion") != (
        "Source-plan requirements remain authoritative; unknown or pending is not complete"
    ):
        raise ValueError("Unexpected R12 completion rule")
    latency = build_latency(output_root)
    completion = build_completion(output_root)
    provenance = write_provenance(root, output_root)
    summary = _load(output_root / "statistics" / "summary.json")
    final_path = output_root / "FINAL.md"
    final_path.write_text(_markdown(completion, summary, latency), encoding="utf-8")
    final_record = {
        "record_id": "r12-finalizer",
        "task": "R12 completion finalizer",
        "status": completion["status"],
        "command": [sys.executable, *sys.argv],
        "output": str(final_path),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    runs_path = output_root / "runs.jsonl"
    existing_ids = {
        str(row.get("record_id")) for row in _load_jsonl(runs_path)
    }
    if final_record["record_id"] not in existing_ids:
        with runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final_record, ensure_ascii=False) + "\n")
    return {
        "status": completion["status"],
        "completion": str((output_root / "COMPLETION.json").resolve()),
        "latency": str((output_root / "LATENCY.json").resolve()),
        "provenance_records": len(provenance["records"]),
        "final": str(final_path.resolve()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run(args.root, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
