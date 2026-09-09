#!/usr/bin/env python
"""Summarize completed R12 artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from run_stage1_r11_task_f import R12_FUSION_IDS, _quality_selection
from summarize_stage1_r11 import grouped_paired_bootstrap


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _query_groups(dataset_root: Path) -> dict[str, str]:
    groups = {}
    for path in sorted((dataset_root / "query_tables").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                groups[str(row["table_id"])] = str(row["source_table_id"])
    if not groups:
        raise ValueError(f"No query source groups found under {dataset_root}")
    return groups


def _c2_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    retrieval = metrics["retrieval"]
    funnel = retrieval["evidence_funnel"]
    return {
        "valid_pool_count": funnel["valid_pool_count"],
        "row_b": funnel["row_b"],
        "valid_b_count": funnel["valid_b_count"],
        "direct_recall@10": retrieval["direct"]["recall@10"],
    }


def _retention_summary(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "valid_b_count",
        "row_b",
        "actual_routed_row_b",
        "matching_upper_row_b",
        "multi_row_2_count",
        "multi_row_3_count",
        "routing_error_rate",
        "mean_selected_evidence",
    )
    return {
        name: {field: row[field] for field in fields}
        for name, row in payload["results"].items()
    }


def _admission_summary(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "recall@10",
        "implicit_recall@10",
        "explicit_recall@10",
        "valid_path_count@10,4",
        "actual_routed_support@10,4",
        "valid_discovery_count@10,4",
        "positive_rescued_vs_f1@10",
        "positive_displaced_vs_f1@10",
    )
    return {
        "frozen_or_run_selection": payload["selection"],
        "exploratory_modality_selection": _quality_selection(
            payload["results"],
            fusion_ids=R12_FUSION_IDS,
            require_explicit=True,
        ),
        "rules": {
            name: {field: row[field] for field in fields}
            for name, row in payload["results"].items()
        },
    }


def _pool_summary(path: Path) -> dict[str, Any]:
    metadata = _load(path)
    return {
        "metadata": str(path.resolve()),
        "metadata_sha256": checkpoint_fingerprint(path),
        "pool": metadata["output"],
        "pool_sha256": metadata["output_sha256"],
    }


def _markdown(summary: dict[str, Any]) -> str:
    selected = summary["c2"]["selected"]
    c_bootstrap = summary["bootstrap"]["c_candidates_minus_c_base_step356_valid_pool"]
    d_bootstrap = summary["bootstrap"]["raw_d2_minus_d1_actual_routed_support"]
    task_f_bootstrap = summary["bootstrap"]["task_f_frozen_fusion_minus_f1"]
    if task_f_bootstrap["status"] == "complete":
        evidence = task_f_bootstrap["evidence_enabled_join"]
        outside = task_f_bootstrap["outside_direct_final_join"]
        task_f_line = (
            "- End-to-end frozen fusion minus F1: EvidenceEnabledJoin delta "
            f"{evidence['delta_mean']:+.2%}, 95% CI "
            f"[{evidence['ci95_low']:+.2%}, {evidence['ci95_high']:+.2%}]; "
            f"OutsideDirectFinalJoin delta {outside['delta_mean']:+.2%}, 95% CI "
            f"[{outside['ci95_low']:+.2%}, {outside['ci95_high']:+.2%}]."
        )
    else:
        task_f_line = (
            "- End-to-end frozen fusion minus F1: pending because Task F "
            "reader/generator outputs do not exist; no Stage 1 proxy is substituted."
        )
    lines = [
        "# R12 results",
        "",
        "## C2 checkpoint selection",
        "",
        f"Frozen ordering selected `{selected['arm']}` step {selected['step']}. ",
        "Because it is step 0, no C2 path-training gain is claimed.",
        "",
        "| Arm | Step | ValidPool | RowB | ValidB | Direct R@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["c2"]["records"]:
        metrics = row["summary"]
        lines.append(
            f"| `{row['arm']}` | {row['step']} | {metrics['valid_pool_count']} | "
            f"{metrics['row_b']:.2%} | {metrics['valid_b_count']} | "
            f"{metrics['direct_recall@10']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Conditional Student variance",
            "",
            "| Seed | Candidate ValidPool | Base ValidPool | Difference |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for seed, row in summary["c1_conditional_student_variance"]["by_seed"].items():
        lines.append(
            f"| {seed} | {row['candidates']['valid_pool_count']} | "
            f"{row['base']['valid_pool_count']} | "
            f"{row['difference']['valid_pool_count']:+d} |"
        )
    lines.extend(
        [
            "",
            "## Primary bootstraps",
            "",
            f"- C-candidates minus C-base step356 ValidPool: "
            f"{c_bootstrap['left_numerator']:.0f}-{c_bootstrap['right_numerator']:.0f} "
            f"of {c_bootstrap['denominator']:.0f}, delta {c_bootstrap['difference']:+.2%}, "
            f"95% CI [{c_bootstrap['ci95_low']:+.2%}, {c_bootstrap['ci95_high']:+.2%}].",
            f"- Raw D2 minus D1 actual RoutedSupport: delta {d_bootstrap['difference']:+.2%}, "
            f"95% CI [{d_bootstrap['ci95_low']:+.2%}, {d_bootstrap['ci95_high']:+.2%}].",
            task_f_line,
            "",
            "All computed intervals use source-table grouped paired bootstrap, 10,000 iterations, seed 13.",
            "",
        ]
    )
    if summary["task_f"]["status"] == "automated_complete_human_audit_pending":
        column = summary["task_f"]["column_evaluation"]
        lines.extend(
            [
                "## Task F end-to-end",
                "",
                f"The held-out candidate-column scorer reached {column['positive_column_accuracy']:.2%} "
                f"accuracy before rejection versus a {column['majority_position_accuracy']:.2%} "
                f"position baseline; control false-accept rate was {column['control_false_accept_rate']:.2%}.",
                "",
                "| System | True final joins | False final joins | Precision | Recall | Correct column | Correct value recovery |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for system, row in summary["task_f"]["by_system"].items():
            lines.append(
                f"| `{system}` | {row['true_final_joins']} | {row['false_final_joins']} | "
                f"{row['final_join_precision']:.2%} | {row['final_join_recall_all_gold']:.2%} | "
                f"{row['correct_column']:.2%} | {row['correct_value_recovery']:.2%} |"
            )
        fa = summary["task_f"]["f_a"]
        fa_coverage = summary["task_f"]["f_a_coverage"]
        lines.extend(
            [
                "",
                f"F-a evaluated {fa['pairs']} of {fa_coverage['recoverable_top10_pairs']} "
                f"recoverable top-10 pairs; {fa_coverage['without_retrieved_evidence']} had no "
                f"retrieved evidence. It found {fa['evidence_enabled_join_pairs']} evidence-enabled "
                "joins. Retrieved and oracle "
                "evidence both recovered 0 correct values; the required review packet is prepared but not reviewed.",
                "",
            ]
        )
    return "\n".join(lines)


def run(root: Path, output_root: Path) -> dict[str, Any]:
    task_c = output_root / "taskC_training"
    supervision = output_root / "taskA_correctness" / "supervision"
    dataset_root = Path(_load(supervision / "manifest.json")["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = root / dataset_root
    groups = _query_groups(dataset_root)
    task_f_sample_path = output_root / "taskF_end_to_end" / "queries_frozen.json"
    task_f_sample = _load(task_f_sample_path)
    legacy_stage2_path = root / "work/stage2_round1_20260831/summary_metrics.json"
    legacy_stage2 = _load(legacy_stage2_path)
    identifiability = legacy_stage2["conclusion"]["identifiability"]
    if identifiability != "confounded_by_degenerate_gold_column_position":
        raise ValueError("Unexpected legacy Stage-2 identifiability diagnosis")
    blocked_task_f_status = {
        "format_version": 1,
        "status": "blocked_protocol_prerequisite",
        "frozen_sample": str(task_f_sample_path.resolve()),
        "frozen_sample_sha256": checkpoint_fingerprint(task_f_sample_path),
        "queries": len(task_f_sample["selected"]),
        "implicit_queries": task_f_sample["actual"]["implicit"],
        "explicit_queries": task_f_sample["actual"]["explicit"],
        "source_groups": task_f_sample["source_groups"],
        "legacy_stage2_audit": str(legacy_stage2_path.resolve()),
        "legacy_stage2_audit_sha256": checkpoint_fingerprint(legacy_stage2_path),
        "legacy_scorer_identifiability": identifiability,
        "legacy_gold_column_position_counts": legacy_stage2["conclusion"][
            "gold_column_position_counts"
        ],
        "legacy_perfect_majority_baseline": legacy_stage2["conclusion"][
            "perfect_majority_baseline"
        ],
        "required_before_execution": [
            "Train and freeze a candidate-column scorer with synchronized seeded column permutations.",
            "Include wrong-target and no-available-column examples and report rejection behavior.",
            "Implement the R12 F-a interventions and paired F1/frozen-fusion full-chain runner.",
        ],
        "local_qwen_model_present": (root / "hf_models/Qwen3.5-9B/config.json").is_file(),
        "reason": (
            "The only existing candidate-column scorer is position-confounded, so "
            "running it would violate the frozen R12 Task F prerequisite"
        ),
    }
    task_f_dir = output_root / "taskF_end_to_end"
    task_f_metrics_path = task_f_dir / "metrics.json"
    column_metrics_path = task_f_dir / "column_scorer" / "cal_check_metrics.json"
    task_f_metrics = _load(task_f_metrics_path) if task_f_metrics_path.is_file() else None
    if task_f_metrics is None:
        task_f_status = blocked_task_f_status
    else:
        column_metrics = _load(column_metrics_path)
        task_f_status = {
            "format_version": 1,
            "status": task_f_metrics["status"],
            "metrics": str(task_f_metrics_path.resolve()),
            "metrics_sha256": checkpoint_fingerprint(task_f_metrics_path),
            "column_metrics": str(column_metrics_path.resolve()),
            "column_metrics_sha256": checkpoint_fingerprint(column_metrics_path),
            "column_identifiability": column_metrics["identifiability"],
            "column_evaluation": {
                "positive_column_accuracy": column_metrics["by_example_type"][
                    "positive"
                ]["column_accuracy_without_rejection"],
                "positive_decision_accuracy": column_metrics["by_example_type"][
                    "positive"
                ]["decision_accuracy"],
                "control_false_accept_rate": column_metrics[
                    "control_false_accept_rate"
                ],
                "majority_position_accuracy": column_metrics[
                    "majority_position_baseline"
                ]["accuracy"],
            },
            "by_system": task_f_metrics["by_system"],
            "f_a": task_f_metrics["f_a"],
            "f_a_coverage": {
                "recoverable_top10_pairs": task_f_metrics["inputs"][
                    "recoverable_pairs_in_top10_union"
                ],
                "evaluated_with_retrieved_evidence": task_f_metrics["f_a"][
                    "pairs"
                ],
                "without_retrieved_evidence": task_f_metrics["inputs"][
                    "recoverable_pairs_in_top10_union"
                ]
                - task_f_metrics["f_a"]["pairs"],
            },
            "human_audit": task_f_metrics["human_audit"],
            "human_audit_required": True,
            "reason": task_f_metrics["limitations"][0],
            "historical_blocked_prerequisite": blocked_task_f_status,
        }

    c_paths = {
        arm: task_c / "full_lake_evaluations" / f"r12_{arm}" / "step356" / "metrics.json"
        for arm in ("base", "candidates")
    }
    c_metrics = {name: _load(path) for name, path in c_paths.items()}
    c_rows = {
        name: metrics["retrieval"]["evidence_funnel"]["per_pair"]
        for name, metrics in c_metrics.items()
    }
    seed_metrics = {}
    for seed in (13, 17, 23):
        checkpoint_id = "step356" if seed == 13 else f"seed{seed}_step356"
        seed_metrics[seed] = {
            arm: _load(
                task_c
                / "full_lake_evaluations"
                / f"r12_{arm}"
                / checkpoint_id
                / "metrics.json"
            )
            for arm in ("base", "candidates")
        }
    variance_by_seed = {}
    for seed, metrics in seed_metrics.items():
        compact = {arm: _c2_summary(value) for arm, value in metrics.items()}
        variance_by_seed[str(seed)] = {
            **compact,
            "difference": {
                key: compact["candidates"][key] - compact["base"][key]
                for key in compact["base"]
            },
        }
    variance_fields = tuple(variance_by_seed["13"]["difference"])
    variance_dispersion = {
        key: {
            "mean": statistics.fmean(
                row["difference"][key] for row in variance_by_seed.values()
            ),
            "population_std": statistics.pstdev(
                row["difference"][key] for row in variance_by_seed.values()
            ),
            "min": min(
                row["difference"][key] for row in variance_by_seed.values()
            ),
            "max": max(
                row["difference"][key] for row in variance_by_seed.values()
            ),
        }
        for key in variance_fields
    }

    raw_d_path = output_root / "taskD_retention" / "raw" / "original" / "metrics.json"
    student_d_path = (
        output_root / "taskD_retention" / "student_selected" / "original" / "metrics.json"
    )
    raw_d = _load(raw_d_path)
    student_d = _load(student_d_path)
    d1_rows = raw_d["results"]["d1_soft_row_coverage"]["per_pair"]
    d2_rows = raw_d["results"]["d2_unique_argmax"]["per_pair"]

    bootstraps = {
        "c_candidates_minus_c_base_step356_valid_pool": grouped_paired_bootstrap(
            c_rows["candidates"],
            c_rows["base"],
            groups,
            key=lambda row: (row["query_id"], row["target_id"]),
            query_id=lambda row: str(row["query_id"]),
            numerator=lambda row: float(row["valid_pool"]),
            denominator=lambda _row: 1.0,
        ),
        "raw_d2_minus_d1_actual_routed_support": grouped_paired_bootstrap(
            d2_rows,
            d1_rows,
            groups,
            key=lambda row: (row["query_id"], row["target_id"]),
            query_id=lambda row: str(row["query_id"]),
            numerator=lambda row: float(row["actual_routed_row_b"]),
            denominator=lambda _row: 1.0,
        ),
        "task_f_frozen_fusion_minus_f1": (
            {
                "status": "complete",
                **task_f_metrics["bootstrap"],
            }
            if task_f_metrics is not None
            else {
                "status": "pending",
                "reason": task_f_status["reason"],
                "iterations": 10_000,
                "seed": 13,
                "unit": "source_table_id",
            }
        ),
    }

    c2_comparison_path = task_c / "selected_c2_seed13" / "comparison.json"
    c2 = _load(c2_comparison_path)
    admission_paths = {
        "raw_mixed": output_root / "taskE_admission" / "raw_mixed_d1" / "metrics.json",
        "raw_text40": output_root / "taskE_admission" / "raw_text40_d1" / "metrics.json",
        "raw_image40": output_root / "taskE_admission" / "raw_image40_d1" / "metrics.json",
        "student_mixed": output_root / "taskE_admission" / "student_selected_mixed_d1" / "metrics.json",
        "student_text40": output_root / "taskE_admission" / "student_selected_text40_d1" / "metrics.json",
        "student_image40": output_root / "taskE_admission" / "student_selected_image40_d1" / "metrics.json",
    }
    pool_dir = task_c / "selected_c2_seed13" / "path_pools"
    pool_metadata_paths = sorted(pool_dir.glob("*.jsonl.metadata.json"))
    summary = {
        "format_version": 1,
        "scope": (
            "Completed R12 C2, Task D, Task E, and automated Task F; "
            "Task F human audit remains pending"
            if task_f_metrics is not None
            else "Completed R12 C2, Task D, and Task E; Task F remains pending"
        ),
        "bootstrap_protocol": {
            "unit": "source_table_id",
            "paired": True,
            "iterations": 10_000,
            "seed": 13,
        },
        "c2": {
            "selected": c2["selected"],
            "records": c2["records"],
            "comparison_path": str(c2_comparison_path.resolve()),
            "comparison_sha256": checkpoint_fingerprint(c2_comparison_path),
            "c1_common_endpoint": {
                name: _c2_summary(metrics) for name, metrics in c_metrics.items()
            },
        },
        "c1_conditional_student_variance": {
            "teacher_fixed": True,
            "replicate_definition": (
                "seeded optimizer-batch permutation of the same frozen, fully "
                "Teacher-scored seed13 batches"
            ),
            "by_seed": variance_by_seed,
            "difference_dispersion": variance_dispersion,
            "interpretation": (
                "Conditional Student optimization variance only; this does not "
                "establish full Teacher-Student stability"
            ),
        },
        "task_d": {
            "raw": _retention_summary(raw_d),
            "student_selected": _retention_summary(student_d),
            "raw_metrics_sha256": checkpoint_fingerprint(raw_d_path),
            "student_metrics_sha256": checkpoint_fingerprint(student_d_path),
        },
        "selected_student_path_pools": {
            path.name.removesuffix(".jsonl.metadata.json"): _pool_summary(path)
            for path in pool_metadata_paths
        },
        "task_e": {
            name: {
                **_admission_summary(_load(path)),
                "metrics": str(path.resolve()),
                "metrics_sha256": checkpoint_fingerprint(path),
            }
            for name, path in admission_paths.items()
        },
        "task_f": task_f_status,
        "bootstrap": bootstraps,
        "limitations": [
            "The selected C2 checkpoint is step 0, so path training did not improve the frozen selection endpoint.",
            "No independent confirmation source groups are available; completed dev comparisons are exploratory.",
            *(
                task_f_metrics["limitations"]
                if task_f_metrics is not None
                else [
                    "Task F is not run because its existing candidate-column scorer is position-confounded and the corrected R12 runner is not implemented."
                ]
            ),
        ],
    }
    output = output_root / "statistics"
    write_json(output_root / "taskF_end_to_end" / "STATUS.json", task_f_status)
    write_json(output / "bootstrap.json", bootstraps)
    write_json(output / "summary.json", summary)
    (output / "RESULTS.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"output": str(output), "bootstrap": bootstraps}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output_root.resolve())
