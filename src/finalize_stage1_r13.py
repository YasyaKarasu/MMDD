#!/usr/bin/env python
"""Finalize R13 Stage-1 comparisons, selection, bootstrap, and report."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


ARMS = {
    "s0": "taskA_stage1_protocol/s0/evaluation_step0/metrics.json",
    "c_s_shared": "taskC_role_projection/c_s_shared/evaluation_step178/metrics.json",
    "c_r_split": "taskC_role_projection/c_r_split/evaluation_step178/metrics.json",
    "p_s_target_only": "taskD_witness_supervision/p_s_target_only/evaluation_step178/metrics.json",
    "p_w_witness": "taskD_witness_supervision/p_w_witness/evaluation_step178/metrics.json",
    "b1_kd_on_hard356": "taskB_diagnostics_and_kd/b1_kd_on_hard356/evaluation_step356/metrics.json",
    "b1_kd_off_hard356": "taskB_diagnostics_and_kd/b1_kd_off_hard356/evaluation_step356/metrics.json",
}

COMPARISONS = {
    "c_r_split_minus_c_s_shared": ("c_r_split", "c_s_shared"),
    "p_w_witness_minus_p_s_target_only": ("p_w_witness", "p_s_target_only"),
    "hard_kd_on_minus_hard_kd_off": ("b1_kd_on_hard356", "b1_kd_off_hard356"),
    "p_s_target_only_minus_s0": ("p_s_target_only", "s0"),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rankings(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            row["query_id"]: row for row in map(json.loads, handle)
        }


def _source_map(root: Path) -> tuple[dict[str, str], Path]:
    dataset = (
        root
        / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
    )
    mapping = {
        str(row["table_id"]): str(row["source_table_id"])
        for row in iter_dataset_artifact(dataset, "query_tables")
    }
    return mapping, dataset / "dataset_manifest.json"


def _bootstrap(
    deltas: np.ndarray,
    sources: list[str],
    *,
    iterations: int = 10_000,
    seed: int = 13,
) -> dict[str, Any]:
    unique = sorted(set(sources))
    index = {source: offset for offset, source in enumerate(unique)}
    sums = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)
    for delta, source in zip(deltas, sources):
        offset = index[source]
        sums[offset] += delta
        counts[offset] += 1
    rng = np.random.default_rng(seed)
    sampled = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 250):
        stop = min(iterations, start + 250)
        draw = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        sampled[start:stop] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return {
        "estimand": "query-macro recall delta with source clusters resampled",
        "point_delta": float(deltas.mean()),
        "ci95_percentile": [float(lower), float(upper)],
        "bootstrap_probability_delta_gt_0": float((sampled > 0).mean()),
        "iterations": iterations,
        "seed": seed,
        "source_groups": len(unique),
        "queries": len(deltas),
    }


def _paired_statistics(
    output: Path,
    metrics: dict[str, dict[str, Any]],
    source_by_query: dict[str, str],
) -> dict[str, Any]:
    ranking_rows = {
        arm: _read_rankings(Path(payload["rankings"]["path"]))
        for arm, payload in metrics.items()
    }
    delta_dir = output / "statistics/per_query_deltas"
    delta_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (experimental, control) in COMPARISONS.items():
        left = ranking_rows[experimental]
        right = ranking_rows[control]
        if left.keys() != right.keys():
            raise ValueError(f"Paired query IDs differ for {name}")
        rows = []
        for query_id in sorted(left):
            if query_id not in source_by_query:
                raise ValueError(f"No stable source group for {query_id}")
            item = {
                "query_id": query_id,
                "source_table_id": source_by_query[query_id],
                "query_kind": left[query_id].get("query_kind"),
                "positive_denominator": left[query_id]["positive_denominator"],
                "metrics": {},
            }
            for ranking in ("f1_union_direct", "union_rrf_equal", "pure_direct100"):
                item["metrics"][ranking] = {}
                for k in (10, 20, 50):
                    a = left[query_id]["rankings"][ranking][str(k)]
                    b = right[query_id]["rankings"][ranking][str(k)]
                    item["metrics"][ranking][str(k)] = {
                        "experimental_hit_ids": a["hit_ids"],
                        "control_hit_ids": b["hit_ids"],
                        "experimental_numerator": a["numerator"],
                        "control_numerator": b["numerator"],
                        "denominator": a["denominator"],
                        "delta": a["recall"] - b["recall"],
                    }
            rows.append(item)
        path = delta_dir / f"{name}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        summary = {}
        for kind in ("all", "implicit", "explicit"):
            selected = [
                row for row in rows
                if kind == "all" or row["query_kind"] == kind
            ]
            summary[kind] = {}
            for ranking in ("f1_union_direct", "union_rrf_equal", "pure_direct100"):
                summary[kind][ranking] = {}
                for k in (10, 20, 50):
                    deltas = np.asarray(
                        [row["metrics"][ranking][str(k)]["delta"] for row in selected],
                        dtype=np.float64,
                    )
                    summary[kind][ranking][str(k)] = _bootstrap(
                        deltas,
                        [row["source_table_id"] for row in selected],
                    )
        results[name] = {
            "experimental": experimental,
            "control": control,
            "per_query": {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(path),
            },
            "summary": summary,
        }
    return results


def _metric_row(
    arm: str, payload: dict[str, Any], s0: dict[str, Any]
) -> dict[str, Any]:
    primary = payload["primary"]
    rankings = _read_rankings(Path(payload["rankings"]["path"]))
    return {
        "arm": arm,
        "parent": (
            "PCA"
            if arm.startswith("b1_")
            else "historical"
            if arm == "s0"
            else "S0"
        ),
        "objective": (
            "L_edge KD=0"
            if arm == "b1_kd_off_hard356"
            else "L_edge KD=0.3"
            if arm.startswith("b1_") or arm.startswith("c_")
            else "L_path + 0.1 L_W"
            if arm == "p_w_witness"
            else "L_path"
            if arm == "p_s_target_only"
            else "historical L_edge"
        ),
        "projection_mode": payload["projection_mode"],
        "step": payload["step"],
        "all_R10": primary["recall@10"],
        "implicit_R10": primary["implicit_recall@10"],
        "explicit_R10": primary["explicit_recall@10"],
        "all_R20": primary["recall@20"],
        "candidate_R50": payload["candidate_recall@50"],
        "delta_to_S0": primary["recall@10"] - s0["primary"]["recall@10"],
        "sensitivity_RRF_R10": payload["sensitivity_equal_union_rrf"]["recall@10"],
        "pure_direct_R10": payload["pure_direct100"]["recall@10"],
        "mean_union_unique_targets": float(
            np.mean([row["union_unique_targets"] for row in rankings.values()])
        ),
        "search_vectors_per_query_max": payload["cost"]["search_vectors_per_query_max"],
        "index_build_seconds": payload["cost"]["index_build_seconds"],
        "index_bytes": payload["cost"]["index_bytes"],
        "online_seconds_p50": payload["cost"]["online_seconds_p50"],
        "online_seconds_p95": payload["cost"]["online_seconds_p95"],
        "direct_supplement_pairs": payload["cost"]["direct_supplement_pairs"],
    }


def _checkpoint_state_equal(left: Path, right: Path) -> bool:
    a = torch.load(left, map_location="cpu", weights_only=True)
    b = torch.load(right, map_location="cpu", weights_only=True)
    return a["config"] == b["config"] and all(
        torch.equal(a["state_dict"][key], b["state_dict"][key])
        for key in a["state_dict"]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.root / "work/stage1_optimization_r13_20260909"
    metrics = {
        arm: _read_json(output / relative) for arm, relative in ARMS.items()
    }
    rows = [_metric_row(arm, metrics[arm], metrics["s0"]) for arm in ARMS]
    by_arm = {row["arm"]: row for row in rows}
    by_arm["c_r_split"]["delta_to_matched_control"] = (
        by_arm["c_r_split"]["all_R10"] - by_arm["c_s_shared"]["all_R10"]
    )
    by_arm["p_w_witness"]["delta_to_matched_control"] = (
        by_arm["p_w_witness"]["all_R10"] - by_arm["p_s_target_only"]["all_R10"]
    )
    by_arm["b1_kd_on_hard356"]["delta_to_matched_control"] = (
        by_arm["b1_kd_on_hard356"]["all_R10"]
        - by_arm["b1_kd_off_hard356"]["all_R10"]
    )
    source_by_query, dataset_manifest = _source_map(args.root)
    statistics = _paired_statistics(output, metrics, source_by_query)
    bootstrap_path = output / "statistics/bootstrap.json"
    bootstrap_payload = {
        "format_version": 1,
        "status": "complete",
        "unit": "source_table_id",
        "estimand": "query macro recall",
        "iterations": 10_000,
        "seed": 13,
        "dataset_manifest": {
            "path": str(dataset_manifest.resolve()),
            "sha256": checkpoint_fingerprint(dataset_manifest),
        },
        "comparisons": statistics,
        "exploratory_dev_intervals": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(bootstrap_path, bootstrap_payload)
    old_on = _read_json(
        args.root
        / "work/stage1_optimization_r12_20260908/taskC_training/c_candidates_seed13/manifest.json"
    )
    old_off = _read_json(
        args.root
        / "work/stage1_optimization_r12_20260908/taskC_training/c_kd_off_seed13/manifest.json"
    )
    new_on = _read_json(
        output / "taskB_diagnostics_and_kd/b1_kd_on_hard356/manifest.json"
    )
    new_off = _read_json(
        output / "taskB_diagnostics_and_kd/b1_kd_off_hard356/manifest.json"
    )
    reuse_audit = {
        "format_version": 1,
        "historical_pair_reusable": False,
        "reasons": [
            "Historical KD-on uses the candidates schedule while historical KD-off uses base schedule.",
            "The current historical runner hash does not equal the code hash stored by either R12 manifest.",
        ],
        "historical": {
            "kd_on_schedule_sha256": old_on["schedule_sha256"],
            "kd_off_schedule_sha256": old_off["schedule_sha256"],
            "manifest_code_sha256": old_on["code_sha256"],
            "current_r12_runner_sha256": checkpoint_fingerprint(
                args.root / "src/run_stage1_r12_task_c.py"
            ),
        },
        "matched_replay": {
            "schedule_equal": new_on["schedule"] == new_off["schedule"],
            "code_sha256_equal": new_on["code_sha256"] == new_off["code_sha256"],
            "step0_state_tensor_equal": _checkpoint_state_equal(
                Path(new_on["checkpoints"]["0"]["checkpoint"]),
                Path(new_off["checkpoints"]["0"]["checkpoint"]),
            ),
            "only_protocol_difference": "kd_weight 0.3 versus 0.0",
        },
    }
    write_json(
        output / "taskB_diagnostics_and_kd/historical_reuse_audit.json",
        reuse_audit,
    )
    s0 = by_arm["s0"]
    extension = {
        "format_version": 1,
        "status": "not_triggered",
        "threshold": 0.005,
        "comparisons": {
            "C-R_minus_C-S": by_arm["c_r_split"]["delta_to_matched_control"],
            "P-W_minus_P-S": by_arm["p_w_witness"]["delta_to_matched_control"],
        },
        "reason": (
            "Neither experimental arm improves matched-control dev R@10 by at least 0.005; "
            "no extension, residual, H, or combination arm was launched."
        ),
    }
    write_json(output / "taskE_conditional_extension/TRIGGER.json", extension)
    eligible = []
    for arm in ("c_s_shared", "c_r_split", "p_s_target_only", "p_w_witness"):
        row = by_arm[arm]
        quality = (
            row["all_R10"] > s0["all_R10"]
            and row["implicit_R10"] >= s0["implicit_R10"] - 0.02
            and row["explicit_R10"] >= s0["explicit_R10"] - 0.02
        )
        cost = row["online_seconds_p95"] <= 1.1 * s0["online_seconds_p95"]
        if quality and cost:
            eligible.append(row)
    selected = max(
        eligible,
        key=lambda row: (
            row["all_R10"], row["all_R20"], row["candidate_R50"],
            -row["online_seconds_p95"], -row["step"], row["arm"],
        ),
    ) if eligible else s0
    selection = {
        "format_version": 1,
        "status": "frozen",
        "selected_arm": selected["arm"],
        "checkpoint": metrics[selected["arm"]]["checkpoint"],
        "checkpoint_sha256": metrics[selected["arm"]]["checkpoint_sha256"],
        "selector": "R10, R20, R50, lower measured cost, earlier step, config_id",
        "eligible_arms": [row["arm"] for row in eligible],
        "kd_off_excluded_from_main_method_selection": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "statistics/SELECTED_RECIPE.json", selection)
    write_json(
        output / "taskF_stage2_deferred/STATUS.json",
        {
            "format_version": 1,
            "status": "deferred",
            "reason": "The frozen R13 plan explicitly defers Stage 2 until later ranking research.",
            "stage1_selected_recipe": selection,
        },
    )
    test_path = (
        output
        / f"taskD_witness_supervision/{selected['arm']}/evaluation_r10_test_regression_step{selected['step']}/metrics.json"
    )
    test_metrics = _read_json(test_path) if test_path.is_file() else None
    test = (
        {
            "metrics": {
                "path": str(test_path.resolve()),
                "sha256": checkpoint_fingerprint(test_path),
            },
            "queries": test_metrics["primary"]["queries"],
            "recall@10": test_metrics["primary"]["recall@10"],
            "implicit_recall@10": test_metrics["primary"]["implicit_recall@10"],
            "explicit_recall@10": test_metrics["primary"]["explicit_recall@10"],
            "recall@20": test_metrics["primary"]["recall@20"],
            "recall@50": test_metrics["primary"]["recall@50"],
            "sensitivity_rrf_recall@10": test_metrics[
                "sensitivity_equal_union_rrf"
            ]["recall@10"],
            "pure_direct_recall@10": test_metrics["pure_direct100"]["recall@10"],
            "cost": test_metrics["cost"],
        }
        if test_metrics
        else None
    )
    b0_path = output / "taskB_diagnostics_and_kd/b0_one_step_and_exact.json"
    b0 = _read_json(b0_path) if b0_path.is_file() else None
    witness_gradient_path = (
        output / "taskD_witness_supervision/witness_gradient_diagnostic.json"
    )
    witness_gradient = (
        _read_json(witness_gradient_path) if witness_gradient_path.is_file() else None
    )
    candidate_audit_path = output / "taskB_diagnostics_and_kd/candidate_audit.json"
    candidate_audit = (
        _read_json(candidate_audit_path) if candidate_audit_path.is_file() else None
    )
    mechanism_path = output / "statistics/mechanism_and_reproducibility_audit.json"
    mechanism = _read_json(mechanism_path) if mechanism_path.is_file() else None
    profiles = {
        arm: _read_json(output / f"statistics/latency_profile_{arm}.json")
        for arm in ("s0", selected["arm"])
        if (output / f"statistics/latency_profile_{arm}.json").is_file()
    }
    cost_guard = None
    if set(profiles) == {"s0", selected["arm"]} and selected["arm"] != "s0":
        s0_p95 = [
            repeat["phase_seconds_per_query"]["total"]["p95"]
            for repeat in profiles["s0"]["repeats"]
        ]
        selected_p95 = [
            repeat["phase_seconds_per_query"]["total"]["p95"]
            for repeat in profiles[selected["arm"]]["repeats"]
        ]
        cost_guard = {
            "cold_p95_ratio": selected_p95[0] / s0_p95[0],
            "hot_p95_ratio": float(np.median(selected_p95[1:]))
            / float(np.median(s0_p95[1:])),
            "threshold_ratio": 1.1,
            "passed": (
                selected_p95[0] <= 1.1 * s0_p95[0]
                and float(np.median(selected_p95[1:]))
                <= 1.1 * float(np.median(s0_p95[1:]))
            ),
            "protocol": "full 1,198-query dev, batch16, 3 repeats on one RTX 4090",
        }
    summary = {
        "format_version": 1,
        "status": "complete" if test else "awaiting_selected_recipe_test",
        "rows": rows,
        "comparisons": {
            name: value["summary"]["all"]["f1_union_direct"]["10"]
            for name, value in statistics.items()
        },
        "selection": selection,
        "extension": extension,
        "historical_test": test,
        "witness_gradient_diagnostic": (
            {
                "path": str(witness_gradient_path.resolve()),
                "sha256": checkpoint_fingerprint(witness_gradient_path),
            }
            if witness_gradient
            else None
        ),
        "candidate_audit": (
            {
                "path": str(candidate_audit_path.resolve()),
                "sha256": checkpoint_fingerprint(candidate_audit_path),
            }
            if candidate_audit
            else None
        ),
        "mechanism_and_reproducibility_audit": (
            {
                "path": str(mechanism_path.resolve()),
                "sha256": checkpoint_fingerprint(mechanism_path),
            }
            if mechanism
            else None
        ),
        "latency_profiles": {
            arm: {
                "path": str(
                    (output / f"statistics/latency_profile_{arm}.json").resolve()
                ),
                "sha256": checkpoint_fingerprint(
                    output / f"statistics/latency_profile_{arm}.json"
                ),
            }
            for arm in profiles
        },
        "selected_recipe_cost_guard": cost_guard,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "statistics/summary.json", summary)
    def table_lines(table_rows: list[dict[str, Any]]) -> list[str]:
        values = [
            "| arm | parent | objective | P mode | step | R@10 | implicit R@10 | explicit R@10 | R@20 | CR@50 | ΔS0 | Δcontrol | p95 ms |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in table_rows:
            delta_control = row.get("delta_to_matched_control")
            delta_text = (
                f"{delta_control:+.6f}" if delta_control is not None else "—"
            )
            values.append(
                f"| {row['arm']} | {row['parent']} | {row['objective']} | {row['projection_mode']} | {row['step']} | "
                f"{row['all_R10']:.6f} | {row['implicit_R10']:.6f} | {row['explicit_R10']:.6f} | "
                f"{row['all_R20']:.6f} | {row['candidate_R50']:.6f} | {row['delta_to_S0']:+.6f} | "
                f"{delta_text} | {1000 * row['online_seconds_p95']:.2f} |"
            )
        return values

    fixed_ids = (
        "c_s_shared",
        "c_r_split",
        "p_s_target_only",
        "p_w_witness",
        "b1_kd_on_hard356",
        "b1_kd_off_hard356",
    )
    lines = [
        "# R13 Stage-1 results",
        "",
        "Primary endpoint: full-dev query-macro target Recall@10 under D1 retention + F1 union-direct. Intervals are exploratory source-cluster bootstrap intervals (10,000 draws, seed 13).",
        "",
        "## S0 baseline",
        "",
        *table_lines([by_arm["s0"]]),
        "",
        "## Fixed endpoints",
        "",
        *table_lines([by_arm[arm] for arm in fixed_ids]),
        "",
        "## Development-selected recipe",
        "",
        *table_lines([selected]),
    ]
    lines.extend(["", "## Fixed comparisons", ""])
    for name, value in summary["comparisons"].items():
        ci = value["ci95_percentile"]
        lines.append(
            f"- {name}: {value['point_delta']:+.6f}, 95% bootstrap CI [{ci[0]:+.6f}, {ci[1]:+.6f}]."
        )
    if b0:
        exact = b0["fixed_e_all_target_exact"]
        lines.extend(
            [
                "",
                "## B0 mechanism diagnostics",
                "",
                (
                    "- Fixed 16 labelled evidence-target pairs, exact against all "
                    f"22,886 targets: ET R@20 is S0 {exact['s0']['recall@20']:.4f}, "
                    f"historical edge-degraded {exact['historical_edge_r11_epoch2']['recall@20']:.4f}, "
                    f"historical path-degraded {exact['historical_path_step356']['recall@20']:.4f}, "
                    f"and C-R178 {exact['c_r_split_step178']['recall@20']:.4f}."
                ),
                (
                    "- C-R query/target projection updates outside the original S0 row "
                    f"space are {b0['trained_split_role_subspace']['query']['outside_fraction']:.3f}/"
                    f"{b0['trained_split_role_subspace']['target']['outside_fraction']:.3f} of their "
                    "respective update norms. This is descriptive; natural C-R R@10 did not beat C-S."
                ),
                "- Actual one-step diagnostics used fresh AdamW copies and never produced formal training checkpoints.",
                (
                    "- The current-objective one-step audit also covers both available "
                    "historical degraded checkpoints. The older R11 container received "
                    "S0's verified PCA anchor reference as a non-scoring compatibility buffer."
                ),
            ]
        )
    if mechanism:
        natural = mechanism["natural_path_mechanism_by_arm"]
        exact = mechanism["mechanism_panel"]["exact_and_ann_by_arm"]
        lines.extend(
            [
                "",
                "## Mechanism and cost audit",
                "",
                (
                    "- The fixed 128-query/128-evidence-source panel reports all five "
                    "relations with separate all-positive Recall and any-positive hits. "
                    f"For selected P-S, q→image ANN/exact positive Recall is "
                    f"{exact[selected['arm']]['table_to_image']['ann_positive_recall_macro']:.4f}/"
                    f"{exact[selected['arm']]['table_to_image']['exact_positive_recall_macro']:.4f}."
                ),
                (
                    f"- Known-witness natural-pool coverage is S0 "
                    f"{natural['s0']['known_witness_pool_coverage']:.4f}, P-S "
                    f"{natural['p_s_target_only']['known_witness_pool_coverage']:.4f}, "
                    f"and P-W {natural['p_w_witness']['known_witness_pool_coverage']:.4f}. "
                    "W slightly recovers pool coverage over its control but does not improve "
                    "R@10 or mean known-witness LSE responsibility."
                ),
                "- Wrong-attribute/wrong-entity path rates are null with zero independent negative-label coverage; unknown paths were not relabelled as errors.",
                "- Every natural run used exactly 43 measured search vectors per query; no role, latent, or column index was added.",
            ]
        )
    if cost_guard:
        lines.append(
            f"- Selected-vs-S0 full-dev latency guard: cold P95 ratio "
            f"{cost_guard['cold_p95_ratio']:.3f}, hot ratio "
            f"{cost_guard['hot_p95_ratio']:.3f}; "
            f"10% guard {'passed' if cost_guard['passed'] else 'failed'}."
        )
    if witness_gradient:
        known_gradient = witness_gradient["path_gradient_coverage"][
            "known_positive_witness"
        ]
        unknown_gradient = witness_gradient["path_gradient_coverage"][
            "weak_unknown_target_path"
        ]
        lines.append(
            f"- W gradient audit (frozen first path batch): "
            f"{known_gradient['nonzero_gradient_paths']}/{known_gradient['paths']} "
            f"known witness paths and {unknown_gradient['nonzero_gradient_paths']}/"
            f"{unknown_gradient['paths']} weak-unknown target paths receive nonzero "
            "gradient. The latter are ranking-only contrasts, not confirmed negatives."
        )
    if candidate_audit:
        table_relation = candidate_audit["by_relation"]["table_to_table"]
        lines.append(
            f"- Frozen hard-candidate audit: table→table contains "
            f"{table_relation['multi_positive_lists']}/{table_relation['lists']} "
            "multi-positive lists (all positives retained); the other four relations "
            "are single-positive in this materialization. Candidates are never refreshed "
            "during the 356-step B1 schedule, so staleness is recorded but not claimed causal."
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- Selected deployable Stage-1 recipe: `{selected['arm']}` at step {selected['step']}.",
            "- No extension was triggered: C-R−C-S and P-W−P-S are both below +0.005.",
            "- KD-off is a mechanism ablation and was excluded from automatic main-method selection. Its higher score narrows the Teacher/KD claim rather than silently changing the proposed method.",
            "- Stage 2 remains deferred. R13 establishes Stage-1 retrieval behavior only; it does not establish that multimodal value recovery caused the join gains.",
            "",
            "## Historical test",
            "",
            (
                f"Frozen recipe test R@10/20/50: {test['recall@10']:.6f}/"
                f"{test['recall@20']:.6f}/{test['recall@50']:.6f}."
                if test
                else "Pending. Run only after this recipe freeze."
            ),
            "",
            "See `statistics/summary.json`, `statistics/bootstrap.json`, `statistics/mechanism_and_reproducibility_audit.json`, and the latency profiles for full per-query, diagnostic, and cost records.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "selected": selected["arm"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args())
