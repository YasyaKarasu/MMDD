#!/usr/bin/env python
"""Run Task A for the Stage-1 unified-pipeline experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import TargetExample, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import (
    RAW_EDGE_SCORE_PATH_AGGREGATIONS,
    PathAggregator,
)
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    build_indices,
    checkpoint_fingerprint,
    fuse_ranked_channels,
    load_corpus_ids,
    rank_detailed_paths,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.selection import load_stage1_selection
from mmdd_stage1.significance import paired_bootstrap_delta
from mmdd_stage1.sweep_metrics import (
    append_values as _append_values,
    finalize_records as _finalize_records,
    path_pool as _path_pool,
    query_values as _query_values,
)
from run_stage1_r6_sweeps import RECALL_KS
from run_stage1_r7_task_q import _lake_data

R5_ANCHORS = {"entitables": 0.3774, "wdc": 0.6436}


def aggregation_configs() -> list[dict[str, Any]]:
    """Return every supported form with a small allowed parameter grid."""

    configs = [
        {"form": "logsumexp", "top_k": 4, "temperature": 1.0, "power": 2.0},
        {"form": "max", "top_k": 4, "temperature": 1.0, "power": 2.0},
        {"form": "comb_mnz", "top_k": 4, "temperature": 1.0, "power": 2.0},
    ]
    for form in ("topk_mean", "topk_sum"):
        for top_k in (2, 4):
            configs.append(
                {"form": form, "top_k": top_k, "temperature": 1.0, "power": 2.0}
            )
    for form in ("logmeanexp", "topk_logmeanexp", "topk_logsumexp"):
        for temperature in (0.1, 0.3, 1.0):
            configs.append(
                {
                    "form": form,
                    "top_k": 4,
                    "temperature": temperature,
                    "power": 2.0,
                }
            )
    for temperature in (0.1, 0.3, 1.0):
        configs.append(
            {
                "form": "softmax_weighted_mean",
                "top_k": 4,
                "temperature": temperature,
                "power": 2.0,
            }
        )
    for power in (2.0, 3.0):
        configs.append(
            {"form": "power_mean", "top_k": 4, "temperature": 1.0, "power": power}
        )
    if {config["form"] for config in configs} != RAW_EDGE_SCORE_PATH_AGGREGATIONS:
        raise AssertionError(
            "Task A must cover every aggregation defined for raw edge scores"
        )
    for config in configs:
        suffix = ""
        if config["form"] in {"topk_mean", "topk_sum"}:
            suffix = f"_k{config['top_k']}"
        elif config["form"] in {"topk_logmeanexp", "topk_logsumexp"}:
            suffix = f"_k{config['top_k']}_t{config['temperature']:g}"
        elif config["form"] == "logmeanexp":
            suffix = f"_t{config['temperature']:g}"
        elif config["form"] == "softmax_weighted_mean":
            suffix = f"_t{config['temperature']:g}"
        elif config["form"] == "power_mean":
            suffix = f"_p{config['power']:g}"
        config["name"] = f"{config['form']}{suffix}"
    return configs


def fusion_configs() -> list[dict[str, Any]]:
    """Return RRF and normalized-score candidates with tunable lake weights."""

    configs = []
    for weight in (0.05, 0.1, 0.25, 0.5):
        configs.append(
            {
                "variant": "weighted_rrf",
                "mode": "weighted_rrf",
                "normalization": "none",
                "evidence_weight": weight,
                "name": f"weighted_rrf_e{weight:g}",
            }
        )
        for normalization in ("zscore", "minmax"):
            configs.append(
                {
                    "variant": f"normalized_score_{normalization}",
                    "mode": "normalized_score",
                    "normalization": normalization,
                    "evidence_weight": weight,
                    "name": f"normalized_score_{normalization}_e{weight:g}",
                }
            )
    return configs


def exact_name(aggregation: dict[str, Any], fusion: dict[str, Any]) -> str:
    return f"{aggregation['name']}__{fusion['name']}"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def select_parameterizations(
    records: dict[str, dict[int, dict[str, list[float]]]],
    specifications: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Select allowed parameters for each aggregation × fusion variant."""

    groups: dict[str, list[str]] = defaultdict(list)
    for name, specification in specifications.items():
        group = f"{specification['aggregation']['form']}__{specification['fusion']['variant']}"
        groups[group].append(name)
    selected = {}
    for group, names in groups.items():
        selected[group] = max(
            names,
            key=lambda name: (
                _mean(records[name][10]["fused_recall"]),
                _mean(records[name][10]["evidence_recall"]),
                _mean(records[name][10]["coverage"]),
                name,
            ),
        )
    return selected


def _unified_rows(lake_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    aggregation_forms = sorted(PATH_AGGREGATIONS)
    for aggregation in aggregation_forms:
        for fusion_mode in ("weighted_rrf", "normalized_score"):
            form_name = f"{aggregation}__{fusion_mode}"
            per_lake = {}
            for lake, payload in lake_payloads.items():
                variants = (
                    ["weighted_rrf"]
                    if fusion_mode == "weighted_rrf"
                    else ["normalized_score_zscore", "normalized_score_minmax"]
                )
                candidates = [
                    payload["family_metrics"][f"{aggregation}__{variant}"]
                    for variant in variants
                ]
                selected = max(
                    candidates,
                    key=lambda row: (
                        float(row["metrics"]["recall@10"]),
                        float(row["metrics"]["evidence"]["recall@10"]),
                        float(row["metrics"]["positive_evidence_path_coverage@10"]),
                    ),
                )
                per_lake[lake] = selected
            point_gate = all(row["external_point_gate"] for row in per_lake.values())
            formal_gate_available = all(
                row["formal_r5_gate"] is not None for row in per_lake.values()
            )
            ci_gate = formal_gate_available and all(
                row["formal_r5_gate"] for row in per_lake.values()
            )
            deltas = [
                float(row["delta_vs_same_run_baseline"]["delta_mean"])
                for row in per_lake.values()
            ]
            rows[form_name] = {
                "aggregation_form": aggregation,
                "fusion_form": fusion_mode,
                "per_lake": per_lake,
                "point_gate": point_gate,
                "ci_gate": ci_gate,
                "formal_r5_gate_available": formal_gate_available,
                "same_run_ci_gate": all(
                    row["same_run_ci_gate"] for row in per_lake.values()
                ),
                "minimum_delta": min(deltas),
                "mean_delta": sum(deltas) / len(deltas),
            }
    return rows


def select_unified_form(rows: dict[str, dict[str, Any]]) -> tuple[str, bool]:
    """Prefer a form passing both lakes, otherwise return the maximin candidate."""

    eligible = [
        name for name, row in rows.items() if row["point_gate"] and row["ci_gate"]
    ]
    candidates = eligible or list(rows)
    selected = max(
        candidates,
        key=lambda name: (
            float(rows[name]["minimum_delta"]),
            float(rows[name]["mean_delta"]),
            name,
        ),
    )
    return selected, bool(eligible)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _upgrade_lake_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach fair same-pool deltas and explicitly qualify the r5 reference."""

    r5_metrics = json.loads(
        Path(payload["r5_metrics_source"]).read_text(encoding="utf-8")
    )["systems"]["student"]["metrics"]
    r5_per_query = r5_metrics["per_query"]["fused"]["recall@10"]
    baseline = payload["family_metrics"]["logsumexp__weighted_rrf"]
    baseline_per_query = baseline["metrics"]["per_query"]["recall@10"]
    pool_reproduced = baseline_per_query == r5_per_query
    tolerance = float(payload["bootstrap"]["ci_tolerance"])
    for row in payload["family_metrics"].values():
        same_run_delta = row["delta_vs_baseline"]["recall@10"]
        external_delta = row.get("delta_vs_r5")
        if external_delta is None:
            external_delta = paired_bootstrap_delta(
                row["metrics"]["per_query"]["recall@10"],
                r5_per_query,
                iterations=int(payload["bootstrap"]["iterations"]),
                seed=int(payload["bootstrap"]["seed"]),
            )
        external_point_gate = (
            float(row["metrics"]["recall@10"]) >= float(payload["r5_anchor"])
        )
        row.update(
            {
                "delta_vs_same_run_baseline": same_run_delta,
                "delta_vs_r5_external": external_delta,
                "external_gap_vs_r5": float(row["metrics"]["recall@10"])
                - float(payload["r5_anchor"]),
                "external_point_gate": external_point_gate,
                "same_run_ci_gate": float(same_run_delta["ci_low"]) >= -tolerance,
                "formal_r5_gate": (
                    external_point_gate
                    and float(external_delta["ci_low"]) >= -tolerance
                    if pool_reproduced
                    else None
                ),
            }
        )
    payload.update(
        {
            "format_version": 2,
            "same_run_baseline": {
                "exact_configuration": baseline["exact_configuration"],
                "recall@10": baseline["metrics"]["recall@10"],
            },
            "r5_comparison": {
                "candidate_pool_reproduced": pool_reproduced,
                "formal_ci_gate_available": pool_reproduced,
                "interpretation": (
                    "same candidate pool; paired r5 gate is formal"
                    if pool_reproduced
                    else "different ANN candidate pool; r5 is an external absolute reference only"
                ),
            },
        }
    )
    return payload


def _rank_variants(
    paths: dict[str, list[dict[str, Any]]],
    aggregations: list[dict[str, Any]],
    fusions: list[dict[str, Any]],
) -> Iterable[tuple[str, dict[str, list[dict[str, Any]]]]]:
    for aggregation in aggregations:
        channels = rank_detailed_paths(
            paths,
            aggregator=PathAggregator(
                aggregation["form"],
                aggregation["top_k"],
                temperature=aggregation["temperature"],
                power=aggregation["power"],
            ),
            path_edge_normalization="none",
            rrf_k=60,
            fusion_mode="weighted_rrf",
            direct_weight=1.0,
            evidence_weight=0.05,
            gated_evidence_min_paths=2,
            gated_evidence_quantile=0.75,
        )
        for fusion in fusions:
            fused = fuse_ranked_channels(
                channels["direct"],
                channels["evidence"],
                rrf_k=60,
                fusion_mode=fusion["mode"],
                direct_weight=1.0,
                evidence_weight=fusion["evidence_weight"],
                score_normalization=fusion["normalization"],
                score_temperature=1.0,
            )
            yield exact_name(aggregation, fusion), {
                "fused": fused,
                "direct": channels["direct"],
                "evidence": channels["evidence"],
            }


def _append_batch(
    records: dict[str, dict[int, dict[str, list[float]]]],
    examples: list[TargetExample],
    detailed: list[dict[str, Any]],
    *,
    k: int,
    aggregations: list[dict[str, Any]],
    fusions: list[dict[str, Any]],
    pool_hash: hashlib._Hash,
) -> None:
    for example, baseline in zip(examples, detailed):
        paths = _path_pool(baseline)
        pool_hash.update(example.query_id.encode("utf-8"))
        for target_id in sorted(paths):
            pool_hash.update(b"\0")
            pool_hash.update(target_id.encode("utf-8"))
        for name, variant in _rank_variants(paths, aggregations, fusions):
            _append_values(records, name, k, _query_values(variant, example, k))


def _lake_result_lines(payload: dict[str, Any]) -> list[str]:
    r5_comparison = payload["r5_comparison"]
    lines = [
        f"# Task A: {payload['lake']} zero-training unified-form sweep",
        "",
        "These are selection-only results on the unchanged r5 full-R Student. "
        "Any row whose aggregation is not `logsumexp` is training–inference "
        "inconsistent and cannot be adopted without Task B/C retraining.",
        "",
        f"Same-run baseline: `{payload['same_run_baseline']['exact_configuration']}` "
        f"at {payload['same_run_baseline']['recall@10']:.2%}. Stored r5 reference: "
        f"{payload['r5_anchor']:.2%}.",
        "",
        f"r5 comparison status: **{r5_comparison['interpretation']}**. The paired "
        "same-run column is the valid Task-A aggregation/fusion comparison. The "
        "r5 gap is shown only as an absolute external reference when the original "
        "ANN pool was not reproduced.",
        "",
        "## Full R@10–50 candidates after lake-local parameter selection",
        "",
        "| Aggregation | Fusion variant | Exact parameters | Fused R@10 | Direct R@10 | Evidence R@10 | Coverage@10 | MRR@50 | Same-run Δ / 95% CI | Gap vs r5 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for group, row in sorted(payload["family_metrics"].items()):
        metrics = row["metrics"]
        delta = row["delta_vs_same_run_baseline"]
        aggregation, fusion = group.split("__", 1)
        lines.append(
            f"| {aggregation} | {fusion} | `{row['exact_configuration']}` | "
            f"{metrics['recall@10']:.2%} | {metrics['direct']['recall@10']:.2%} | "
            f"{metrics['evidence']['recall@10']:.2%} | "
            f"{metrics['positive_evidence_path_coverage@10']:.2%} | "
            f"{metrics['mrr@50']:.4f} | {delta['delta_mean']:+.2%} "
            f"[{delta['ci_low']:+.2%}, {delta['ci_high']:+.2%}] | "
            f"{row['external_gap_vs_r5']:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## R@10 parameter-screening grid",
            "",
            "| Exact configuration | Fused R@10 | Evidence R@10 | Coverage@10 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, row in sorted(
        payload["screening_metrics"].items(),
        key=lambda item: float(item[1]["metrics"]["recall@10"]),
        reverse=True,
    ):
        metrics = row["metrics"]
        lines.append(
            f"| `{name}` | {metrics['recall@10']:.2%} | "
            f"{metrics['evidence']['recall@10']:.2%} | "
            f"{metrics['positive_evidence_path_coverage@10']:.2%} |"
        )
    lines.append("")
    return lines


def _write_summary(task_root: Path) -> None:
    lake_payloads = {}
    for lake in ("entitables", "wdc"):
        path = task_root / lake / "metrics.json"
        if path.is_file():
            lake_payloads[lake] = json.loads(path.read_text(encoding="utf-8"))
    if len(lake_payloads) != 2:
        return
    rows = _unified_rows(lake_payloads)
    selected, any_eligible = select_unified_form(rows)
    payload = {
        "format_version": 2,
        "decision": (
            "zero_training_form_passes_both_r5_anchors"
            if any_eligible
            else "requires_training_side_validation"
        ),
        "selected_unified_form": selected,
        "eligible_unified_form_exists": any_eligible,
        "rows": rows,
        "lake_sources": {
            lake: str((task_root / lake / "metrics.json").resolve())
            for lake in lake_payloads
        },
    }
    _write_json(task_root / "summary.json", payload)
    lines = [
        "# Stage-1 pipeline unification Task A",
        "",
        f"Decision: **{payload['decision']}**. Zero-training recommendation for "
        f"Task B/C: `{selected}`.",
        "",
        "The aggregation and fusion function columns are unified across lakes. "
        "Exact top-k/temperature/power, evidence weight, and normalized-score "
        "normalization may differ by lake, as allowed by the plan.",
        "",
        "Formal r5 CI gates are available only when the saved r5 ANN candidate pool "
        "is reproduced. Otherwise selection uses paired same-run deltas and treats "
        "r5 as an external absolute reference.",
        "",
        "| Unified aggregation | Unified fusion | Enti exact / R@10 / same-run Δ / r5 gap | WDC exact / R@10 / same-run Δ / r5 gap | External point / formal CI gates |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in sorted(
        rows.items(),
        key=lambda item: (item[1]["minimum_delta"], item[1]["mean_delta"]),
        reverse=True,
    ):
        cells = []
        for lake in ("entitables", "wdc"):
            lake_row = row["per_lake"][lake]
            metrics = lake_row["metrics"]
            delta = lake_row["delta_vs_same_run_baseline"]
            cells.append(
                f"`{lake_row['exact_configuration']}` / "
                f"{metrics['recall@10']:.2%} / {delta['delta_mean']:+.2%} / "
                f"{lake_row['external_gap_vs_r5']:+.2%}"
            )
        lines.append(
            f"| {row['aggregation_form']} | {row['fusion_form']} | "
            f"{cells[0]} | {cells[1]} | {row['point_gate']} / {row['ci_gate']} |"
        )
    lines.extend(
        [
            "",
            "All Task-A rows use the existing r5 checkpoint. Therefore non-"
            "`logsumexp` rows are diagnostic only: they deliberately test the "
            "hypothesis before the training-consistent Task B/C rerun. The r8 "
            "WDC 65.35% retrieval-only result is not an adoptable unified-pipeline "
            "number.",
            "",
        ]
    )
    (task_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def refresh_existing_reports(output_root: Path) -> None:
    """Upgrade completed Task-A metrics and regenerate Markdown without retrieval."""

    task_root = output_root / "taskA_zero_training"
    for lake in ("entitables", "wdc"):
        metrics_path = task_root / lake / "metrics.json"
        if not metrics_path.is_file():
            continue
        payload = _upgrade_lake_payload(
            json.loads(metrics_path.read_text(encoding="utf-8"))
        )
        _write_json(metrics_path, payload)
        (metrics_path.parent / "RESULTS.md").write_text(
            "\n".join(_lake_result_lines(payload)), encoding="utf-8"
        )
    _write_summary(task_root)


def run(args: argparse.Namespace) -> None:
    if getattr(args, "report_only", False):
        refresh_existing_reports(args.output_root)
        return
    root = Path(__file__).resolve().parents[1]
    r5_root = root / "work/stage1_optimization_r5_20260829"
    final_selection = json.loads(
        (r5_root / "task4_final/selection.json").read_text(encoding="utf-8")
    )["selected"][args.lake]
    selection_path = Path(final_selection["student_selection"])
    selection = load_stage1_selection(selection_path)
    checkpoint = Path(final_selection["student_checkpoint"])
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    if checkpoint_sha256 != final_selection["student_checkpoint_sha256"]:
        raise ValueError("r5 final selection/checkpoint fingerprint mismatch")
    if checkpoint_sha256 != selection["best_checkpoint_sha256"]:
        raise ValueError("Student selection/checkpoint fingerprint mismatch")

    target_data, corpus = _lake_data(root, args.lake)
    corpus_sha256 = checkpoint_fingerprint(corpus)
    if corpus_sha256 != selection["corpus_sha256"]:
        raise ValueError("Student selection/corpus fingerprint mismatch")
    features = (
        root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    )
    device = torch.device(args.device)
    store = FeatureStore.from_path(features, cache_size=args.feature_cache_size)
    student = load_student(checkpoint, device).eval()
    index_path = Path(selection["best_index"])
    rebuilt_index = False
    if not (index_path / "manifest.json").is_file():
        index_path = args.output_root / "taskA_zero_training/indexes" / args.lake
        rebuilt_index = True
        if not (index_path / "manifest.json").is_file():
            ids_by_type = load_corpus_ids(corpus, store)
            build_indices(
                student,
                store,
                ids_by_type,
                index_path,
                device=device,
                checkpoint_sha256=checkpoint_sha256,
                corpus_sha256=corpus_sha256,
            )
    indices = StudentANNIndices(
        student,
        store,
        index_path,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )
    examples = load_target_examples(target_data, split="dev", dataset_name=target_data.stem)
    aggregations = aggregation_configs()
    fusions = fusion_configs()
    specifications = {
        exact_name(aggregation, fusion): {
            "aggregation": aggregation,
            "fusion": fusion,
        }
        for aggregation in aggregations
        for fusion in fusions
    }

    screening_records: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    pool_hashes = {k: hashlib.sha256() for k in RECALL_KS}
    started = time.perf_counter()
    for start in range(0, len(examples), args.query_batch_size):
        batch = examples[start : start + args.query_batch_size]
        detailed = retrieve_zero_one_hop_detailed_many(
            [example.query_id for example in batch],
            indices,
            k=10,
            gamma=10,
            gamma_evidence=2,
            evidence_types=("text", "image"),
            evidence_aggregation="logsumexp",
            evidence_top_k=4,
            fusion_mode="weighted_rrf",
            direct_weight=1.0,
            evidence_weight=0.05,
            query_batch_size=args.query_batch_size,
        )
        _append_batch(
            screening_records,
            batch,
            detailed,
            k=10,
            aggregations=aggregations,
            fusions=fusions,
            pool_hash=pool_hashes[10],
        )

    selected_parameterizations = select_parameterizations(
        screening_records, specifications
    )
    baseline = "logsumexp__weighted_rrf_e0.05"
    active_names = set(selected_parameterizations.values()) | {baseline}
    active_aggregations = {
        name: specifications[name]["aggregation"] for name in active_names
    }
    active_fusions = {name: specifications[name]["fusion"] for name in active_names}
    full_records = {
        name: defaultdict(
            lambda: defaultdict(list),
            {10: deepcopy(screening_records[name][10])},
        )
        for name in active_names
    }
    for k in RECALL_KS[1:]:
        for start in range(0, len(examples), args.query_batch_size):
            batch = examples[start : start + args.query_batch_size]
            detailed = retrieve_zero_one_hop_detailed_many(
                [example.query_id for example in batch],
                indices,
                k=k,
                gamma=10,
                gamma_evidence=2,
                evidence_types=("text", "image"),
                evidence_aggregation="logsumexp",
                evidence_top_k=4,
                fusion_mode="weighted_rrf",
                direct_weight=1.0,
                evidence_weight=0.05,
                query_batch_size=args.query_batch_size,
            )
            for example, baseline_result in zip(batch, detailed):
                paths = _path_pool(baseline_result)
                pool_hashes[k].update(example.query_id.encode("utf-8"))
                for target_id in sorted(paths):
                    pool_hashes[k].update(b"\0")
                    pool_hashes[k].update(target_id.encode("utf-8"))
                by_aggregation: dict[str, dict[str, list[dict[str, Any]]]] = {}
                for name in active_names:
                    aggregation = active_aggregations[name]
                    aggregation_name = aggregation["name"]
                    if aggregation_name not in by_aggregation:
                        by_aggregation[aggregation_name] = next(
                            _rank_variants(paths, [aggregation], [active_fusions[name]])
                        )[1]
                    channels = by_aggregation[aggregation_name]
                    fusion = active_fusions[name]
                    fused = fuse_ranked_channels(
                        channels["direct"],
                        channels["evidence"],
                        rrf_k=60,
                        fusion_mode=fusion["mode"],
                        direct_weight=1.0,
                        evidence_weight=fusion["evidence_weight"],
                        score_normalization=fusion["normalization"],
                        score_temperature=1.0,
                    )
                    _append_values(
                        full_records,
                        name,
                        k,
                        _query_values(
                            {
                                "fused": fused,
                                "direct": channels["direct"],
                                "evidence": channels["evidence"],
                            },
                            example,
                            k,
                        ),
                    )
    elapsed = time.perf_counter() - started

    screening_metrics = _finalize_records(
        screening_records,
        baseline,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    full_metrics = _finalize_records(
        full_records,
        baseline,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    r5_metrics_path = r5_root / f"task4_final/final_evaluation/{args.lake}/metrics.json"
    r5_metrics = json.loads(r5_metrics_path.read_text(encoding="utf-8"))
    r5_student = r5_metrics["systems"]["student"]["metrics"]
    r5_per_query = r5_student["per_query"]["fused"]["recall@10"]
    r5_anchor = float(r5_student["recall@10"])
    family_metrics = {}
    for group, name in selected_parameterizations.items():
        row = deepcopy(full_metrics[name])
        delta = paired_bootstrap_delta(
            row["metrics"]["per_query"]["recall@10"],
            r5_per_query,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        row.update(
            {
                "exact_configuration": name,
                "configuration": specifications[name],
                "delta_vs_r5": delta,
                "point_gate": row["metrics"]["recall@10"] >= r5_anchor - 1e-12,
                "ci_gate": delta["ci_low"] >= -0.02,
            }
        )
        family_metrics[group] = row

    payload = _upgrade_lake_payload({
        "format_version": 1,
        "lake": args.lake,
        "queries": len(examples),
        "student_checkpoint": str(checkpoint.resolve()),
        "student_checkpoint_sha256": checkpoint_sha256,
        "student_selection": str(selection_path.resolve()),
        "student_index": str(index_path.resolve()),
        "student_index_rebuilt": rebuilt_index,
        "r5_anchor": r5_anchor,
        "r5_plan_anchor": R5_ANCHORS[args.lake],
        "r5_metrics_source": str(r5_metrics_path.resolve()),
        "screening_protocol": "all exact parameter combinations at k=10",
        "full_protocol": "best lake-local parameters per aggregation form and fusion variant at k=10, then R@10-50",
        "training_inference_consistency": "diagnostic only; non-logsumexp aggregations require Task B/C retraining",
        "aggregations": aggregations,
        "fusions": fusions,
        "selected_parameterizations": selected_parameterizations,
        "candidate_pool_sha256": {
            str(k): digest.hexdigest() for k, digest in pool_hashes.items()
        },
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
            "ci_tolerance": 0.02,
        },
        "screening_metrics": screening_metrics,
        "family_metrics": family_metrics,
        "timing": {
            "seconds": elapsed,
            "device": args.device,
            "seconds_per_query_per_k": elapsed / (len(examples) * len(RECALL_KS)),
        },
    })
    task_root = args.output_root / "taskA_zero_training"
    output_dir = task_root / args.lake
    _write_json(output_dir / "metrics.json", payload)
    (output_dir / "RESULTS.md").write_text(
        "\n".join(_lake_result_lines(payload)), encoding="utf-8"
    )
    _write_summary(task_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", choices=["entitables", "wdc"])
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate reports from completed metrics without rerunning retrieval.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--feature-cache-size", type=int, default=16_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_pipeline_unification_20260831"),
    )
    args = parser.parse_args()
    if min(
        args.query_batch_size,
        args.feature_cache_size,
        args.bootstrap_iterations,
    ) <= 0:
        parser.error("batch sizes and bootstrap iterations must be positive")
    if not args.report_only and args.lake is None:
        parser.error("--lake is required unless --report-only is set")
    return args


if __name__ == "__main__":
    run(parse_args())
