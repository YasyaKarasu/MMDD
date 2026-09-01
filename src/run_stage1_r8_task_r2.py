#!/usr/bin/env python
"""Evaluate r5 full-R Students with the four r7 Task-Q retrieval combinations."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import TargetExample, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    checkpoint_fingerprint,
    rank_detailed_paths,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.selection import load_stage1_selection
from mmdd_stage1.significance import paired_bootstrap_delta
from run_stage1_r6_sweeps import (
    _append_values,
    _finalize_records,
    _path_pool,
    _query_values,
)
from run_stage1_r7_task_q import (
    RECALL_KS,
    _aggregation_configs,
    _fusion_configs,
    _lake_data,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _r7_lowrank_metrics(root: Path, lake: str) -> Path:
    relative = (
        "entitables/lowrank_k_256_mu_0/selected_epoch_002/metrics.json"
        if lake == "entitables"
        else "wdc/lowrank_k_256_mu_0/metrics.json"
    )
    return (
        root
        / "work/stage1_optimization_r7_20260831/taskQ_joint_selection"
        / relative
    )


def _r6_controls(root: Path, lake: str) -> dict[str, Any]:
    task_root = root / "work/stage1_optimization_r6_20260830"
    fusion_path = task_root / f"taskB_fusion_normalization/{lake}.json"
    aggregation_path = task_root / f"taskD_path_aggregation/{lake}.json"
    fusion_payload = json.loads(fusion_path.read_text(encoding="utf-8"))
    aggregation_payload = json.loads(aggregation_path.read_text(encoding="utf-8"))
    fusion_name = _fusion_configs(lake)[1]["name"]
    aggregation_name = _aggregation_configs(lake)[1]["name"]
    return {
        "fusion_only": {
            "name": fusion_name,
            "metrics": fusion_payload["systems"]["student"]["configs"][fusion_name][
                "metrics"
            ],
            "source": str(fusion_path.resolve()),
        },
        "aggregation_only": {
            "name": aggregation_name,
            "metrics": aggregation_payload["systems"]["student"]["configs"][
                aggregation_name
            ]["metrics"],
            "source": str(aggregation_path.resolve()),
        },
    }


def select_configuration(
    metrics: dict[str, dict[str, Any]], lake: str, r5_anchor: float
) -> tuple[str, bool]:
    """Apply the r8 point-anchor and EntiTables evidence gates."""

    anchor_eligible = [
        name
        for name, row in metrics.items()
        if float(row["metrics"]["recall@10"]) >= r5_anchor
    ]
    double_win = [
        name
        for name in anchor_eligible
        if lake != "entitables"
        or float(metrics[name]["metrics"]["evidence"]["recall@10"]) >= 0.10
    ]
    candidates = double_win or anchor_eligible or list(metrics)
    selected = max(
        candidates,
        key=lambda name: (
            float(metrics[name]["metrics"]["recall@10"]),
            float(metrics[name]["metrics"]["evidence"]["recall@10"]),
            float(metrics[name]["metrics"]["positive_evidence_path_coverage@10"]),
        ),
    )
    enti_double_win = (
        lake == "entitables"
        and selected in double_win
        and float(metrics[selected]["metrics"]["evidence"]["recall@10"]) >= 0.10
    )
    return selected, enti_double_win


def _result_lines(payload: dict[str, Any]) -> list[str]:
    lake = str(payload["lake"])
    selected_name = str(payload["selected"])
    selected = payload["metrics"][selected_name]
    lines = [
        f"# Stage-1 r8 Task R2: {lake} full-R × Task-Q",
        "",
        f"Selected `{selected_name}`. Point anchor: "
        f"`{selected['r5_point_anchor_satisfied']}`; CI tolerance gate: "
        f"`{selected['r5_ci_gate_satisfied']}`; EntiTables fused+evidence double win: "
        f"`{payload['entitables_double_win']}`.",
        "",
        "| Configuration | Fused R@10 | Direct R@10 | Evidence R@10 | Coverage@10 | MRR@50 | Δ vs r5 / 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in sorted(
        payload["metrics"].items(),
        key=lambda item: float(item[1]["metrics"]["recall@10"]),
        reverse=True,
    ):
        values = row["metrics"]
        delta = row["delta_vs_r5"]
        lines.append(
            f"| {name} | {values['recall@10']:.2%} | "
            f"{values['direct']['recall@10']:.2%} | "
            f"{values['evidence']['recall@10']:.2%} | "
            f"{values['positive_evidence_path_coverage@10']:.2%} | "
            f"{values['mrr@50']:.4f} | {delta['delta_mean']:+.2%} "
            f"[{delta['ci_low']:+.2%}, {delta['ci_high']:+.2%}] |"
        )

    lines.extend(
        [
            "",
            "## r7 low-rank k=256 comparison (same four retrieval combinations)",
            "",
            "| Configuration | Fused R@10 | Evidence R@10 | Coverage@10 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, row in sorted(payload["r7_lowrank_k256"]["metrics"].items()):
        values = row["metrics"]
        lines.append(
            f"| {name} | {values['recall@10']:.2%} | "
            f"{values['evidence']['recall@10']:.2%} | "
            f"{values['positive_evidence_path_coverage@10']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## r6 one-dimensional full-R controls",
            "",
            "| Control | Configuration | Fused R@10 | Evidence R@10 | Coverage@10 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for label, control in payload["r6_controls"].items():
        values = control["metrics"]
        lines.append(
            f"| {label} | {control['name']} | {values['recall@10']:.2%} | "
            f"{values['evidence']['recall@10']:.2%} | "
            f"{values['positive_evidence_path_coverage@10']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Evaluation time: {payload['timing']['seconds']:.1f}s total, "
            f"{payload['timing']['seconds_per_query_per_k']:.4f}s/query/k.",
            "",
        ]
    )
    return lines


def _write_task_summary(task_root: Path) -> None:
    payloads = []
    for lake in ("entitables", "wdc"):
        path = task_root / lake / "metrics.json"
        if path.is_file():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    lines = [
        "# Stage-1 r8 Task R2: full-R × Task-Q retrieval recombination",
        "",
        "| Lake | Selected configuration | Fused R@10 | Evidence R@10 | Coverage@10 | Δ vs r5 / 95% CI | Point / CI gates |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for payload in payloads:
        name = payload["selected"]
        row = payload["metrics"][name]
        values = row["metrics"]
        delta = row["delta_vs_r5"]
        lines.append(
            f"| {payload['lake']} | {name} | {values['recall@10']:.2%} | "
            f"{values['evidence']['recall@10']:.2%} | "
            f"{values['positive_evidence_path_coverage@10']:.2%} | "
            f"{delta['delta_mean']:+.2%} [{delta['ci_low']:+.2%}, "
            f"{delta['ci_high']:+.2%}] | {row['r5_point_anchor_satisfied']} / "
            f"{row['r5_ci_gate_satisfied']} |"
        )
    lines.append("")
    (task_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    r5_root = root / "work/stage1_optimization_r5_20260829"
    final_selection_path = r5_root / "task4_final/selection.json"
    final_selection = json.loads(final_selection_path.read_text(encoding="utf-8"))[
        "selected"
    ][args.lake]
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
    if selection["corpus_sha256"] != corpus_sha256:
        raise ValueError("Student selection/corpus fingerprint mismatch")
    features = (
        root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    )
    device = torch.device(args.device)
    store = FeatureStore.from_path(features, cache_size=args.feature_cache_size)
    student = load_student(checkpoint, device).eval()
    indices = StudentANNIndices(
        student,
        store,
        Path(selection["best_index"]),
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )
    examples: list[TargetExample] = load_target_examples(
        target_data, split="dev", dataset_name=target_data.stem
    )
    fusion_configs = _fusion_configs(args.lake)
    aggregation_configs = _aggregation_configs(args.lake)
    records: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    pool_hashes = {k: hashlib.sha256() for k in RECALL_KS}
    started = time.perf_counter()
    for k in RECALL_KS:
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
            for example, baseline in zip(batch, detailed):
                paths = _path_pool(baseline)
                pool_hashes[k].update(example.query_id.encode("utf-8"))
                for target_id in sorted(paths):
                    pool_hashes[k].update(b"\0")
                    pool_hashes[k].update(target_id.encode("utf-8"))
                for aggregation in aggregation_configs:
                    for fusion in fusion_configs:
                        name = f"{fusion['name']}__{aggregation['name']}"
                        variant = rank_detailed_paths(
                            paths,
                            aggregator=PathAggregator(
                                aggregation["aggregation"], 4
                            ),
                            path_edge_normalization=aggregation["normalization"],
                            rrf_k=60,
                            fusion_mode=fusion["mode"],
                            direct_weight=1.0,
                            evidence_weight=fusion["evidence_weight"],
                            fusion_score_normalization=fusion["normalization"],
                            fusion_score_temperature=1.0,
                            gated_evidence_min_paths=2,
                            gated_evidence_quantile=0.75,
                        )
                        _append_values(
                            records, name, k, _query_values(variant, example, k)
                        )
    elapsed = time.perf_counter() - started

    baseline = "weighted_rrf_e0.05__logsumexp_edges_none"
    metrics = _finalize_records(
        records,
        baseline,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    r5_metrics_path = r5_root / f"task4_final/final_evaluation/{args.lake}/metrics.json"
    r5_metrics = json.loads(r5_metrics_path.read_text(encoding="utf-8"))
    r5_student = r5_metrics["systems"]["student"]["metrics"]
    r5_per_query = r5_student["per_query"]["fused"]["recall@10"]
    r5_anchor = float(r5_student["recall@10"])
    for row in metrics.values():
        delta = paired_bootstrap_delta(
            row["metrics"]["per_query"]["recall@10"],
            r5_per_query,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        row["delta_vs_r5"] = delta
        row["r5_point_anchor_satisfied"] = delta["delta_mean"] >= -1e-12
        row["r5_ci_gate_satisfied"] = delta["ci_low"] >= -0.02
    selected_name, enti_double_win = select_configuration(
        metrics, args.lake, r5_anchor
    )

    lowrank_path = _r7_lowrank_metrics(root, args.lake)
    lowrank_payload = json.loads(lowrank_path.read_text(encoding="utf-8"))
    payload = {
        "format_version": 1,
        "lake": args.lake,
        "queries": len(examples),
        "student_relation_param": "full",
        "student_checkpoint": str(checkpoint.resolve()),
        "student_checkpoint_sha256": checkpoint_sha256,
        "student_selection": str(selection_path.resolve()),
        "student_index": str(Path(selection["best_index"]).resolve()),
        "r5_fused_anchor": r5_anchor,
        "r5_metrics_source": str(r5_metrics_path.resolve()),
        "configurations": {
            "fusion": fusion_configs,
            "aggregation": aggregation_configs,
        },
        "candidate_pool_sha256": {
            str(k): digest.hexdigest() for k, digest in pool_hashes.items()
        },
        "candidate_pool_policy": "one fixed path pool per query/k across all four configurations",
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
            "ci_tolerance": 0.02,
        },
        "metrics": metrics,
        "selected": selected_name,
        "entitables_double_win": enti_double_win,
        "r7_lowrank_k256": {
            "source": str(lowrank_path.resolve()),
            "selected": lowrank_payload["selected"],
            "metrics": lowrank_payload["metrics"],
        },
        "r6_controls": _r6_controls(root, args.lake),
        "timing": {
            "seconds": elapsed,
            "seconds_per_query_per_k": elapsed / (len(examples) * len(RECALL_KS)),
            "device": args.device,
        },
    }
    task_root = args.output_root / "taskR2_full_r_task_q"
    output_dir = task_root / args.lake
    _write_json(output_dir / "metrics.json", payload)
    (output_dir / "RESULTS.md").write_text(
        "\n".join(_result_lines(payload)), encoding="utf-8"
    )
    _write_task_summary(task_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", required=True, choices=["entitables", "wdc"])
    parser.add_argument("--device", required=True)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--feature-cache-size", type=int, default=16_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r8_20260831"),
    )
    values = parser.parse_args()
    if min(
        values.query_batch_size,
        values.feature_cache_size,
        values.bootstrap_iterations,
    ) <= 0:
        parser.error("batch sizes and bootstrap iterations must be positive")
    return values


if __name__ == "__main__":
    run(parse_args())
