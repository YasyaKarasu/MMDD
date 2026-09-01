#!/usr/bin/env python
"""Jointly evaluate r7 fusion and path aggregation for one Student run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import TargetExample, load_target_examples
from mmdd_stage1.evaluation import DEFAULT_RECALL_KS
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

RECALL_KS = DEFAULT_RECALL_KS


def _label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _lake_data(root: Path, lake: str) -> tuple[Path, Path]:
    r4 = root / "work/stage1_optimization_r4_20260829"
    if lake == "entitables":
        data = root / "work/stage1_stage2_entitables20k_v4_20260827/stage1_data"
    else:
        data = (
            root
            / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data"
        )
    return (
        data / "target_lists.jsonl",
        r4 / f"taskJ_per_lake_baselines/corpora/{lake}_corpus.jsonl",
    )


def _fusion_configs(lake: str) -> list[dict[str, Any]]:
    configs = [
        {
            "name": "weighted_rrf_e0.05",
            "mode": "weighted_rrf",
            "normalization": "none",
            "evidence_weight": 0.05,
        }
    ]
    if lake == "wdc":
        configs.append(
            {
                "name": "normalized_score_minmax_e0.05",
                "mode": "normalized_score",
                "normalization": "minmax",
                "evidence_weight": 0.05,
            }
        )
    else:
        configs.append(
            {
                "name": "normalized_score_zscore_e0.1",
                "mode": "normalized_score",
                "normalization": "zscore",
                "evidence_weight": 0.1,
            }
        )
    return configs


def _aggregation_configs(lake: str) -> list[dict[str, Any]]:
    configs = [
        {
            "name": "logsumexp_edges_none",
            "aggregation": "logsumexp",
            "normalization": "none",
        }
    ]
    configs.append(
        {
            "name": (
                "topk_sum_edges_none"
                if lake == "wdc"
                else "logsumexp_edges_zscore"
            ),
            "aggregation": "topk_sum" if lake == "wdc" else "logsumexp",
            "normalization": "none" if lake == "wdc" else "zscore",
        }
    )
    return configs


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_run_name(run_name: str | None) -> None:
    if run_name is not None and (
        Path(run_name).name != run_name or run_name in {"", ".", ".."}
    ):
        raise ValueError("--run-name must be one non-empty path component")


def _output_dir(args: argparse.Namespace) -> Path:
    """Return an isolated output directory for an optional checkpoint label."""
    directory = (
        args.output_root
        / "taskQ_joint_selection"
        / args.lake
        / f"lowrank_k_{args.rank}_mu_{_label(args.anchor_weight)}"
    )
    if args.run_name is not None:
        directory /= args.run_name
    return directory


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    target_data, corpus = _lake_data(root, args.lake)
    features = (
        root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    )
    source_run = (
        args.output_root
        / "taskP_lowrank"
        / args.lake
        / f"k_{args.rank}"
        / f"mu_{_label(args.anchor_weight)}"
    )
    checkpoint = Path(args.student_checkpoint) if args.student_checkpoint else source_run / "student_path.pt"
    selection_path = (
        Path(args.selection)
        if args.selection
        else checkpoint.with_suffix(checkpoint.suffix + ".selection.json")
    )
    history_path = checkpoint.with_suffix(checkpoint.suffix + ".history.json")
    selection = load_stage1_selection(selection_path)
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    if selection["best_checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Student selection/checkpoint fingerprint mismatch")

    device = torch.device(args.device)
    store = FeatureStore.from_path(features, cache_size=args.feature_cache_size)
    student = load_student(checkpoint, device).eval()
    index_path = Path(args.index_dir) if args.index_dir else Path(selection["best_index"])
    indices = StudentANNIndices(
        student,
        store,
        index_path,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=checkpoint_fingerprint(corpus),
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
                            records,
                            name,
                            k,
                            _query_values(variant, example, k),
                        )

    baseline = "weighted_rrf_e0.05__logsumexp_edges_none"
    metrics = _finalize_records(
        records,
        baseline,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    r5_metrics_path = (
        root
        / "work/stage1_optimization_r5_20260829/task4_final/final_evaluation"
        / args.lake
        / "metrics.json"
    )
    r5_metrics = json.loads(r5_metrics_path.read_text(encoding="utf-8"))
    r5_per_query = r5_metrics["systems"]["student"]["metrics"]["per_query"][
        "fused"
    ]["recall@10"]
    r5_anchor = float(
        r5_metrics["systems"]["student"]["metrics"]["recall@10"]
    )
    for row in metrics.values():
        row["delta_vs_r5"] = paired_bootstrap_delta(
            row["metrics"]["per_query"]["recall@10"],
            r5_per_query,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
    eligible = [
        (name, row)
        for name, row in metrics.items()
        if float(row["metrics"]["recall@10"]) >= r5_anchor
    ]
    ranked = sorted(
        eligible or list(metrics.items()),
        key=lambda item: (
            float(item[1]["metrics"]["recall@10"]),
            float(item[1]["metrics"]["evidence"]["recall@10"])
            + float(
                item[1]["metrics"]["positive_evidence_path_coverage@10"]
            ),
        ),
        reverse=True,
    )
    selected_name = ranked[0][0]
    # Retrieval-side selection does not require the training history.  Some
    # interrupted epoch-0-only runs have a valid checkpoint and index but no
    # history sidecar; keep those runs evaluable while making the provenance
    # gap explicit instead of reconstructing or inventing training metadata.
    history_source = "sidecar"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history_source = "missing"
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        history = {
            "student_config": checkpoint_payload.get(
                "config", student.config()
            ),
            "kd_target_teacher_alpha": None,
            "relation_loss_weights": None,
        }
    output_dir = _output_dir(args)
    payload = {
        "format_version": 1,
        "lake": args.lake,
        "student_checkpoint": str(checkpoint.resolve()),
        "student_checkpoint_sha256": checkpoint_sha256,
        "run_name": args.run_name,
        "student_config": history["student_config"],
        "adaptive_tau": history["kd_target_teacher_alpha"],
        "relation_loss_weights": history["relation_loss_weights"],
        "history_source": history_source,
        "r5_fused_anchor": r5_anchor,
        "r5_metrics_source": str(r5_metrics_path.resolve()),
        "planned_configuration_count": 16,
        "evaluated_configuration_count": len(metrics),
        "unevaluated_dimensions": {
            "relation_weight": "fixed by the selected training checkpoint",
            "tau": "fixed by the selected training checkpoint",
        },
        "hard_anchor_satisfied_by_any": bool(eligible),
        "selected": selected_name,
        "candidate_pool_sha256": {
            str(k): digest.hexdigest() for k, digest in pool_hashes.items()
        },
        "configurations": {
            "fusion": fusion_configs,
            "aggregation": aggregation_configs,
        },
        "metrics": metrics,
    }
    _write_json(output_dir / "metrics.json", payload)
    selected = metrics[selected_name]["metrics"]
    lines = [
        f"# Stage-1 r7 Task Q: {args.lake}",
        "",
        f"Checkpoint: `{checkpoint}` (SHA-256 `{checkpoint_sha256}`).",
        "",
        "This run evaluates the four retrieval-side combinations (2 fusion × 2 "
        "path aggregation). Relation weight and KD tau remain fixed by the "
        "training checkpoint; the planned 16-way joint sweep is therefore not "
        "claimed complete.",
        "",
        f"Selected `{selected_name}`; hard r5 anchor satisfied: `{bool(eligible)}`.",
        "",
        "| Configuration | Fused R@10 | Δ vs r5 / 95% CI | Evidence R@10 | Coverage@10 | MRR@50 |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for name, row in sorted(
        metrics.items(),
        key=lambda item: float(item[1]["metrics"]["recall@10"]),
        reverse=True,
    ):
        values = row["metrics"]
        delta = row["delta_vs_r5"]
        lines.append(
            f"| {name} | {values['recall@10']:.2%} | "
            f"{delta['delta_mean']:+.2%} [{delta['ci_low']:+.2%}, "
            f"{delta['ci_high']:+.2%}] | "
            f"{values['evidence']['recall@10']:.2%} | "
            f"{values['positive_evidence_path_coverage@10']:.2%} | "
            f"{values['mrr@50']:.4f} |"
        )
    lines.append("")
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", required=True, choices=["entitables", "wdc"])
    parser.add_argument("--rank", required=True, type=int, choices=[16, 64, 256])
    parser.add_argument("--anchor-weight", required=True, type=float, choices=[0.0, 0.1])
    parser.add_argument(
        "--student-checkpoint",
        help="Optional checkpoint override for recovered/intermediate runs.",
    )
    parser.add_argument(
        "--selection", help="Optional selection manifest matching --student-checkpoint."
    )
    parser.add_argument("--index-dir", help="Optional Student index directory override.")
    parser.add_argument(
        "--run-name",
        help=(
            "Optional one-component result label. Use this for recovered or "
            "intermediate checkpoints so their metrics cannot overwrite another run."
        ),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--feature-cache-size", type=int, default=16_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r7_20260831"),
    )
    values = parser.parse_args()
    if min(
        values.query_batch_size,
        values.feature_cache_size,
        values.bootstrap_iterations,
    ) <= 0:
        parser.error("batch sizes and bootstrap iterations must be positive")
    try:
        _validate_run_name(values.run_name)
    except ValueError as error:
        parser.error(str(error))
    return values


if __name__ == "__main__":
    run(parse_args())
