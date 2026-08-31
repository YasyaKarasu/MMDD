#!/usr/bin/env python
"""Run Stage-1 round-six zero-training diagnostics and sweeps for one lake."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from evaluate_stage1_teacher_retrieval import _TeacherFeatureView, _load_object_ids
from mmdd_stage1.checkpoints import load_student, load_teacher
from mmdd_stage1.data import TargetExample, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    checkpoint_fingerprint,
    fuse_ranked_channels,
    load_corpus_ids,
    load_or_build_raw_embedding_indices,
    rank_detailed_paths,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.selection import load_stage1_selection
from mmdd_stage1.significance import paired_bootstrap_delta
from mmdd_stage1.teacher_rerank import TeacherRerankedANNIndices

RECALL_KS = (10, 20, 30, 40, 50)


@dataclass(frozen=True)
class LakeInputs:
    lake: str
    dataset: str
    dev_data: Path
    corpus: Path
    raw_index: Path
    student_selection: Path
    student_checkpoint: Path
    teacher_checkpoint: Path
    manual_tau: float


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.total_absolute = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, values: Iterable[float]) -> None:
        for raw in values:
            value = float(raw)
            self.count += 1
            self.total += value
            self.total_square += value * value
            self.total_absolute += abs(value)
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)

    def result(self) -> dict[str, float | int | None]:
        if not self.count:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "abs_mean": None,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "abs_mean": self.total_absolute / self.count,
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _lake_inputs(root: Path, lake: str) -> LakeInputs:
    r4 = root / "work/stage1_optimization_r4_20260829"
    r5 = root / "work/stage1_optimization_r5_20260829"
    final_selection = json.loads(
        (r5 / "task4_final/selection.json").read_text(encoding="utf-8")
    )["selected"][lake]
    if lake == "entitables":
        data = root / "work/stage1_stage2_entitables20k_v4_20260827/stage1_data"
        teacher = r4 / "taskM_entitables_teacher/checkpoints/teacher_path.pt"
        dataset = "entitables20k_v4"
    else:
        data = root / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data"
        teacher = r4 / "taskM_entitables_teacher/per_lake/wdc/checkpoints/teacher_path.pt"
        dataset = "wdc2k_v2"
    return LakeInputs(
        lake=lake,
        dataset=dataset,
        dev_data=data / "target_lists.jsonl",
        corpus=r4 / f"taskJ_per_lake_baselines/corpora/{lake}_corpus.jsonl",
        raw_index=r4 / f"taskJ_per_lake_baselines/epoch0/{lake}/raw_index",
        student_selection=Path(final_selection["student_selection"]),
        student_checkpoint=Path(final_selection["student_checkpoint"]),
        teacher_checkpoint=teacher,
        manual_tau=float(final_selection["tau"]),
    )


def _examples(path: Path) -> list[TargetExample]:
    return load_target_examples(path, split="dev", dataset_name=path.stem)


def _path_pool(result: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_target = {}
    for row in [*result["direct"], *result["evidence"]]:
        by_target[str(row["target_id"])] = row["paths"]
    return by_target


def _positive_evidence(example: TargetExample) -> dict[str, set[str]]:
    positives = set(example.positive_target_ids)
    return {
        candidate.target_id: set(candidate.evidence_ids)
        for candidate in example.candidates
        if candidate.target_id in positives and candidate.evidence_ids
    }


def _query_values(
    result: dict[str, list[dict[str, Any]]], example: TargetExample, k: int
) -> dict[str, float]:
    positives = set(example.positive_target_ids)

    def recall(channel: str) -> float:
        ids = {str(row["target_id"]) for row in result[channel][:k]}
        return len(ids & positives) / len(positives)

    fused = result["fused"]
    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, row in enumerate(fused[:k], 1)
            if str(row["target_id"]) in positives
        ),
        0.0,
    )
    gold_evidence = _positive_evidence(example)
    coverage = any(
        str(row["target_id"]) in gold_evidence
        and any(
            path["kind"] == "evidence"
            and str(path["evidence_id"]) in gold_evidence[str(row["target_id"])]
            for path in row["paths"]
        )
        for row in fused[:10]
    )
    return {
        "fused_recall": recall("fused"),
        "direct_recall": recall("direct"),
        "evidence_recall": recall("evidence"),
        "reciprocal_rank": reciprocal_rank,
        "coverage": float(coverage),
    }


def _append_values(
    records: dict[str, dict[int, dict[str, list[float]]]],
    config: str,
    k: int,
    values: dict[str, float],
) -> None:
    for name, value in values.items():
        records[config][k][name].append(value)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _finalize_records(
    records: dict[str, dict[int, dict[str, list[float]]]],
    baseline: str,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    output = {}
    for config, by_k in records.items():
        metrics: dict[str, Any] = {"queries": len(by_k[min(by_k)]["fused_recall"])}
        for k, values in sorted(by_k.items()):
            metrics[f"recall@{k}"] = _mean(values["fused_recall"])
            metrics.setdefault("direct", {})[f"recall@{k}"] = _mean(
                values["direct_recall"]
            )
            metrics.setdefault("evidence", {})[f"recall@{k}"] = _mean(
                values["evidence_recall"]
            )
        max_k = max(by_k)
        metrics[f"mrr@{max_k}"] = _mean(by_k[max_k]["reciprocal_rank"])
        if 10 in by_k:
            metrics["positive_evidence_path_coverage@10"] = _mean(
                by_k[10]["coverage"]
            )
        metrics["per_query"] = {
            "recall@10": by_k[10]["fused_recall"] if 10 in by_k else [],
            f"mrr@{max_k}": by_k[max_k]["reciprocal_rank"],
            "coverage@10": by_k[10]["coverage"] if 10 in by_k else [],
        }
        output[config] = {"metrics": metrics}

    baseline_values = records[baseline]
    for config, payload in output.items():
        values = records[config]
        deltas = {}
        for name, k, field in (
            ("recall@10", 10, "fused_recall"),
            ("coverage@10", 10, "coverage"),
            (f"mrr@{max(baseline_values)}", max(baseline_values), "reciprocal_rank"),
        ):
            if k not in values or k not in baseline_values:
                continue
            deltas[name] = paired_bootstrap_delta(
                values[k][field],
                baseline_values[k][field],
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
        payload["delta_vs_baseline"] = deltas
    return output


def _fusion_configs() -> list[dict[str, Any]]:
    normalizations = [
        ("zscore", 1.0),
        ("minmax", 1.0),
        ("softmax", 1.0),
        ("softmax", 0.1),
        ("softmax", 0.3),
    ]
    configs = [
        {
            "name": "weighted_rrf_e0.05",
            "fusion_mode": "weighted_rrf",
            "score_normalization": "none",
            "score_temperature": 1.0,
            "evidence_weight": 0.05,
        }
    ]
    for mode in ("normalized_score", "normalized_rrc"):
        for normalization, temperature in normalizations:
            suffix = (
                f"_{normalization}_t{temperature:g}"
                if normalization == "softmax"
                else f"_{normalization}"
            )
            for weight in (0.05, 0.1, 0.25, 0.5):
                configs.append(
                    {
                        "name": f"{mode}{suffix}_e{weight:g}",
                        "fusion_mode": mode,
                        "score_normalization": normalization,
                        "score_temperature": temperature,
                        "evidence_weight": weight,
                    }
                )
    return configs


def _aggregation_configs() -> list[dict[str, Any]]:
    methods = [
        ("logsumexp", 1.0, 2.0),
        ("max", 1.0, 2.0),
        ("topk_mean", 1.0, 2.0),
        ("topk_sum", 1.0, 2.0),
        ("softmax_weighted_mean", 1.0, 2.0),
        ("softmax_weighted_mean", 0.1, 2.0),
        ("softmax_weighted_mean", 0.3, 2.0),
        ("power_mean", 1.0, 2.0),
        ("power_mean", 1.0, 3.0),
        ("comb_mnz", 1.0, 2.0),
    ]
    configs = []
    for aggregation, temperature, power in methods:
        parameter = ""
        if aggregation == "softmax_weighted_mean":
            parameter = f"_t{temperature:g}"
        elif aggregation == "power_mean":
            parameter = f"_p{power:g}"
        for normalization in ("none", "zscore"):
            configs.append(
                {
                    "name": f"{aggregation}{parameter}_edges_{normalization}",
                    "aggregation": aggregation,
                    "temperature": temperature,
                    "power": power,
                    "path_edge_normalization": normalization,
                }
            )
    return configs


def _add_scale_stats(
    stats: dict[str, RunningStats], result: dict[str, list[dict[str, Any]]]
) -> None:
    paths_by_target = _path_pool(result)
    query_edges: dict[tuple[str, str], float] = {}
    for paths in paths_by_target.values():
        for path in paths:
            if path["kind"] == "direct":
                stats["table_to_table"].add([float(path["path_score"])])
                continue
            evidence_type = str(path["evidence_type"])
            query_edges[(evidence_type, str(path["evidence_id"]))] = float(
                path["query_evidence_score"]
            )
            stats[f"{evidence_type}_to_table"].add(
                [float(path["evidence_target_score"])]
            )
    for (evidence_type, _evidence_id), score in query_edges.items():
        stats[f"table_to_{evidence_type}"].add([score])


def _scale_payload(stats: dict[str, RunningStats]) -> dict[str, Any]:
    for relation in (
        "table_to_table",
        "table_to_text",
        "text_to_table",
        "table_to_image",
        "image_to_table",
    ):
        stats[relation]
    relations = {name: values.result() for name, values in sorted(stats.items())}
    ratios = {}
    for modality in ("text", "image"):
        left = relations[f"table_to_{modality}"]
        right = relations[f"{modality}_to_table"]
        pair = {}
        for key in ("abs_mean", "std"):
            values = [float(left[key] or 0.0), float(right[key] or 0.0)]
            pair[f"{key}_ratio"] = (
                None if min(values) <= 1e-12 else max(values) / min(values)
            )
        ratios[modality] = pair
    return {"relations": relations, "opposite_edge_scale_ratios": ratios}


def _write_task_results(
    path: Path, title: str, lake: str, systems: dict[str, Any], primary: str
) -> None:
    lines = [f"# {title}: {lake}", "", f"Primary metric: `{primary}`.", ""]
    for system, payload in systems.items():
        lines.extend([f"## {system}", ""])
        configs = payload.get("configs", {})
        if configs:
            lines.extend(
                [
                    "| Config | Fused R@10 | Evidence R@10 | Coverage@10 | MRR |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            ordered = sorted(
                configs.items(),
                key=lambda item: (
                    float(item[1]["metrics"].get("recall@10", 0.0)),
                    float(
                        item[1]["metrics"].get(
                            "positive_evidence_path_coverage@10", 0.0
                        )
                    ),
                ),
                reverse=True,
            )
            for name, record in ordered:
                metrics = record["metrics"]
                mrr_key = next(key for key in metrics if key.startswith("mrr@"))
                lines.append(
                    f"| {name} | {metrics.get('recall@10', 0.0):.2%} | "
                    f"{metrics['evidence'].get('recall@10', 0.0):.2%} | "
                    f"{metrics.get('positive_evidence_path_coverage@10', 0.0):.2%} | "
                    f"{metrics[mrr_key]:.4f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_abd(
    args: argparse.Namespace,
    inputs: LakeInputs,
    examples: list[TargetExample],
    store: FeatureStore,
    student_indices: StudentANNIndices,
    raw_indices: RawEmbeddingANNIndices,
) -> None:
    task_set = set(args.tasks)
    fusion_configs = _fusion_configs()
    aggregation_configs = _aggregation_configs()
    b_records_by_system = {}
    d_records_by_system = {}
    scale_by_system = {}
    pool_hashes_by_system = {}

    for system, indices in (("student", student_indices), ("raw", raw_indices)):
        b_records: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        d_records: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        scale_stats = defaultdict(RunningStats)
        pool_hashes = {k: hashlib.sha256() for k in RECALL_KS}
        needed_ks = RECALL_KS if "B" in task_set else (10,)
        for k in needed_ks:
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
                    pool_hashes[k].update(b"\0")
                    for target_id in sorted(paths):
                        pool_hashes[k].update(target_id.encode("utf-8"))
                        pool_hashes[k].update(b"\0")
                    if k == 10 and "A" in task_set:
                        _add_scale_stats(scale_stats, baseline_result)

                    if "B" in task_set:
                        for config in fusion_configs:
                            fused = fuse_ranked_channels(
                                baseline_result["direct"],
                                baseline_result["evidence"],
                                rrf_k=60,
                                fusion_mode=config["fusion_mode"],
                                direct_weight=1.0,
                                evidence_weight=config["evidence_weight"],
                                score_normalization=config["score_normalization"],
                                score_temperature=config["score_temperature"],
                            )
                            variant = {
                                "fused": fused,
                                "direct": baseline_result["direct"],
                                "evidence": baseline_result["evidence"],
                            }
                            _append_values(
                                b_records,
                                config["name"],
                                k,
                                _query_values(variant, example, k),
                            )

                    if k == 10 and "D" in task_set:
                        for config in aggregation_configs:
                            variant = rank_detailed_paths(
                                paths,
                                aggregator=PathAggregator(
                                    config["aggregation"],
                                    4,
                                    temperature=config["temperature"],
                                    power=config["power"],
                                ),
                                path_edge_normalization=config[
                                    "path_edge_normalization"
                                ],
                                rrf_k=60,
                                fusion_mode="weighted_rrf",
                                direct_weight=1.0,
                                evidence_weight=0.05,
                                gated_evidence_min_paths=2,
                                gated_evidence_quantile=0.75,
                            )
                            _append_values(
                                d_records,
                                config["name"],
                                k,
                                _query_values(variant, example, k),
                            )

        pool_hashes_by_system[system] = {
            str(k): digest.hexdigest() for k, digest in pool_hashes.items()
        }
        if "A" in task_set:
            scale_by_system[system] = _scale_payload(scale_stats)
        if "B" in task_set:
            b_records_by_system[system] = _finalize_records(
                b_records,
                "weighted_rrf_e0.05",
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=args.bootstrap_seed,
            )
        if "D" in task_set:
            d_records_by_system[system] = _finalize_records(
                d_records,
                "logsumexp_edges_none",
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=args.bootstrap_seed,
            )

    common = {
        "format_version": 1,
        "lake": inputs.lake,
        "dataset": inputs.dataset,
        "queries": len(examples),
        "candidate_pool_sha256": pool_hashes_by_system,
        "candidate_pool_policy": "one fixed ANN path pool per system/query/k",
        "evidence_types": ["text", "image"],
        "student_checkpoint": str(inputs.student_checkpoint.resolve()),
        "teacher_checkpoint": str(inputs.teacher_checkpoint.resolve()),
    }
    if "A" in task_set:
        directory = args.output_root / "taskA_scale_diagnostic"
        payload = {**common, "systems": scale_by_system}
        _write_json(directory / f"{inputs.lake}.json", payload)
        lines = [f"# Task A scale diagnostic: {inputs.lake}", ""]
        for system, values in scale_by_system.items():
            lines.extend(
                [
                    f"## {system}",
                    "",
                    "| Relation | Count | Mean | Std | Mean absolute | Min | Max |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for relation, row in values["relations"].items():
                lines.append(
                    f"| {relation} | {row['count']} | {row['mean'] or 0:.6f} | "
                    f"{row['std'] or 0:.6f} | {row['abs_mean'] or 0:.6f} | "
                    f"{row['min'] or 0:.6f} | {row['max'] or 0:.6f} |"
                )
            lines.append("")
        (directory / f"{inputs.lake}_RESULTS.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    if "B" in task_set:
        directory = args.output_root / "taskB_fusion_normalization"
        payload = {
            **common,
            "baseline": "weighted_rrf_e0.05",
            "configurations": fusion_configs,
            "systems": {
                name: {"configs": configs}
                for name, configs in b_records_by_system.items()
            },
        }
        _write_json(directory / f"{inputs.lake}.json", payload)
        _write_task_results(
            directory / f"{inputs.lake}_RESULTS.md",
            "Task B fusion normalization",
            inputs.lake,
            payload["systems"],
            "fused recall@10, coverage@10, and MRR@50",
        )
    if "D" in task_set:
        directory = args.output_root / "taskD_path_aggregation"
        payload = {
            **common,
            "baseline": "logsumexp_edges_none",
            "configurations": aggregation_configs,
            "systems": {
                name: {"configs": configs}
                for name, configs in d_records_by_system.items()
            },
        }
        _write_json(directory / f"{inputs.lake}.json", payload)
        _write_task_results(
            directory / f"{inputs.lake}_RESULTS.md",
            "Task D path aggregation",
            inputs.lake,
            payload["systems"],
            "evidence/fused recall@10 and coverage@10",
        )


def run_c(
    args: argparse.Namespace,
    root: Path,
    inputs: LakeInputs,
    examples: list[TargetExample],
    store: FeatureStore,
    raw_indices: RawEmbeddingANNIndices,
    device: torch.device,
) -> None:
    teacher = load_teacher(inputs.teacher_checkpoint, device).eval()
    invalid_images = (
        root
        / "work/stage1_optimization_20260828/task7_teacher_retrain/data/invalid_teacher_images.jsonl"
    )
    teacher_store = _TeacherFeatureView(
        store,
        missing_policy="error",
        allowed_fallback_ids=(
            _load_object_ids([str(invalid_images)]) if invalid_images.is_file() else set()
        ),
    )
    score_cache: dict[tuple[str, str], float] = {}
    taus = (0.0, 0.3, 0.5, 0.7, 1.0)
    records = {tau: defaultdict(list) for tau in taus}
    pool_hashes = {tau: hashlib.sha256() for tau in taus}
    indices = {
        tau: TeacherRerankedANNIndices(
            raw_indices,
            teacher,
            teacher_store,
            device=device,
            batch_size=args.teacher_batch_size,
            alpha=tau,
            score_cache=score_cache,
        )
        for tau in taus
    }
    for start in range(0, len(examples), args.teacher_query_batch_size):
        batch = examples[start : start + args.teacher_query_batch_size]
        for tau in taus:
            detailed = retrieve_zero_one_hop_detailed_many(
                [example.query_id for example in batch],
                indices[tau],
                k=10,
                gamma=10,
                gamma_evidence=2,
                evidence_types=("text", "image"),
                evidence_aggregation="logsumexp",
                evidence_top_k=4,
                fusion_mode="weighted_rrf",
                evidence_weight=0.05,
                query_batch_size=args.teacher_query_batch_size,
            )
            for example, result in zip(batch, detailed):
                values = _query_values(result, example, 10)
                for name, value in values.items():
                    records[tau][name].append(value)
                pool_hashes[tau].update(example.query_id.encode("utf-8"))
                pool_hashes[tau].update(b"\0")
                for target_id in sorted(_path_pool(result)):
                    pool_hashes[tau].update(target_id.encode("utf-8"))
                    pool_hashes[tau].update(b"\0")

    metrics = {
        str(tau): {
            "tau": tau,
            "fused_recall@10": _mean(records[tau]["fused_recall"]),
            "direct_recall@10": _mean(records[tau]["direct_recall"]),
            "evidence_recall@10": _mean(records[tau]["evidence_recall"]),
            "coverage@10": _mean(records[tau]["coverage"]),
            "mrr@10": _mean(records[tau]["reciprocal_rank"]),
        }
        for tau in taus
    }
    teacher_delta = metrics["1.0"]["fused_recall@10"] - metrics["0.0"][
        "fused_recall@10"
    ]
    bracket = (0.5, 0.7, 1.0) if teacher_delta >= 0.03 else (0.0, 0.3, 0.5, 0.7)
    selected = max(
        bracket,
        key=lambda tau: (
            metrics[str(tau)]["fused_recall@10"],
            metrics[str(tau)]["coverage@10"],
            tau,
        ),
    )
    pool_digests = {str(tau): digest.hexdigest() for tau, digest in pool_hashes.items()}
    payload = {
        "format_version": 1,
        "lake": inputs.lake,
        "queries": len(examples),
        "selection_source": "frozen raw/Teacher edge-score ensemble on fixed raw ANN pools",
        "teacher_rerank_delta": teacher_delta,
        "bracket": list(bracket),
        "tie_break": "higher tau after fused recall@10 and coverage@10",
        "metrics_by_tau": metrics,
        "selected_tau": selected,
        "manual_r5_tau": inputs.manual_tau,
        "matches_manual_r5_tau": selected == inputs.manual_tau,
        "candidate_pool_sha256_by_tau": pool_digests,
        "same_candidate_pool": len(set(pool_digests.values())) == 1,
        "evidence_types": ["text", "image"],
        "student_proxy": "epoch-0 PCA-1024 approximation via frozen cosine raw index",
        "teacher_checkpoint": str(inputs.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": checkpoint_fingerprint(inputs.teacher_checkpoint),
        "teacher_feature_coverage": teacher_store.coverage(),
    }
    directory = args.output_root / "taskC_adaptive_tau"
    _write_json(directory / f"{inputs.lake}.json", payload)
    lines = [
        f"# Task C adaptive tau: {inputs.lake}",
        "",
        "| Tau | Fused R@10 | Direct R@10 | Evidence R@10 | Coverage@10 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for tau in taus:
        row = metrics[str(tau)]
        lines.append(
            f"| {tau:g} | {row['fused_recall@10']:.2%} | "
            f"{row['direct_recall@10']:.2%} | {row['evidence_recall@10']:.2%} | "
            f"{row['coverage@10']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Teacher delta: {teacher_delta:.2%}; bracket: {list(bracket)}.",
            f"Selected tau: {selected:g}; r5 manual tau: {inputs.manual_tau:g}.",
            f"Fixed candidate pool invariant: {payload['same_candidate_pool']}.",
            "",
        ]
    )
    (directory / f"{inputs.lake}_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    inputs = _lake_inputs(root, args.lake)
    device = torch.device(args.device)
    examples = _examples(inputs.dev_data)
    features = root / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    store = FeatureStore.from_path(features, cache_size=args.feature_cache_size)
    student_selection = load_stage1_selection(inputs.student_selection)
    checkpoint_sha256 = checkpoint_fingerprint(inputs.student_checkpoint)
    corpus_sha256 = checkpoint_fingerprint(inputs.corpus)
    if checkpoint_sha256 != student_selection["best_checkpoint_sha256"]:
        raise ValueError("r5 Student selection/checkpoint fingerprint mismatch")
    student = load_student(inputs.student_checkpoint, device).eval()
    student_indices = StudentANNIndices(
        student,
        store,
        Path(student_selection["best_index"]),
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )
    raw_indices = load_or_build_raw_embedding_indices(
        store,
        load_corpus_ids(inputs.corpus, store),
        inputs.raw_index,
        corpus_sha256=corpus_sha256,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if set(args.tasks) & {"A", "B", "D"}:
        run_abd(args, inputs, examples, store, student_indices, raw_indices)
    if "C" in set(args.tasks):
        del student_indices, student
        if device.type == "cuda":
            torch.cuda.empty_cache()
        run_c(args, root, inputs, examples, store, raw_indices, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", required=True, choices=["entitables", "wdc"])
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r6_20260830"),
    )
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--teacher-query-batch-size", type=int, default=4)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--feature-cache-size", type=int, default=16_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    values = parser.parse_args()
    if min(
        values.query_batch_size,
        values.teacher_query_batch_size,
        values.teacher_batch_size,
        values.bootstrap_iterations,
    ) <= 0:
        parser.error("batch sizes and bootstrap iterations must be positive")
    return values


if __name__ == "__main__":
    run(parse_args())
