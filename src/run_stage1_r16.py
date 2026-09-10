"""Run the frozen R16 Teacher reranking experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student, load_teacher
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import StudentANNIndices, fuse_ranked_channels
from mmdd_stage1.row_support import load_evidence_content_keys
from run_stage1_r11_task_e import select_evidence


def _paths(root: Path) -> dict[str, Path]:
    r13 = root / "work/stage1_optimization_r13_20260909"
    b13 = (
        r13
        / "taskD_witness_supervision/p_s_target_only/evaluation_step178"
    )
    return {
        "plan": root
        / "mmdd_r15_review/stage1_optimization_r16_teacher_rerank_plan_20260910_revised.md",
        "student": r13
        / "taskD_witness_supervision/p_s_target_only/checkpoints/step_000178.pt",
        "pool": b13 / "path_pool.jsonl.gz",
        "rankings": b13 / "rankings.jsonl.gz",
        "metrics": b13 / "metrics.json",
        "index": b13 / "index",
        "provenance": root
        / "work/stage1_optimization_r15_20260909/stageC_candidate_delivery/candidate_provenance.jsonl.gz",
        "teacher": root
        / "work/stage1_optimization_r11_20260908/taskC_clean/teacher/teacher_edge.pt",
        "teacher_selection": root
        / "work/stage1_optimization_r11_20260908/taskC_clean/teacher/teacher_edge.pt.selection.json",
        "teacher_manifest": root
        / "work/stage1_optimization_r11_20260908/taskC_clean/teacher/manifest.json",
        "teacher_extra": root
        / "work/stage1_optimization_r12_20260908/taskC_training/teacher_extra",
        "teacher_extra_matched_0": root
        / "work/stage1_optimization_r16_20260910/teacher_extra_matched_gpu0",
        "teacher_extra_matched_1": root
        / "work/stage1_optimization_r16_20260910/teacher_extra_matched_gpu1",
        "teacher_extra_edges_0": root
        / "work/stage1_optimization_r16_20260910/teacher_extra_edges_gpu0",
        "teacher_extra_edges_1": root
        / "work/stage1_optimization_r16_20260910/teacher_extra_edges_gpu1",
        "features": root
        / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
        "exact_new_cases": root
        / "mmdd_r15_review/B13_80_exact_new_known_witness_cases.csv",
        "fixed_82_cases": root / "B13_evidence_only_known_witness_cases.csv",
        "content_keys": root
        / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl",
    }


def _output(root: Path) -> Path:
    return root / "work/stage1_optimization_r16_20260910"


def _teacher_feature_paths(paths: dict[str, Path]) -> list[Path]:
    return [
        path
        for name, path in paths.items()
        if name.startswith("teacher_extra")
        and (path / "teacher_manifest.jsonl").is_file()
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl_gz(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty list")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_group_bootstrap(
    rows: list[dict[str, Any]],
    left_key: str,
    right_key: str,
    *,
    iterations: int = 10_000,
    seed: int = 160910,
) -> dict[str, Any]:
    """Bootstrap a query-macro delta by resampling source groups."""

    grouped: dict[str, list[float]] = {}
    for row in rows:
        delta = float(row[left_key]) - float(row[right_key])
        grouped.setdefault(str(row["source_table_id"]), []).append(delta)
    groups = sorted(grouped)
    sums = [sum(grouped[group]) for group in groups]
    counts = [len(grouped[group]) for group in groups]
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        selected = [rng.randrange(len(groups)) for _group in groups]
        draws.append(
            sum(sums[index] for index in selected)
            / sum(counts[index] for index in selected)
        )
    observed = statistics.fmean(
        float(row[left_key]) - float(row[right_key]) for row in rows
    )
    return {
        "unit": "source_table_id",
        "source_groups": len(groups),
        "queries": len(rows),
        "iterations": iterations,
        "seed": seed,
        "observed_delta": observed,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
    }


def _frozen_plan(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    output = _output(root)
    output.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        if name == "index":
            if not (path / "manifest.json").is_file():
                raise FileNotFoundError(path / "manifest.json")
        elif name.startswith("teacher_extra"):
            if not (path / "teacher_manifest.jsonl").is_file():
                raise FileNotFoundError(path / "teacher_manifest.jsonl")
        elif name != "features" and not path.is_file():
            raise FileNotFoundError(path)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    selection = json.loads(paths["teacher_selection"].read_text(encoding="utf-8"))
    student_sha = checkpoint_fingerprint(paths["student"])
    teacher_sha = checkpoint_fingerprint(paths["teacher"])
    if metrics["checkpoint_sha256"] != student_sha:
        raise ValueError("B13 checkpoint differs from its evaluation manifest")
    if metrics["path_pool"]["sha256"] != checkpoint_fingerprint(paths["pool"]):
        raise ValueError("B13 path pool differs from its evaluation manifest")
    if metrics["rankings"]["sha256"] != checkpoint_fingerprint(paths["rankings"]):
        raise ValueError("B13 rankings differ from its evaluation manifest")
    if selection["best_checkpoint_sha256"] != teacher_sha:
        raise ValueError("Teacher checkpoint differs from its selection manifest")
    plan = {
        "format_version": 1,
        "status": "frozen",
        "frozen_at_utc": _now(),
        "protocol": {
            "student_arm": "B13_S_full_seed13/p_s_target_only@178",
            "teacher_arm": "R12 recorded source-clean edge Teacher",
            "score_space": "raw_logit",
            "primary_metric": "full_dev_query_macro_target_recall@10",
            "delivery_n": 50,
            "final_k": 10,
            "matched_budget": "M_q = size of frozen natural union",
            "weighted_rrf_tuning": False,
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(
                    path / "manifest.json"
                    if name == "index"
                    else path / "teacher_manifest.jsonl"
                    if name.startswith("teacher_extra")
                    else path
                ),
            }
            for name, path in paths.items()
            if name != "features"
        },
        "features": {
            "path": str(paths["features"].resolve()),
            "manifest_sha256": checkpoint_fingerprint(
                paths["features"] / "manifest.jsonl"
            ),
        },
    }
    write_json(output / "PLAN_FROZEN.json", plan)
    return plan


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    plan = _frozen_plan(args.root)
    paths = _paths(args.root)
    output = _output(args.root)
    candidate_path = output / "candidate_pools.jsonl.gz"

    provenance = {
        str(row["query_id"]): row for row in _read_jsonl_gz(paths["provenance"])
    }
    rankings = {
        str(row["query_id"]): row for row in _read_jsonl_gz(paths["rankings"])
    }
    pool_rows = list(_read_jsonl_gz(paths["pool"]))
    if len(pool_rows) != 1198 or len(provenance) != 1198 or len(rankings) != 1198:
        raise ValueError("R16 expects the frozen 1,198-query B13 dev set")

    device = torch.device(args.device)
    student = load_student(paths["student"], device).eval()
    store = FeatureStore.from_path(paths["features"], cache_size=2048)
    indices = StudentANNIndices(
        student,
        store,
        paths["index"],
        device=device,
        checkpoint_sha256=plan["inputs"]["student"]["sha256"],
        destination_types=("table",),
        score_space="raw_logit",
    )
    query_ids = [str(row["query_id"]) for row in pool_rows]
    union_sizes = [len(row["paths_by_target"]) for row in pool_rows]
    max_size = max(union_sizes)
    matched_hits = indices.search_many(query_ids, "table", max_size)

    feature_ids = set(store.object_ids())
    missing_features: set[str] = set()
    records = []
    natural_pairs = 0
    matched_pairs = 0
    unique_scoring_pairs = 0
    for pool_row, matched, union_size in zip(pool_rows, matched_hits, union_sizes):
        query_id = str(pool_row["query_id"])
        source = provenance[query_id]
        natural = [str(value) for value in source["U"]]
        if set(natural) != set(pool_row["paths_by_target"]):
            raise ValueError(f"{query_id}: R15 U differs from the B13 raw pool")
        if len(natural) != union_size:
            raise ValueError(f"{query_id}: B13 union contains duplicates")
        matched_ids = [str(value[0]) for value in matched[:union_size]]
        if len(matched_ids) != union_size or len(set(matched_ids)) != union_size:
            raise ValueError(f"{query_id}: invalid matched-direct ANN result")
        combined = sorted(set(natural) | set(matched_ids))
        missing_features.update(
            value for value in [query_id, *combined] if value not in feature_ids
        )
        ranking = rankings[query_id]
        records.append(
            {
                "query_id": query_id,
                "source_table_id": str(source["source_table_id"]),
                "query_kind": str(source["query_kind"]),
                "positive_target_ids": [str(value) for value in source["positive_target_ids"]],
                "natural_candidate_ids": natural,
                "matched_direct_candidate_ids": matched_ids,
                "combined_score_candidate_ids": combined,
                "ann_direct100_ids": [str(value) for value in source["D100_ANN"]],
                "exact_direct100_ids": [str(value) for value in source["D100_exact"]],
                "evidence_candidate_ids": [str(value) for value in source["E"]],
                "student_b0": {
                    key: ranking["rankings"]["f1_union_direct"][str(key)]["target_ids"]
                    for key in (10, 20, 50)
                },
            }
        )
        natural_pairs += len(natural)
        matched_pairs += len(matched_ids)
        unique_scoring_pairs += len(combined)
    if missing_features:
        raise ValueError(
            f"R16 candidate pools have {len(missing_features)} objects without features"
        )
    _write_jsonl_gz(candidate_path, records)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "queries": len(records),
        "source_groups": len({row["source_table_id"] for row in records}),
        "known_positive_pairs": sum(len(row["positive_target_ids"]) for row in records),
        "natural_pairs": natural_pairs,
        "matched_direct_pairs": matched_pairs,
        "unique_qt_pairs_to_score": unique_scoring_pairs,
        "union_size": {
            "mean": statistics.fmean(union_sizes),
            "min": min(union_sizes),
            "max": max(union_sizes),
        },
        "candidate_pools": str(candidate_path.resolve()),
        "candidate_pools_sha256": checkpoint_fingerprint(candidate_path),
        "object_feature_coverage": 1.0,
        "matched_direct_kind": "actual B13 HNSW ANN with query-specific M_q",
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "completed_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "PREPARE.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def _device_feature(
    store: FeatureStore,
    object_id: str,
    device: torch.device,
    dtype: torch.dtype,
):
    return store.get(object_id, include_hidden=True).for_scoring(
        device, include_hidden=True, hidden_dtype=dtype
    )


@torch.inference_mode()
def preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    plan = _frozen_plan(args.root)
    paths = _paths(args.root)
    output = _output(args.root)
    device = torch.device(args.device)
    teacher = load_teacher(paths["teacher"], device).eval()
    teacher.set_compute_dtype(None)
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=64,
        teacher_paths=_teacher_feature_paths(paths),
    )
    pool = next(iter(_read_jsonl_gz(paths["pool"])))
    query_id = str(pool["query_id"])
    candidate_ids = list(pool["paths_by_target"])[:4]
    dtype = next(teacher.parameters()).dtype
    query = _device_feature(store, query_id, device, dtype)
    candidates = [_device_feature(store, value, device, dtype) for value in candidate_ids]

    batched = teacher.score_pairs([query] * len(candidates), candidates)
    unbatched = torch.stack(
        [teacher.score_pairs([query], [candidate])[0] for candidate in candidates]
    )
    reverse_order = [3, 1, 0, 2]
    composition = teacher.score_pairs(
        [query] * len(candidates), [candidates[index] for index in reverse_order]
    )
    restored = torch.empty_like(composition)
    for position, original in enumerate(reverse_order):
        restored[original] = composition[position]
    compressed: dict[str, torch.Tensor] = {}
    teacher.compress_many([query, *candidates], compressed)
    direct_compressed = teacher.score_compressed_pairs(
        [compressed[query_id]] * len(candidates),
        [query.object_type] * len(candidates),
        [compressed[value] for value in candidate_ids],
        [candidate.object_type for candidate in candidates],
    )
    dropout_modules = [
        module.training
        for module in teacher.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    max_batch_difference = float((batched - unbatched).abs().max())
    max_composition_difference = float((batched - restored).abs().max())
    max_compressed_difference = float((batched - direct_compressed).abs().max())
    prepared_path = output / "candidate_pools.jsonl.gz"
    required_ids = {query_id, *candidate_ids}
    if prepared_path.is_file():
        required_ids = {
            value
            for row in _read_jsonl_gz(prepared_path)
            for value in (
                str(row["query_id"]),
                *[str(item) for item in row["combined_score_candidate_ids"]],
            )
        }
    missing_hidden = []
    for object_id in sorted(required_ids):
        features = store.get(object_id, include_hidden=True)
        if features.hidden_states is None:
            missing_hidden.append(object_id)
    passed = (
        not teacher.training
        and not any(dropout_modules)
        and bool(torch.isfinite(batched).all())
        and max_batch_difference <= 2e-5
        and max_composition_difference <= 2e-5
        and max_compressed_difference <= 2e-5
        and not missing_hidden
    )
    payload = {
        "format_version": 1,
        "status": "pass" if passed else "fail",
        "teacher_checkpoint_sha256": plan["inputs"]["teacher"]["sha256"],
        "teacher_config": teacher.config(),
        "parameter_count": sum(parameter.numel() for parameter in teacher.parameters()),
        "model_eval": not teacher.training,
        "dropout_modules": len(dropout_modules),
        "dropout_training_modules": sum(dropout_modules),
        "score_space": "raw_logit",
        "compute_dtype": str(dtype).removeprefix("torch."),
        "probe_pairs": len(candidates),
        "finite_scores": bool(torch.isfinite(batched).all()),
        "max_batched_unbatched_abs_difference": max_batch_difference,
        "max_batch_composition_abs_difference": max_composition_difference,
        "max_compress_interface_abs_difference": max_compressed_difference,
        "source_type": query.object_type,
        "destination_types": [candidate.object_type for candidate in candidates],
        "required_teacher_objects": len(required_ids),
        "teacher_hidden_state_coverage": 1.0 - len(missing_hidden) / len(required_ids),
        "missing_teacher_hidden_state_ids": missing_hidden,
        "teacher_feature_manifest_sha256s": {
            name: plan["inputs"][name]["sha256"]
            for name in plan["inputs"]
            if name.startswith("teacher_extra")
        },
        "device": args.device,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "P0_PREFLIGHT.json", payload)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise RuntimeError("R16 Teacher preflight failed")
    return payload


@torch.inference_mode()
def score_qt(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    started = time.monotonic()
    paths = _paths(args.root)
    output = _output(args.root)
    prepare_manifest = json.loads((output / "PREPARE.json").read_text(encoding="utf-8"))
    candidate_path = Path(prepare_manifest["candidate_pools"])
    if checkpoint_fingerprint(candidate_path) != prepare_manifest["candidate_pools_sha256"]:
        raise ValueError("Prepared R16 candidate pools changed")
    shard_dir = output / "qt_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    result_path = shard_dir / f"scores_{args.shard_index:03d}_of_{args.num_shards:03d}.jsonl.gz"
    manifest_path = result_path.with_suffix(".manifest.json")

    device = torch.device(args.device)
    teacher = load_teacher(paths["teacher"], device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    teacher.set_compute_dtype(None)
    dtype = next(teacher.parameters()).dtype
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=args.feature_cache_size,
        teacher_paths=_teacher_feature_paths(paths),
    )
    compression_cache: dict[str, torch.Tensor] = {}
    selected = [
        row
        for index, row in enumerate(_read_jsonl_gz(candidate_path))
        if index % args.num_shards == args.shard_index
    ]
    latencies = []
    pair_count = 0

    def scored_records() -> Iterable[dict[str, Any]]:
        nonlocal pair_count
        for position, row in enumerate(selected, 1):
            query_started = time.perf_counter()
            query_id = str(row["query_id"])
            candidate_ids = [str(value) for value in row["combined_score_candidate_ids"]]
            query = _device_feature(store, query_id, device, dtype)
            scores = []
            for start in range(0, len(candidate_ids), args.batch_size):
                batch_ids = candidate_ids[start : start + args.batch_size]
                candidates = [
                    _device_feature(store, value, device, dtype) for value in batch_ids
                ]
                values = teacher.score_pairs(
                    [query] * len(candidates),
                    candidates,
                    compression_cache=compression_cache,
                )
                scores.extend(float(value) for value in values.cpu())
            if len(scores) != len(candidate_ids) or not all(math.isfinite(v) for v in scores):
                raise ValueError(f"{query_id}: invalid Teacher-QT scores")
            latency = time.perf_counter() - query_started
            latencies.append(latency)
            pair_count += len(candidate_ids)
            if position % 25 == 0 or position == len(selected):
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "queries": position,
                            "pairs": pair_count,
                            "compressed_objects": len(compression_cache),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            yield {
                "query_id": query_id,
                "candidate_ids": candidate_ids,
                "teacher_scores": scores,
                "latency_seconds": latency,
            }

    _write_jsonl_gz(result_path, scored_records())
    ordered_latencies = sorted(latencies)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "queries": len(selected),
        "pairs": pair_count,
        "teacher_checkpoint_sha256": checkpoint_fingerprint(paths["teacher"]),
        "candidate_pools_sha256": prepare_manifest["candidate_pools_sha256"],
        "score_space": "raw_logit",
        "compute_dtype": "float32",
        "batch_size": args.batch_size,
        "feature_cache_size": args.feature_cache_size,
        "compressed_objects": len(compression_cache),
        "latency_seconds_p50": statistics.median(ordered_latencies),
        "latency_seconds_p95": _percentile(ordered_latencies, 0.95),
        "pairs_per_second": pair_count / (time.monotonic() - started),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "result": str(result_path.resolve()),
        "result_sha256": checkpoint_fingerprint(result_path),
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "completed_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def _recall(positive_ids: list[str], ranking: list[str], k: int) -> float:
    return len(set(positive_ids) & set(ranking[:k])) / len(positive_ids)


def _summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("all", "implicit", "explicit"):
        selected = rows if kind == "all" else [row for row in rows if row["query_kind"] == kind]
        result[kind] = {
            "queries": len(selected),
            "source_groups": len({row["source_table_id"] for row in selected}),
            "recall@10": statistics.fmean(row[f"{prefix}_recall@10"] for row in selected),
            "recall@20": statistics.fmean(row[f"{prefix}_recall@20"] for row in selected),
            "CandidateRecall@50": statistics.fmean(
                row[f"{prefix}_recall@50"] for row in selected
            ),
        }
    return result


def finalize_qt(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output = _output(args.root)
    prepare_manifest = json.loads((output / "PREPARE.json").read_text(encoding="utf-8"))
    candidate_path = Path(prepare_manifest["candidate_pools"])
    candidates = {str(row["query_id"]): row for row in _read_jsonl_gz(candidate_path)}
    scores: dict[str, dict[str, float]] = {}
    shard_manifests = []
    for shard_index in range(args.num_shards):
        result_path = output / "qt_shards" / f"scores_{shard_index:03d}_of_{args.num_shards:03d}.jsonl.gz"
        manifest_path = result_path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete QT shard: {manifest_path}")
        if checkpoint_fingerprint(result_path) != manifest["result_sha256"]:
            raise ValueError(f"Changed QT shard: {result_path}")
        shard_manifests.append(manifest)
        for row in _read_jsonl_gz(result_path):
            query_id = str(row["query_id"])
            if query_id in scores:
                raise ValueError(f"Duplicate QT score query: {query_id}")
            ids = [str(value) for value in row["candidate_ids"]]
            values = [float(value) for value in row["teacher_scores"]]
            if len(ids) != len(values):
                raise ValueError(f"{query_id}: QT score alignment failure")
            scores[query_id] = dict(zip(ids, values))
    if scores.keys() != candidates.keys():
        raise ValueError("QT shards do not cover the prepared queries exactly")

    per_query = []
    pair_records = []
    for query_id, row in candidates.items():
        score_map = scores[query_id]
        combined = row["combined_score_candidate_ids"]
        if set(score_map) != set(combined):
            raise ValueError(f"{query_id}: QT scores do not cover combined candidates")
        natural = [str(value) for value in row["natural_candidate_ids"]]
        matched = [str(value) for value in row["matched_direct_candidate_ids"]]
        natural_ranking = sorted(natural, key=lambda value: (-score_map[value], value))
        matched_ranking = sorted(matched, key=lambda value: (-score_map[value], value))
        positives = [str(value) for value in row["positive_target_ids"]]
        result = {
            "query_id": query_id,
            "source_table_id": row["source_table_id"],
            "query_kind": row["query_kind"],
            "positive_target_ids": positives,
            "natural_union_size": len(natural),
            "matched_direct_size": len(matched),
            "raw_union_recall": _recall(positives, natural, len(natural)),
            "matched_direct_raw_recall": _recall(positives, matched, len(matched)),
            "student_b0_top10": row["student_b0"]["10"],
            "student_b0_top20": row["student_b0"]["20"],
            "student_b0_top50": row["student_b0"]["50"],
            "teacher_natural_top10": natural_ranking[:10],
            "teacher_natural_top20": natural_ranking[:20],
            "teacher_natural_top50": natural_ranking[:50],
            "teacher_matched_top10": matched_ranking[:10],
            "teacher_matched_top20": matched_ranking[:20],
            "teacher_matched_top50": matched_ranking[:50],
            "teacher_natural_positive_ranks": {
                target_id: natural_ranking.index(target_id) + 1
                for target_id in positives
                if target_id in score_map and target_id in set(natural)
            },
            "teacher_matched_positive_ranks": {
                target_id: matched_ranking.index(target_id) + 1
                for target_id in positives
                if target_id in score_map and target_id in set(matched)
            },
        }
        for prefix, ranking in (
            ("student_b0", row["student_b0"]["50"]),
            ("teacher_natural", natural_ranking),
            ("teacher_matched", matched_ranking),
        ):
            for k in (10, 20, 50):
                result[f"{prefix}_recall@{k}"] = _recall(positives, ranking, k)
        per_query.append(result)
        pair_records.extend(
            {
                "query_id": query_id,
                "target_id": target_id,
                "teacher_qt_score": score_map[target_id],
                "in_natural_union": target_id in set(natural),
                "in_matched_direct": target_id in set(matched),
            }
            for target_id in combined
        )

    per_query_path = output / "teacher_rerank_per_query.jsonl.gz"
    pair_path = output / "teacher_pair_scores_qt.jsonl.gz"
    _write_jsonl_gz(per_query_path, per_query)
    _write_jsonl_gz(pair_path, pair_records)

    provenance = {
        str(row["query_id"]): row
        for row in _read_jsonl_gz(_paths(args.root)["provenance"])
    }
    with _paths(args.root)["exact_new_cases"].open(
        newline="", encoding="utf-8"
    ) as handle:
        fixed_80 = {
            (str(row["query_id"]), str(row["target_id"]))
            for row in csv.DictReader(handle)
        }
    with _paths(args.root)["fixed_82_cases"].open(
        newline="", encoding="utf-8"
    ) as handle:
        fixed_82 = {
            (str(row["query_id"]), str(row["target_id"]))
            for row in csv.DictReader(handle)
        }
    if len(fixed_80) != 80 or len(fixed_82) != 82:
        raise ValueError("Frozen evidence-only queues must contain 80 and 82 pairs")
    funnel = []
    candidate_provenance = []
    matched_records = []
    for row in per_query:
        query_id = row["query_id"]
        original = provenance[query_id]
        natural_top50 = set(row["teacher_natural_top50"])
        matched_top50 = set(row["teacher_matched_top50"])
        student_top10 = set(row["student_b0_top10"])
        student_top50 = set(row["student_b0_top50"])
        ann100 = set(original["D100_ANN"])
        exact100 = set(original["D100_exact"])
        evidence = set(original["E"])
        matched = set(candidates[query_id]["matched_direct_candidate_ids"])
        positive_details = {
            str(value["target_id"]): value for value in original["positive_targets"]
        }
        for target_id in row["positive_target_ids"]:
            detail = positive_details[target_id]
            funnel.append(
                {
                    "query_id": query_id,
                    "source_table_id": row["source_table_id"],
                    "query_kind": row["query_kind"],
                    "target_id": target_id,
                    "in_ann_direct100": target_id in ann100,
                    "in_exact_direct100": target_id in exact100,
                    "in_matched_direct_m": target_id in matched,
                    "introduced_by_natural_evidence": target_id in evidence and target_id not in ann100,
                    "exact_direct100_outside_natural_evidence": (
                        target_id in evidence and target_id not in exact100
                    ),
                    "in_fixed_82_ann_new_known_witness_queue": (
                        query_id, target_id
                    ) in fixed_82,
                    "in_fixed_80_exact_new_known_witness_queue": (
                        query_id, target_id
                    ) in fixed_80,
                    "teacher_qt_rank_natural": row[
                        "teacher_natural_positive_ranks"
                    ].get(target_id),
                    "teacher_qt_rank_matched": row[
                        "teacher_matched_positive_ranks"
                    ].get(target_id),
                    "teacher_kept_natural_c50": target_id in natural_top50,
                    "teacher_kept_matched_c50": target_id in matched_top50,
                    "student_b0_kept_c50": target_id in student_top50,
                    "student_b0_top10": target_id in student_top10,
                    "teacher_stage1_top10": target_id
                    in set(row["teacher_natural_top10"]),
                    "known_witness_ids": detail.get("known_witness_ids", []),
                    "selected_evidence_ids": detail.get(
                        "selected_evidence_ids", []
                    ),
                    "retained_known_witness_ids": detail.get(
                        "retained_known_witness_ids", []
                    ),
                    "known_rows_by_evidence": detail.get(
                        "known_rows_by_evidence", {}
                    ),
                    "retained_supported_rows": detail.get(
                        "retained_supported_rows", []
                    ),
                    "retained_supported_row_count": detail.get(
                        "retained_supported_row_count", 0
                    ),
                    "retention_keeps_known_witness": detail.get(
                        "retention_keeps_known_witness", False
                    ),
                    "stage2_recovery_value": None,
                    "correct_join": None,
                    "final_top10": None,
                }
            )
        candidate_provenance.append(
            {
                "query_id": query_id,
                "source_table_id": row["source_table_id"],
                "query_kind": row["query_kind"],
                "positive_target_ids": row["positive_target_ids"],
                "ann_direct100_ids": original["D100_ANN"],
                "exact_direct100_ids": original["D100_exact"],
                "evidence_candidate_ids": original["E"],
                "natural_union_ids": candidates[query_id]["natural_candidate_ids"],
                "teacher_c50_ids": row["teacher_natural_top50"],
                "teacher_top10_ids": row["teacher_natural_top10"],
            }
        )
        matched_records.append(
            {
                "query_id": query_id,
                "source_table_id": row["source_table_id"],
                "query_kind": row["query_kind"],
                "positive_target_ids": row["positive_target_ids"],
                "M_q": row["matched_direct_size"],
                "matched_direct_ids": candidates[query_id]["matched_direct_candidate_ids"],
                "teacher_c50_ids": row["teacher_matched_top50"],
                "teacher_top10_ids": row["teacher_matched_top10"],
            }
        )
    _write_jsonl_gz(output / "candidate_provenance_teacher.jsonl.gz", candidate_provenance)
    _write_jsonl_gz(output / "evidence_only_funnel_teacher.jsonl.gz", funnel)
    _write_jsonl_gz(output / "budget_matched_direct_teacher.jsonl.gz", matched_records)

    metrics = {
        "B0_student_natural": _summary(per_query, "student_b0"),
        "T0_teacher_natural": _summary(per_query, "teacher_natural"),
        "teacher_matched_direct": _summary(per_query, "teacher_matched"),
    }
    for kind in metrics["T0_teacher_natural"]:
        metrics["T0_teacher_natural"][kind]["RawUnionRecall"] = statistics.fmean(
            row["raw_union_recall"]
            for row in per_query
            if kind == "all" or row["query_kind"] == kind
        )
        metrics["teacher_matched_direct"][kind]["RawCandidateRecall"] = statistics.fmean(
            row["matched_direct_raw_recall"]
            for row in per_query
            if kind == "all" or row["query_kind"] == kind
        )
    bootstrap = {
        "teacher_natural_minus_student_b0_recall@10": paired_group_bootstrap(
            per_query, "teacher_natural_recall@10", "student_b0_recall@10"
        ),
        "teacher_natural_minus_teacher_matched_recall@10": paired_group_bootstrap(
            per_query, "teacher_natural_recall@10", "teacher_matched_recall@10"
        ),
        "teacher_natural_minus_teacher_matched_recall@50": paired_group_bootstrap(
            per_query, "teacher_natural_recall@50", "teacher_matched_recall@50"
        ),
    }
    exact_new = [
        row for row in funnel if row["exact_direct100_outside_natural_evidence"]
    ]
    queue_80 = [
        row for row in funnel if row["in_fixed_80_exact_new_known_witness_queue"]
    ]
    queue_82 = [
        row for row in funnel if row["in_fixed_82_ann_new_known_witness_queue"]
    ]
    mechanism = {
        "exact_direct100_outside_natural_evidence_positive_pairs": len(exact_new),
        "teacher_kept_in_c50": sum(row["teacher_kept_natural_c50"] for row in exact_new),
        "teacher_kept_with_retained_known_witness": sum(
            row["teacher_kept_natural_c50"] and row["retention_keeps_known_witness"]
            for row in exact_new
        ),
        "outside_matched_direct_m": sum(not row["in_matched_direct_m"] for row in exact_new),
        "outside_matched_direct_m_teacher_kept": sum(
            not row["in_matched_direct_m"] and row["teacher_kept_natural_c50"]
            for row in exact_new
        ),
        "fixed_80_exact_new_known_witness": {
            "pairs": len(queue_80),
            "teacher_kept_in_c50": sum(
                row["teacher_kept_natural_c50"] for row in queue_80
            ),
            "outside_matched_direct_m": sum(
                not row["in_matched_direct_m"] for row in queue_80
            ),
            "outside_matched_direct_m_teacher_kept": sum(
                not row["in_matched_direct_m"] and row["teacher_kept_natural_c50"]
                for row in queue_80
            ),
        },
        "fixed_82_ann_new_known_witness": {
            "pairs": len(queue_82),
            "teacher_kept_in_c50": sum(
                row["teacher_kept_natural_c50"] for row in queue_82
            ),
        },
    }
    latency = {
        "shards": shard_manifests,
        "total_pairs": sum(int(row["pairs"]) for row in shard_manifests),
        "wall_seconds_upper_bound": max(float(row["elapsed_seconds"]) for row in shard_manifests),
        "aggregate_pairs_per_second": sum(int(row["pairs"]) for row in shard_manifests)
        / max(float(row["elapsed_seconds"]) for row in shard_manifests),
        "peak_gpu_memory_bytes_by_shard": [
            int(row["peak_gpu_memory_bytes"]) for row in shard_manifests
        ],
    }
    write_json(output / "teacher_latency.json", latency)
    cache_manifest = {
        "format_version": 1,
        "cache_kind": "per-shard in-memory compressed Teacher object tokens",
        "persistent_compressed_cache": False,
        "feature_version": json.loads(
            (_output(args.root) / "PLAN_FROZEN.json").read_text(encoding="utf-8")
        )["features"]["manifest_sha256"],
        "teacher_checkpoint_sha256": shard_manifests[0]["teacher_checkpoint_sha256"],
        "compression_config": json.loads(
            (_output(args.root) / "P0_PREFLIGHT.json").read_text(encoding="utf-8")
        )["teacher_config"],
        "dtype": "float32",
        "cache_miss_policy": "load hidden states and compress; never substitute a score",
        "shards": [
            {
                "shard_index": row["shard_index"],
                "compressed_objects": row["compressed_objects"],
            }
            for row in shard_manifests
        ],
    }
    write_json(output / "teacher_cache_manifest.json", cache_manifest)
    payload = {
        "format_version": 1,
        "status": "complete",
        "metrics": metrics,
        "bootstrap": bootstrap,
        "mechanism": mechanism,
        "artifacts": {
            path.name: {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(path),
            }
            for path in (
                per_query_path,
                pair_path,
                output / "candidate_provenance_teacher.jsonl.gz",
                output / "evidence_only_funnel_teacher.jsonl.gz",
                output / "budget_matched_direct_teacher.jsonl.gz",
                output / "teacher_cache_manifest.json",
                output / "teacher_latency.json",
            )
        },
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "QT_RESULTS.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def prepare_edges(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize the frozen unique QE and ET edge requests for P2."""

    started = time.monotonic()
    paths = _paths(args.root)
    output = _output(args.root)
    qe: set[tuple[str, str]] = set()
    et: set[tuple[str, str]] = set()
    for record in _read_jsonl_gz(paths["pool"]):
        query_id = str(record["query_id"])
        for target_id, target_paths in record["paths_by_target"].items():
            for path in target_paths:
                if path.get("kind") != "evidence":
                    continue
                evidence_id = str(path["evidence_id"])
                qe.add((query_id, evidence_id))
                et.add((evidence_id, str(target_id)))
    if len(qe) != 47_920 or len(et) != 389_200:
        raise ValueError(
            f"Frozen P2 edge cardinalities changed: QE={len(qe)}, ET={len(et)}"
        )

    store = FeatureStore.from_path(
        paths["features"],
        cache_size=0,
        teacher_paths=_teacher_feature_paths(paths),
    )
    object_ids = {value for pair in [*qe, *et] for value in pair}
    missing = sorted(
        object_id
        for object_id in object_ids
        if not store.has_teacher_features(object_id)
    )
    if missing:
        for shard in range(2):
            selection_path = output / f"teacher_edge_missing_gpu{shard}.jsonl"
            with selection_path.open("w", encoding="utf-8") as handle:
                for index, object_id in enumerate(missing):
                    if index % 2 == shard:
                        handle.write(
                            json.dumps({"object_id": object_id}, ensure_ascii=False)
                            + "\n"
                        )
        write_json(
            output / "EDGE_PREPARE.json",
            {
                "format_version": 1,
                "status": "needs_teacher_features",
                "missing_teacher_objects": len(missing),
                "selection_files": [
                    str((output / f"teacher_edge_missing_gpu{shard}.jsonl").resolve())
                    for shard in range(2)
                ],
                "completed_at_utc": _now(),
            },
        )
        raise ValueError(f"P2 edges have {len(missing)} missing Teacher features")

    requests = [
        {
            "channel": "qe",
            "source_id": source_id,
            "destination_id": destination_id,
        }
        for source_id, destination_id in sorted(qe)
    ]
    requests.extend(
        {
            "channel": "et",
            "source_id": source_id,
            "destination_id": destination_id,
        }
        for source_id, destination_id in sorted(et)
    )
    request_path = output / "teacher_edge_requests.jsonl.gz"
    _write_jsonl_gz(request_path, requests)
    payload = {
        "format_version": 1,
        "status": "complete",
        "qe_unique_edges": len(qe),
        "et_unique_edges": len(et),
        "total_unique_edges": len(requests),
        "required_teacher_objects": len(object_ids),
        "teacher_hidden_state_coverage": 1.0,
        "requests": str(request_path.resolve()),
        "requests_sha256": checkpoint_fingerprint(request_path),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "EDGE_PREPARE.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


@torch.inference_mode()
def score_edges(args: argparse.Namespace) -> dict[str, Any]:
    """Score one deterministic shard of the frozen QE/ET edge requests."""

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    started = time.monotonic()
    paths = _paths(args.root)
    output = _output(args.root)
    prepared = json.loads((output / "EDGE_PREPARE.json").read_text(encoding="utf-8"))
    request_path = Path(prepared["requests"])
    if checkpoint_fingerprint(request_path) != prepared["requests_sha256"]:
        raise ValueError("Prepared Teacher edge requests changed")
    selected = [
        row
        for index, row in enumerate(_read_jsonl_gz(request_path))
        if index % args.num_shards == args.shard_index
    ]
    shard_dir = output / "edge_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    result_path = (
        shard_dir
        / f"scores_{args.shard_index:03d}_of_{args.num_shards:03d}.jsonl.gz"
    )
    manifest_path = result_path.with_suffix(".manifest.json")

    device = torch.device(args.device)
    teacher = load_teacher(paths["teacher"], device).eval()
    teacher.set_compute_dtype(None)
    torch.cuda.reset_peak_memory_stats(device)
    dtype = next(teacher.parameters()).dtype
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=0,
        teacher_paths=_teacher_feature_paths(paths),
    )
    compression_cache: dict[str, torch.Tensor] = {}
    batch_latencies: list[float] = []
    channel_counts = {"qe": 0, "et": 0}

    def scored_records() -> Iterable[dict[str, Any]]:
        for start in range(0, len(selected), args.batch_size):
            batch_started = time.perf_counter()
            batch = selected[start : start + args.batch_size]
            source_ids = [str(row["source_id"]) for row in batch]
            destination_ids = [str(row["destination_id"]) for row in batch]
            missing_ids = sorted(
                (
                    set(source_ids)
                    | set(destination_ids)
                )
                - compression_cache.keys()
            )
            if missing_ids:
                teacher.compress_many(
                    [
                        _device_feature(store, object_id, device, dtype)
                        for object_id in missing_ids
                    ],
                    compression_cache,
                )
            values = teacher.score_compressed_pairs(
                [compression_cache[object_id] for object_id in source_ids],
                [store.object_type(object_id) for object_id in source_ids],
                [compression_cache[object_id] for object_id in destination_ids],
                [store.object_type(object_id) for object_id in destination_ids],
            ).float().detach().cpu()
            if len(values) != len(batch) or not bool(torch.isfinite(values).all()):
                raise ValueError("Invalid Teacher edge score batch")
            batch_latencies.append(time.perf_counter() - batch_started)
            for row, value in zip(batch, values):
                channel = str(row["channel"])
                channel_counts[channel] += 1
                yield {**row, "teacher_score": float(value)}
            completed = start + len(batch)
            if completed % (25 * args.batch_size) == 0 or completed == len(selected):
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "edges": completed,
                            "compressed_objects": len(compression_cache),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )

    _write_jsonl_gz(result_path, scored_records())
    manifest = {
        "format_version": 1,
        "status": "complete",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "edges": len(selected),
        "channel_counts": channel_counts,
        "teacher_checkpoint_sha256": checkpoint_fingerprint(paths["teacher"]),
        "edge_requests_sha256": prepared["requests_sha256"],
        "score_space": "raw_logit",
        "compute_dtype": "float32",
        "batch_size": args.batch_size,
        "feature_cache_size": 0,
        "feature_cache_policy": "load each hidden state once, then use compressed GPU tokens",
        "compressed_objects": len(compression_cache),
        "batch_latency_seconds_p50": statistics.median(batch_latencies),
        "batch_latency_seconds_p95": _percentile(batch_latencies, 0.95),
        "edges_per_second": len(selected) / (time.monotonic() - started),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "result": str(result_path.resolve()),
        "result_sha256": checkpoint_fingerprint(result_path),
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "completed_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def _p2_channels(
    record: dict[str, Any],
    *,
    direct_scores: dict[str, float],
    store: FeatureStore,
    content_keys: dict[str, str],
    qe_scores: dict[tuple[str, str], float] | None,
    et_scores: dict[tuple[str, str], float] | None,
    support_cache: dict[str, list[float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_id = str(record["query_id"])
    if support_cache is None:
        support_cache = {}
    direct: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for target_id in sorted(str(value) for value in record["paths_by_target"]):
        paths = [dict(path) for path in record["paths_by_target"][target_id]]
        if qe_scores is not None and et_scores is not None:
            for path in paths:
                if path.get("kind") != "evidence":
                    continue
                evidence_id = str(path["evidence_id"])
                qe = qe_scores[(query_id, evidence_id)]
                et = et_scores[(evidence_id, target_id)]
                path["query_evidence_score"] = qe
                path["evidence_target_score"] = et
                path["path_score"] = qe + et
        selected, evidence_score = select_evidence(
            "e2_row_coverage",
            paths,
            query_id=query_id,
            store=store,
            content_keys=content_keys,
            top_l=20,
            budget=4,
            support_cache=support_cache,
        )
        row = {
            "target_id": target_id,
            "direct_score": direct_scores[target_id],
            "evidence_score": evidence_score,
            "selected_evidence_ids": selected,
            "paths": paths,
        }
        direct.append(row)
        if evidence_score is not None:
            evidence.append(row)
    direct.sort(key=lambda row: (-float(row["direct_score"]), str(row["target_id"])))
    evidence.sort(
        key=lambda row: (-float(row["evidence_score"]), str(row["target_id"]))
    )
    return direct, evidence


def finalize_edges(args: argparse.Namespace) -> dict[str, Any]:
    """Merge edge scores and evaluate frozen E0/E1/E2 equal-RRF diagnostics."""

    started = time.monotonic()
    paths = _paths(args.root)
    output = _output(args.root)
    prepared = json.loads((output / "EDGE_PREPARE.json").read_text(encoding="utf-8"))
    qe_scores: dict[tuple[str, str], float] = {}
    et_scores: dict[tuple[str, str], float] = {}
    shard_manifests = []
    qe_records = []
    et_records = []
    for shard_index in range(args.num_shards):
        result_path = (
            output
            / "edge_shards"
            / f"scores_{shard_index:03d}_of_{args.num_shards:03d}.jsonl.gz"
        )
        manifest = json.loads(
            result_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete Teacher edge shard: {result_path}")
        if checkpoint_fingerprint(result_path) != manifest["result_sha256"]:
            raise ValueError(f"Changed Teacher edge shard: {result_path}")
        shard_manifests.append(manifest)
        for row in _read_jsonl_gz(result_path):
            key = (str(row["source_id"]), str(row["destination_id"]))
            score = float(row["teacher_score"])
            if row["channel"] == "qe":
                if key in qe_scores:
                    raise ValueError(f"Duplicate QE edge: {key}")
                qe_scores[key] = score
                qe_records.append(row)
            else:
                if key in et_scores:
                    raise ValueError(f"Duplicate ET edge: {key}")
                et_scores[key] = score
                et_records.append(row)
    if len(qe_scores) != prepared["qe_unique_edges"] or len(et_scores) != prepared["et_unique_edges"]:
        raise ValueError("Teacher edge shards do not exactly cover P2 requests")
    qe_path = output / "teacher_pair_scores_qe.jsonl.gz"
    et_path = output / "teacher_pair_scores_et.jsonl.gz"
    _write_jsonl_gz(qe_path, sorted(qe_records, key=lambda row: (row["source_id"], row["destination_id"])))
    _write_jsonl_gz(et_path, sorted(et_records, key=lambda row: (row["source_id"], row["destination_id"])))

    qt: dict[str, dict[str, float]] = {}
    for row in _read_jsonl_gz(output / "teacher_pair_scores_qt.jsonl.gz"):
        qt.setdefault(str(row["query_id"]), {})[str(row["target_id"])] = float(
            row["teacher_qt_score"]
        )
    frozen_rankings = {
        str(row["query_id"]): row for row in _read_jsonl_gz(paths["rankings"])
    }
    source_by_query = {
        str(row["query_id"]): str(row["source_table_id"])
        for row in _read_jsonl_gz(output / "candidate_pools.jsonl.gz")
    }
    content_keys, content_keys_sha = load_evidence_content_keys(paths["content_keys"])
    store = FeatureStore.from_path(paths["features"], cache_size=args.feature_cache_size)
    per_query = []
    for position, record in enumerate(_read_jsonl_gz(paths["pool"]), 1):
        query_id = str(record["query_id"])
        positives = [str(value) for value in record["positive_target_ids"]]
        teacher_direct = qt[query_id]
        support_cache: dict[str, list[float]] = {}
        e1_direct, e1_evidence = _p2_channels(
            record,
            direct_scores=teacher_direct,
            store=store,
            content_keys=content_keys,
            qe_scores=None,
            et_scores=None,
            support_cache=support_cache,
        )
        e2_direct, e2_evidence = _p2_channels(
            record,
            direct_scores=teacher_direct,
            store=store,
            content_keys=content_keys,
            qe_scores=qe_scores,
            et_scores=et_scores,
            support_cache=support_cache,
        )
        e1 = fuse_ranked_channels(e1_direct, e1_evidence, rrf_k=60)
        e2 = fuse_ranked_channels(e2_direct, e2_evidence, rrf_k=60)
        e1_ids = [str(row["target_id"]) for row in e1]
        e2_ids = [str(row["target_id"]) for row in e2]
        e0 = frozen_rankings[query_id]["rankings"]["union_rrf_equal"]
        e0_ids = [str(value) for value in e0["50"]["target_ids"]]
        result = {
            "query_id": query_id,
            "source_table_id": source_by_query[query_id],
            "query_kind": str(record["query_kind"]),
            "positive_target_ids": positives,
            "E0_student_rrf_top10": [
                str(value) for value in e0["10"]["target_ids"]
            ],
            "E0_student_rrf_top20": [
                str(value) for value in e0["20"]["target_ids"]
            ],
            "E0_student_rrf_top50": e0_ids,
            "E1_teacher_direct_student_evidence_top10": e1_ids[:10],
            "E1_teacher_direct_student_evidence_top20": e1_ids[:20],
            "E1_teacher_direct_student_evidence_top50": e1_ids[:50],
            "E2_teacher_direct_teacher_evidence_top10": e2_ids[:10],
            "E2_teacher_direct_teacher_evidence_top20": e2_ids[:20],
            "E2_teacher_direct_teacher_evidence_top50": e2_ids[:50],
            "E2_positive_selected_evidence": {
                str(row["target_id"]): row["selected_evidence_ids"]
                for row in e2_direct
                if str(row["target_id"]) in set(positives)
            },
            "E1_positive_selected_evidence": {
                str(row["target_id"]): row["selected_evidence_ids"]
                for row in e1_direct
                if str(row["target_id"]) in set(positives)
            },
            "known_witness_ids_by_positive": {
                target_id: [
                    str(value)
                    for value in record.get("positive_evidence_by_target", {}).get(
                        target_id, []
                    )
                ]
                for target_id in positives
            },
        }
        for prefix, ranking in (("E0", e0_ids), ("E1", e1_ids), ("E2", e2_ids)):
            for k in (10, 20, 50):
                result[f"{prefix}_recall@{k}"] = _recall(positives, ranking, k)
        per_query.append(result)
        if position % 25 == 0 or position == 1198:
            print(
                json.dumps(
                    {
                        "queries": position,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    per_query_path = output / "teacher_path_aggregation_per_query.jsonl.gz"
    _write_jsonl_gz(per_query_path, per_query)
    metrics = {
        "E0_equal_rrf_student_direct_student_evidence": _summary(per_query, "E0"),
        "E1_equal_rrf_teacher_direct_student_evidence": _summary(per_query, "E1"),
        "E2_equal_rrf_teacher_direct_teacher_evidence": _summary(per_query, "E2"),
    }
    bootstrap = {
        "E1_minus_E0_recall@10": paired_group_bootstrap(
            per_query, "E1_recall@10", "E0_recall@10"
        ),
        "E2_minus_E1_recall@10": paired_group_bootstrap(
            per_query, "E2_recall@10", "E1_recall@10"
        ),
        "E2_minus_E0_recall@10": paired_group_bootstrap(
            per_query, "E2_recall@10", "E0_recall@10"
        ),
    }
    with paths["exact_new_cases"].open(newline="", encoding="utf-8") as handle:
        fixed_80 = {
            (str(row["query_id"]), str(row["target_id"]))
            for row in csv.DictReader(handle)
        }
    with paths["fixed_82_cases"].open(newline="", encoding="utf-8") as handle:
        fixed_82 = {
            (str(row["query_id"]), str(row["target_id"]))
            for row in csv.DictReader(handle)
        }
    p2_by_query = {str(row["query_id"]): row for row in per_query}

    def queue_summary(queue: set[tuple[str, str]]) -> dict[str, int]:
        values = []
        for query_id, target_id in queue:
            row = p2_by_query[query_id]
            known = set(row["known_witness_ids_by_positive"][target_id])
            e1_selected = set(row["E1_positive_selected_evidence"][target_id])
            e2_selected = set(row["E2_positive_selected_evidence"][target_id])
            values.append(
                {
                    "E1_c50": target_id
                    in set(row["E1_teacher_direct_student_evidence_top50"]),
                    "E2_c50": target_id
                    in set(row["E2_teacher_direct_teacher_evidence_top50"]),
                    "E1_known": bool(known & e1_selected),
                    "E2_known": bool(known & e2_selected),
                }
            )
        return {
            "pairs": len(values),
            "E1_kept_c50": sum(value["E1_c50"] for value in values),
            "E2_kept_c50": sum(value["E2_c50"] for value in values),
            "E1_selected_known_witness": sum(value["E1_known"] for value in values),
            "E2_selected_known_witness": sum(value["E2_known"] for value in values),
            "E1_kept_c50_and_selected_known_witness": sum(
                value["E1_c50"] and value["E1_known"] for value in values
            ),
            "E2_kept_c50_and_selected_known_witness": sum(
                value["E2_c50"] and value["E2_known"] for value in values
            ),
        }
    mechanism = {
        "fixed_80_exact_new_known_witness": queue_summary(fixed_80),
        "fixed_82_ann_new_known_witness": queue_summary(fixed_82),
    }
    payload = {
        "format_version": 1,
        "status": "complete",
        "protocol": {
            "retention": "e2_row_coverage",
            "top_l": 20,
            "evidence_budget": 4,
            "path_score_E2": "raw_H(q,e)+raw_H(e,t)",
            "fusion": "equal_rrf_k60",
            "tuning": False,
        },
        "metrics": metrics,
        "bootstrap": bootstrap,
        "mechanism": mechanism,
        "content_keys_sha256": content_keys_sha,
        "edge_shards": shard_manifests,
        "artifacts": {
            path.name: {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(path),
            }
            for path in (qe_path, et_path, per_query_path)
        },
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "P2_RESULTS.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


@torch.inference_mode()
def benchmark_qt_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Measure cold and fully warm Teacher-QT latency on the natural unions."""

    started = time.monotonic()
    paths = _paths(args.root)
    output = _output(args.root)
    candidate_path = output / "candidate_pools.jsonl.gz"
    candidates = list(_read_jsonl_gz(candidate_path))
    device = torch.device(args.device)
    teacher = load_teacher(paths["teacher"], device).eval()
    teacher.set_compute_dtype(None)
    torch.cuda.reset_peak_memory_stats(device)
    dtype = next(teacher.parameters()).dtype
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=0,
        teacher_paths=_teacher_feature_paths(paths),
    )
    compression_cache: dict[str, torch.Tensor] = {}

    def pass_once(*, allow_load: bool) -> tuple[list[float], list[float]]:
        latencies = []
        checksums = []
        for row in candidates:
            query_started = time.perf_counter()
            source_id = str(row["query_id"])
            destination_ids = [str(value) for value in row["natural_candidate_ids"]]
            all_ids = [source_id, *destination_ids]
            missing_ids = sorted(set(all_ids) - compression_cache.keys())
            if missing_ids and not allow_load:
                raise ValueError("Warm-cache pass encountered a cache miss")
            if missing_ids:
                teacher.compress_many(
                    [
                        _device_feature(store, object_id, device, dtype)
                        for object_id in missing_ids
                    ],
                    compression_cache,
                )
            values = []
            for start in range(0, len(destination_ids), args.batch_size):
                batch_ids = destination_ids[start : start + args.batch_size]
                batch_values = teacher.score_compressed_pairs(
                    [compression_cache[source_id]] * len(batch_ids),
                    [store.object_type(source_id)] * len(batch_ids),
                    [compression_cache[object_id] for object_id in batch_ids],
                    [store.object_type(object_id) for object_id in batch_ids],
                ).float()
                values.append(float(batch_values.sum().detach().cpu()))
            torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - query_started)
            checksums.append(sum(values))
        return latencies, checksums

    cold_latencies, cold_checksums = pass_once(allow_load=True)
    warm_latencies, warm_checksums = pass_once(allow_load=False)
    max_checksum_difference = max(
        abs(left - right)
        for left, right in zip(cold_checksums, warm_checksums)
    )
    payload = {
        "format_version": 1,
        "status": "complete" if max_checksum_difference == 0.0 else "fail",
        "scope": "all 1,198 natural-union queries, one GPU, sequential query order",
        "queries": len(candidates),
        "pairs_per_pass": sum(len(row["natural_candidate_ids"]) for row in candidates),
        "compressed_objects": len(compression_cache),
        "cold_cache": {
            "definition": "hidden-state file I/O plus first compression plus QT scoring",
            "total_seconds": sum(cold_latencies),
            "per_query_seconds_p50": statistics.median(cold_latencies),
            "per_query_seconds_p95": _percentile(cold_latencies, 0.95),
        },
        "warm_cache": {
            "definition": "all object tokens already resident on the same GPU",
            "total_seconds": sum(warm_latencies),
            "per_query_seconds_p50": statistics.median(warm_latencies),
            "per_query_seconds_p95": _percentile(warm_latencies, 0.95),
        },
        "max_cold_warm_query_checksum_abs_difference": max_checksum_difference,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "device": args.device,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "QT_CACHE_BENCHMARK.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "complete":
        raise RuntimeError("Cold/warm QT cache benchmark changed scores")
    return payload


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _line_count(path: Path) -> int:
    return sum(1 for _row in _read_jsonl_gz(path))


def finalize_delivery(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the frozen R16 run and write its final scientific handoff."""

    root = args.root
    paths = _paths(root)
    output = _output(root)
    qt = json.loads((output / "QT_RESULTS.json").read_text(encoding="utf-8"))
    p2 = json.loads((output / "P2_RESULTS.json").read_text(encoding="utf-8"))
    preflight = json.loads((output / "P0_PREFLIGHT.json").read_text(encoding="utf-8"))
    prepare_manifest = json.loads((output / "PREPARE.json").read_text(encoding="utf-8"))
    edge_prepare = json.loads((output / "EDGE_PREPARE.json").read_text(encoding="utf-8"))
    cache_benchmark = json.loads(
        (output / "QT_CACHE_BENCHMARK.json").read_text(encoding="utf-8")
    )
    plan = json.loads((output / "PLAN_FROZEN.json").read_text(encoding="utf-8"))
    funnel = list(_read_jsonl_gz(output / "evidence_only_funnel_teacher.jsonl.gz"))
    fixed_80 = [
        row for row in funnel if row["in_fixed_80_exact_new_known_witness_queue"]
    ]
    student_fixed_kept = sum(row["student_b0_kept_c50"] for row in fixed_80)
    teacher_fixed_kept_known = sum(
        row["teacher_kept_natural_c50"] and row["retention_keeps_known_witness"]
        for row in fixed_80
    )
    b0 = qt["metrics"]["B0_student_natural"]["all"]
    t0 = qt["metrics"]["T0_teacher_natural"]["all"]
    matched = qt["metrics"]["teacher_matched_direct"]["all"]
    e0 = p2["metrics"]["E0_equal_rrf_student_direct_student_evidence"]["all"]
    e1 = p2["metrics"]["E1_equal_rrf_teacher_direct_student_evidence"]["all"]
    e2 = p2["metrics"]["E2_equal_rrf_teacher_direct_teacher_evidence"]["all"]
    stage2_gate = {
        "candidate_recall@50_improved": t0["CandidateRecall@50"]
        > b0["CandidateRecall@50"],
        "primary_recall@10_improved": t0["recall@10"] > b0["recall@10"],
        "fixed_exact_new_retention_improved": qt["mechanism"]
        ["fixed_80_exact_new_known_witness"]["teacher_kept_in_c50"]
        > student_fixed_kept,
        "fixed_known_witness_chain_improved": teacher_fixed_kept_known
        > student_fixed_kept,
    }
    stage2_gate["triggered"] = all(stage2_gate.values())
    stage2_gate["decision"] = (
        "run_stage2" if stage2_gate["triggered"] else "do_not_run_conditional_p5"
    )
    stage2_gate["reason"] = (
        "Teacher substantially reduced both the primary R@10 and CandidateRecall@50; "
        "the plan's conditional P5 gate was not met despite retaining some exact-new cases."
    )

    cache_dirs = {
        name: path
        for name, path in paths.items()
        if name.startswith("teacher_extra")
    }
    cache_records = {}
    for name, path in cache_dirs.items():
        manifest = path / "teacher_manifest.jsonl"
        metadata = path / "metadata.json"
        cache_records[name] = {
            "path": str(path.resolve()),
            "objects": sum(1 for line in manifest.open(encoding="utf-8") if line.strip()),
            "bytes": _directory_bytes(path),
            "teacher_manifest_sha256": checkpoint_fingerprint(manifest),
            "observed_build_window_seconds": (
                max(0.0, manifest.stat().st_mtime - metadata.stat().st_mtime)
                if metadata.is_file()
                else None
            ),
            "reused_from_prior_round": name == "teacher_extra",
        }
    edge_shards = p2["edge_shards"]
    latency = json.loads((output / "teacher_latency.json").read_text(encoding="utf-8"))
    latency.update(
        {
            "qt_cache_benchmark": cache_benchmark,
            "qe_et_scoring": {
                "edges": sum(row["edges"] for row in edge_shards),
                "parallel_wall_seconds_upper_bound": max(
                    row["elapsed_seconds"] for row in edge_shards
                ),
                "batch_latency_seconds_p50_by_shard": [
                    row["batch_latency_seconds_p50"] for row in edge_shards
                ],
                "batch_latency_seconds_p95_by_shard": [
                    row["batch_latency_seconds_p95"] for row in edge_shards
                ],
                "peak_gpu_memory_bytes_by_shard": [
                    row["peak_gpu_memory_bytes"] for row in edge_shards
                ],
            },
            "offline_hidden_state_caches": cache_records,
            "measurement_note": (
                "QT cold/warm is a sequential all-dev single-GPU benchmark. Sharded "
                "throughput timings include startup and are not isolated service latency."
            ),
        }
    )
    write_json(output / "teacher_latency.json", latency)
    cache_manifest = json.loads(
        (output / "teacher_cache_manifest.json").read_text(encoding="utf-8")
    )
    cache_manifest.update(
        {
            "persistent_hidden_state_cache": True,
            "persistent_compressed_cache": False,
            "hidden_state_cache_directories": cache_records,
            "complete_qt_teacher_objects": preflight["required_teacher_objects"],
            "complete_p2_teacher_objects": edge_prepare["required_teacher_objects"],
            "teacher_hidden_state_coverage": 1.0,
        }
    )
    write_json(output / "teacher_cache_manifest.json", cache_manifest)
    for name in ("teacher_latency.json", "teacher_cache_manifest.json"):
        path = output / name
        qt["artifacts"][name] = {
            "path": str(path.resolve()),
            "sha256": checkpoint_fingerprint(path),
        }
    write_json(output / "QT_RESULTS.json", qt)

    t0_delta = qt["bootstrap"]["teacher_natural_minus_student_b0_recall@10"]
    natural_matched_r10 = qt["bootstrap"][
        "teacher_natural_minus_teacher_matched_recall@10"
    ]
    natural_matched_c50 = qt["bootstrap"][
        "teacher_natural_minus_teacher_matched_recall@50"
    ]
    e1_delta = p2["bootstrap"]["E1_minus_E0_recall@10"]
    e2_delta = p2["bootstrap"]["E2_minus_E1_recall@10"]

    def pct(value: float) -> str:
        return f"{100.0 * value:.4f}"

    results_text = f"""# R16：冻结 Teacher 候选重排与边打分实验

## 结论与实验状态

P0–P4 已按冻结协议完成；P5 是条件阶段，本轮未触发。Teacher-QT 在相同自然候选池上显著弱于 B13 Student，不能上线替代 Student；Teacher-QE/ET 替换证据边分数后又进一步恶化。证据仍然扩大了可达候选空间，但当前 pairwise Teacher 没有建立可靠的语义验证能力，因此完整的“证据发现 → Teacher 保留 → Stage2 正确值/正确 join”链条仍未成立。

## 本地事实

- 固定 1,198 个 dev query、1,000 个 source group、1,279 个 known positive pair；599 implicit、599 explicit。
- 冻结 Student 为 B13 `p_s_target_only@178`，SHA256 `{plan['inputs']['student']['sha256']}`；Teacher SHA256 `{preflight['teacher_checkpoint_sha256']}`，18,147,329 参数，eval 模式。
- 自然并集平均 289.374 个 target；QT 唯一评分 517,695 对。P2 唯一 QE 47,920 对、ET 389,200 对；39,899 个 P2 端点 Teacher hidden-state 覆盖率为 100%。
- 没有 Teacher 重训、Student 重训、weighted RRF、阈值或权重扫描，也没有把 unknown 当作负例。

## 假说与单因素干预

主假说是：冻结 Teacher 在同一自然候选池上比 Student 有更高 ranking fidelity。唯一主干预 B0→T0 只替换 QT scorer；候选池、N=50、K=10、分母和 tie-break 不变。P2 依次只替换 direct 边（E0→E1）和 evidence 边（E1→E2），继续使用 top-20、内容去重、预算 4 行覆盖与等权 RRF60。

## 主 Recall（query-macro，%）

| arm | R@10 | R@20 | CandidateRecall@50 |
|---|---:|---:|---:|
| B0 Student / natural U | {pct(b0['recall@10'])} | {pct(b0['recall@20'])} | {pct(b0['CandidateRecall@50'])} |
| T0 Teacher-QT / natural U | {pct(t0['recall@10'])} | {pct(t0['recall@20'])} | {pct(t0['CandidateRecall@50'])} |
| Teacher / matched direct M(q) | {pct(matched['recall@10'])} | {pct(matched['recall@20'])} | {pct(matched['CandidateRecall@50'])} |
| E0 Student direct + Student evidence | {pct(e0['recall@10'])} | {pct(e0['recall@20'])} | {pct(e0['CandidateRecall@50'])} |
| E1 Teacher direct + Student evidence | {pct(e1['recall@10'])} | {pct(e1['recall@20'])} | {pct(e1['CandidateRecall@50'])} |
| E2 Teacher direct + Teacher evidence | {pct(e2['recall@10'])} | {pct(e2['recall@20'])} | {pct(e2['CandidateRecall@50'])} |

T0−B0 R@10 为 {pct(t0_delta['observed_delta'])}pp，source-group bootstrap 95% CI [{pct(t0_delta['ci95'][0])}, {pct(t0_delta['ci95'][1])}]pp。E1−E0 为 {pct(e1_delta['observed_delta'])}pp，95% CI [{pct(e1_delta['ci95'][0])}, {pct(e1_delta['ci95'][1])}]pp；E2−E1 为 {pct(e2_delta['observed_delta'])}pp，95% CI [{pct(e2_delta['ci95'][0])}, {pct(e2_delta['ci95'][1])}]pp。三项均指向负向变化。

## Budget-matched 与机制诊断

自然 U 与 matched-direct M(q) 在 Teacher 下的 R@10 差为 {pct(natural_matched_r10['observed_delta'])}pp，95% CI [{pct(natural_matched_r10['ci95'][0])}, {pct(natural_matched_r10['ci95'][1])}]pp；CR50 差为 {pct(natural_matched_c50['observed_delta'])}pp，95% CI [{pct(natural_matched_c50['ci95'][0])}, {pct(natural_matched_c50['ci95'][1])}]pp。自然 evidence 候选组成在 C50 有优势，但这没有抵消 Teacher 相对 Student 的总体退化。

209 个正例位于 exact direct100 外且由自然 evidence 引入；Teacher-QT 在 C50 保留 87 个，其中 28 个同时保留已知 witness。固定 80 个 exact-new known-witness pair 中保留 31 个；43 个还位于 matched-direct M(q) 外，其中 16 个被 Teacher 保留。P2 固定 80 对里，E1 有 54 对同时进入 C50 并选中 known witness，E2 降为 43 对。

这些是 qrel/known-witness 可达性与排序诊断，不等于正确属性值或正确 join。Stage2 条件门未触发：T0 的 R@10 和 CR50 都显著/实质下降；因此 `stage2_recovery_value`、`correct_join` 与最终 Stage2 Top10 保持 unknown/null，没有以已知 witness 或 joinable flag 冒充最终验证。

## 成本与缓存

QT 全 dev 单 GPU冷缓存（hidden-state I/O + 首次压缩 + 评分）p50/p95 为 {1000 * cache_benchmark['cold_cache']['per_query_seconds_p50']:.3f}/{1000 * cache_benchmark['cold_cache']['per_query_seconds_p95']:.3f} ms；完全热 token cache 为 {1000 * cache_benchmark['warm_cache']['per_query_seconds_p50']:.3f}/{1000 * cache_benchmark['warm_cache']['per_query_seconds_p95']:.3f} ms，冷/热 query checksum 最大差为 0。QE/ET 两卡并行评分墙钟上界 {max(row['elapsed_seconds'] for row in edge_shards):.3f}s，单卡峰值显存 {max(row['peak_gpu_memory_bytes'] for row in edge_shards) / 2**30:.3f} GiB。完整 hidden-state 缓存磁盘占用见 `teacher_latency.json`；compressed token cache 只在进程内存在。

## 竞争解释

结果支持“冻结 Teacher 与自然 U 的候选分布/监督目标错位”这一解释，也可能涉及 pairwise raw-logit 的跨关系校准、Teacher 本身排序能力不足，或 Qwen hidden-state 表示与 B13 Student 检索空间不匹配。P2 的 E2 明显劣于 E1，反对“Teacher QE/ET 边分数已经可用、只差融合权重”的解释。补缓存时 Qwen 官方 wrapper 对至少两个损坏/截断图像触发其既有 NULL 视觉输入回退；这不是分数缺失回退，但属于输入数据质量限制，应在后续清洗实验中单独控制。

## 失败意味着什么

本轮不能支持“Teacher 提升 ranking fidelity”、不能部署 Teacher 在线 reranker，也不能声称 Stage2 已完成正确值/正确 join。它不否定 evidence 的候选扩展作用：RawUnionRecall 为 {pct(t0['RawUnionRecall'])}%，且存在 matched-direct 预算外被保留的 exact-new target；失败定位在当前 Teacher 的识别/保留环节。后续若继续，应先做 train-candidate distribution、监督和 listwise/calibration 诊断，而不是扫描 RRF 权重或直接启动 joint QET/新蒸馏。
"""
    (output / "RESULTS.md").write_text(results_text, encoding="utf-8")

    required_counts = {
        "teacher_rerank_per_query.jsonl.gz": 1198,
        "teacher_pair_scores_qt.jsonl.gz": prepare_manifest["unique_qt_pairs_to_score"],
        "teacher_pair_scores_qe.jsonl.gz": edge_prepare["qe_unique_edges"],
        "teacher_pair_scores_et.jsonl.gz": edge_prepare["et_unique_edges"],
        "candidate_provenance_teacher.jsonl.gz": 1198,
        "evidence_only_funnel_teacher.jsonl.gz": 1279,
        "budget_matched_direct_teacher.jsonl.gz": 1198,
    }
    count_checks = {
        name: {"expected": expected, "actual": _line_count(output / name)}
        for name, expected in required_counts.items()
    }
    validation = {
        "status": "pass",
        "p0_teacher_correctness": {
            "status": preflight["status"],
            "eval": preflight["model_eval"],
            "dropout_training_modules": preflight["dropout_training_modules"],
            "hidden_state_coverage": preflight["teacher_hidden_state_coverage"],
            "max_batched_unbatched_abs_difference": preflight[
                "max_batched_unbatched_abs_difference"
            ],
            "max_compress_interface_abs_difference": preflight[
                "max_compress_interface_abs_difference"
            ],
        },
        "frozen_denominators": {
            "queries": prepare_manifest["queries"],
            "source_groups": prepare_manifest["source_groups"],
            "known_positive_pairs": prepare_manifest["known_positive_pairs"],
            "fixed_80_pairs": len(fixed_80),
            "fixed_82_pairs": sum(
                row["in_fixed_82_ann_new_known_witness_queue"] for row in funnel
            ),
        },
        "artifact_line_counts": count_checks,
        "edge_coverage": {
            "qe": edge_prepare["qe_unique_edges"],
            "et": edge_prepare["et_unique_edges"],
            "teacher_objects": edge_prepare["required_teacher_objects"],
            "hidden_state_coverage": edge_prepare["teacher_hidden_state_coverage"],
        },
        "metric_reproduction": {
            "B13_B0_recall@10": b0["recall@10"],
            "B13_E0_recall@10": e0["recall@10"],
            "B13_E0_candidate_recall@50": e0["CandidateRecall@50"],
        },
        "matched_budget": {
            "natural_pairs": prepare_manifest["natural_pairs"],
            "matched_direct_pairs": prepare_manifest["matched_direct_pairs"],
            "equal": prepare_manifest["natural_pairs"]
            == prepare_manifest["matched_direct_pairs"],
            "retrieval": prepare_manifest["matched_direct_kind"],
        },
        "stage2_gate": stage2_gate,
        "unknown_not_negative": True,
        "weighted_rrf_or_parameter_tuning": False,
        "score_fallback_used": False,
        "input_quality_caveat": (
            "The frozen Qwen wrapper emitted NULL-input fallbacks for at least two "
            "corrupt/truncated image files during supplemental cache construction."
        ),
        "completed_at_utc": _now(),
    }
    if not all(value["expected"] == value["actual"] for value in count_checks.values()):
        validation["status"] = "fail"
    if preflight["status"] != "pass" or stage2_gate["triggered"]:
        validation["status"] = "fail"
    write_json(output / "VALIDATION.json", validation)

    required_names = [
        *required_counts,
        "teacher_cache_manifest.json",
        "teacher_latency.json",
        "RESULTS.md",
        "VALIDATION.json",
        "P0_PREFLIGHT.json",
        "QT_RESULTS.json",
        "P2_RESULTS.json",
        "QT_CACHE_BENCHMARK.json",
    ]
    audit = {
        "status": "complete_negative_result_p5_gate_not_triggered",
        "strict_plan_completion_verified": validation["status"] == "pass",
        "stages": {
            "P0": "complete_pass",
            "P1": "complete_negative",
            "P2": "complete_negative_diagnostic",
            "P3": "complete",
            "P4": "complete",
            "P5": "conditional_not_triggered",
        },
        "stage2_gate": stage2_gate,
        "scientific_claim": (
            "Evidence expands reachability, but the frozen pairwise Teacher does not "
            "reliably recognize/rank the natural candidates or evidence edges."
        ),
        "required_artifacts": {
            name: {
                "path": str((output / name).resolve()),
                "sha256": checkpoint_fingerprint(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in required_names
        },
        "completed_at_utc": _now(),
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "COMPLETION_AUDIT.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["strict_plan_completion_verified"]:
        raise RuntimeError("R16 delivery validation failed")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--device", default="cuda:0")

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--device", default="cuda:0")

    score_parser = subparsers.add_parser("score-qt")
    score_parser.add_argument("--device", required=True)
    score_parser.add_argument("--shard-index", type=int, required=True)
    score_parser.add_argument("--num-shards", type=int, default=2)
    score_parser.add_argument("--batch-size", type=int, default=512)
    score_parser.add_argument("--feature-cache-size", type=int, default=40_000)

    finalize_parser = subparsers.add_parser("finalize-qt")
    finalize_parser.add_argument("--num-shards", type=int, default=2)

    subparsers.add_parser("prepare-edges")

    edge_score_parser = subparsers.add_parser("score-edges")
    edge_score_parser.add_argument("--device", required=True)
    edge_score_parser.add_argument("--shard-index", type=int, required=True)
    edge_score_parser.add_argument("--num-shards", type=int, default=2)
    edge_score_parser.add_argument("--batch-size", type=int, default=512)
    edge_score_parser.add_argument("--feature-cache-size", type=int, default=60_000)

    edge_finalize_parser = subparsers.add_parser("finalize-edges")
    edge_finalize_parser.add_argument("--num-shards", type=int, default=2)
    edge_finalize_parser.add_argument("--feature-cache-size", type=int, default=40_000)

    cache_parser = subparsers.add_parser("benchmark-qt-cache")
    cache_parser.add_argument("--device", required=True)
    cache_parser.add_argument("--batch-size", type=int, default=512)

    subparsers.add_parser("finalize-delivery")

    args = parser.parse_args()
    args.root = args.root.resolve()
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "prepare":
        prepare(arguments)
    elif arguments.command == "preflight":
        preflight(arguments)
    elif arguments.command == "score-qt":
        score_qt(arguments)
    elif arguments.command == "finalize-qt":
        finalize_qt(arguments)
    elif arguments.command == "prepare-edges":
        prepare_edges(arguments)
    elif arguments.command == "score-edges":
        score_edges(arguments)
    elif arguments.command == "finalize-edges":
        finalize_edges(arguments)
    elif arguments.command == "benchmark-qt-cache":
        benchmark_qt_cache(arguments)
    else:
        finalize_delivery(arguments)
