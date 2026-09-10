"""Run the gated R17 Teacher correctness and frozen-checkpoint experiments."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_teacher
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from run_stage1_r16 import _read_jsonl_gz, paired_group_bootstrap


RELATIONS = (
    "table_to_table",
    "table_to_text",
    "table_to_image",
    "text_to_table",
    "image_to_table",
)
REPLAY_PER_RELATION = 100
REPLAY_SEED = 170910


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output(root: Path) -> Path:
    return root / "work/stage1_optimization_r17_20260910"


def _paths(root: Path) -> dict[str, Path]:
    r10 = root / "work/stage1_optimization_r10_20260907"
    r11 = root / "work/stage1_optimization_r11_20260908"
    r12 = root / "work/stage1_optimization_r12_20260908"
    r16 = root / "work/stage1_optimization_r16_20260910"
    return {
        "plan": root
        / "mmdd_r16_review/stage1_optimization_r17_final_20260910.md",
        "r16_plan": r16 / "PLAN_FROZEN.json",
        "candidate_pools": r16 / "candidate_pools.jsonl.gz",
        "r16_qt_scores": r16 / "teacher_pair_scores_qt.jsonl.gz",
        "r16_qe_scores": r16 / "teacher_pair_scores_qe.jsonl.gz",
        "r16_et_scores": r16 / "teacher_pair_scores_et.jsonl.gz",
        "r16_paths": r16 / "teacher_path_aggregation_per_query.jsonl.gz",
        "teacher_edge": r11 / "taskC_clean/teacher/teacher_edge.pt",
        "teacher_path": r11 / "taskC_clean/teacher/teacher_path.pt",
        "teacher_history": r11 / "taskC_clean/teacher/teacher_edge.pt.history.json",
        "teacher_manifest": r11 / "taskC_clean/teacher/manifest.json",
        "historical_score_manifest": root
        / "work/stage1_optimization_r12_20260908/taskC_training/teacher_pair_scores/manifest.json",
        "historical_scores": root
        / "work/stage1_optimization_r12_20260908/taskC_training/teacher_pair_scores/scores.pt",
        "historical_pairs": root
        / (
            "work/stage1_optimization_r12_20260908/taskC_training/"
            "candidates_seed13_steps356/teacher_pairs.jsonl.gz"
        ),
        "features": r10 / "features_qwen3_vl_embedding_8b",
        "objects": r10 / "stage1_data/stage1_objects.jsonl",
        "edge_train": r12
        / "taskA_correctness/supervision/edge_lists.train_fit.jsonl",
        "edge_dev": r12 / "taskA_correctness/supervision/edge_lists.dev.jsonl",
        "teacher_extra": root
        / "work/stage1_optimization_r12_20260908/taskC_training/teacher_extra",
        "teacher_extra_matched_0": r16 / "teacher_extra_matched_gpu0",
        "teacher_extra_matched_1": r16 / "teacher_extra_matched_gpu1",
        "teacher_extra_edges_0": r16 / "teacher_extra_edges_gpu0",
        "teacher_extra_edges_1": r16 / "teacher_extra_edges_gpu1",
        "models_code": root / "src/mmdd_stage1/models.py",
        "features_code": root / "src/mmdd_stage1/features.py",
        "checkpoints_code": root / "src/mmdd_stage1/checkpoints.py",
        "r16_code": root / "src/run_stage1_r16.py",
    }


def _teacher_feature_paths(paths: dict[str, Path]) -> list[Path]:
    return [
        path
        for name, path in paths.items()
        if name.startswith("teacher_extra")
        and (path / "teacher_manifest.jsonl").is_file()
    ]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256_text(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _relation(source_type: str, destination_type: str) -> str:
    return f"{source_type}_to_{destination_type}"


def _known_pairs(paths: dict[str, Path]) -> dict[str, set[tuple[str, str]]]:
    known: dict[str, set[tuple[str, str]]] = {name: set() for name in RELATIONS}
    for row in _read_jsonl_gz(paths["r16_paths"]):
        query_id = str(row["query_id"])
        for target_id, evidence_ids in row["known_witness_ids_by_positive"].items():
            known["table_to_table"].add((query_id, str(target_id)))
            for evidence_id in evidence_ids:
                modality = "image" if str(evidence_id).startswith("asset_img_") else "text"
                known[f"table_to_{modality}"].add((query_id, str(evidence_id)))
                known[f"{modality}_to_table"].add((str(evidence_id), str(target_id)))
    return known


def _score_rows(paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in RELATIONS}
    for row in _read_jsonl_gz(paths["r16_qt_scores"]):
        rows["table_to_table"].append(
            {
                "source_id": str(row["query_id"]),
                "destination_id": str(row["target_id"]),
                "saved_score": float(row["teacher_qt_score"]),
            }
        )
    for channel in ("qe", "et"):
        score_path = paths[f"r16_{channel}_scores"]
        for row in _read_jsonl_gz(score_path):
            source_id = str(row["source_id"])
            destination_id = str(row["destination_id"])
            evidence_id = destination_id if channel == "qe" else source_id
            modality = "image" if evidence_id.startswith("asset_img_") else "text"
            relation = (
                f"table_to_{modality}" if channel == "qe" else f"{modality}_to_table"
            )
            rows[relation].append(
                {
                    "source_id": source_id,
                    "destination_id": destination_id,
                    "saved_score": float(row["teacher_score"]),
                }
            )
    return rows


def select_replay_samples(
    rows: Sequence[dict[str, Any]],
    known_pairs: set[tuple[str, str]],
    *,
    count: int = REPLAY_PER_RELATION,
    seed: int = REPLAY_SEED,
) -> list[dict[str, Any]]:
    """Freeze a known/high/ordinary replay sample without using replay results."""

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(dict(row))
    high_keys: set[tuple[str, str]] = set()
    for source_rows in by_source.values():
        ordered = sorted(
            source_rows,
            key=lambda item: (-float(item["saved_score"]), str(item["destination_id"])),
        )
        high_keys.update(
            (str(row["source_id"]), str(row["destination_id"]))
            for row in ordered[: min(10, len(ordered))]
        )
    buckets: dict[str, list[dict[str, Any]]] = {
        "known": [],
        "high_unlabeled": [],
        "ordinary": [],
    }
    for row in rows:
        key = (str(row["source_id"]), str(row["destination_id"]))
        if key in known_pairs:
            label = "known"
        elif key in high_keys:
            label = "high_unlabeled"
        else:
            label = "ordinary"
        candidate = dict(row)
        candidate["sample_stratum"] = label
        buckets[label].append(candidate)
    rng = random.Random(seed)
    quotas = {
        "known": math.ceil(count / 3),
        "high_unlabeled": count // 3,
        "ordinary": count - math.ceil(count / 3) - count // 3,
    }
    selected: list[dict[str, Any]] = []
    for name in buckets:
        candidates = sorted(
            buckets[name],
            key=lambda row: (str(row["source_id"]), str(row["destination_id"])),
        )
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[name]])
    if len(selected) < count:
        used = {
            (str(row["source_id"]), str(row["destination_id"])) for row in selected
        }
        remainder = [
            dict(row)
            for row in rows
            if (str(row["source_id"]), str(row["destination_id"])) not in used
        ]
        rng.shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} replay rows available; expected {count}")
    return selected


def _input_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    files = {}
    for name, path in paths.items():
        manifest = (
            path / "teacher_manifest.jsonl"
            if name.startswith("teacher_extra")
            else path / "manifest.jsonl"
            if name == "features"
            else path
        )
        if manifest.is_file():
            files[name] = {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(manifest),
                "bytes": manifest.stat().st_size,
            }
    return {"format_version": 1, "inputs": files}


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args.root)
    output = _output(args.root)
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "plan",
        "r16_plan",
        "candidate_pools",
        "r16_qt_scores",
        "r16_qe_scores",
        "r16_et_scores",
        "r16_paths",
        "teacher_edge",
        "teacher_path",
        "features",
        "objects",
        "edge_train",
        "edge_dev",
    ):
        if not paths[name].exists():
            raise FileNotFoundError(paths[name])
    known = _known_pairs(paths)
    score_rows = _score_rows(paths)
    samples = []
    for relation in RELATIONS:
        relation_samples = select_replay_samples(
            score_rows[relation], known[relation], seed=REPLAY_SEED + RELATIONS.index(relation)
        )
        for sample in relation_samples:
            sample["relation"] = relation
        samples.extend(relation_samples)
    _write_jsonl_gz(output / "P0_REPLAY_SAMPLE.jsonl.gz", samples)
    historical_payload = torch.load(
        paths["historical_scores"], map_location="cpu", weights_only=True
    )
    historical_scores = historical_payload["scores"]
    historical_buckets: dict[str, list[dict[str, Any]]] = {
        relation: [] for relation in RELATIONS
    }
    for row in _read_jsonl_gz(paths["historical_pairs"]):
        relation = _relation(str(row["source_type"]), str(row["destination_type"]))
        if relation not in historical_buckets:
            continue
        bucket = historical_buckets[relation]
        if len(bucket) < REPLAY_PER_RELATION:
            pair_id = int(row["pair_id"])
            bucket.append(
                {
                    "relation": relation,
                    "source_id": str(row["source_id"]),
                    "destination_id": str(row["destination_id"]),
                    "pair_id": pair_id,
                    "saved_score": float(historical_scores[pair_id]),
                }
            )
        if all(len(values) == REPLAY_PER_RELATION for values in historical_buckets.values()):
            break
    if not all(len(values) == REPLAY_PER_RELATION for values in historical_buckets.values()):
        raise ValueError("Historical Teacher score fixture lacks a five-relation sample")
    historical_samples = [
        row for relation in RELATIONS for row in historical_buckets[relation]
    ]
    _write_jsonl_gz(output / "P0_HISTORICAL_REPLAY_SAMPLE.jsonl.gz", historical_samples)
    input_manifest = _input_manifest(paths)
    write_json(output / "INPUT_MANIFEST.json", input_manifest)
    code_manifest = {
        "format_version": 1,
        "files": {
            name: {
                "path": str(paths[name].resolve()),
                "sha256": checkpoint_fingerprint(paths[name]),
            }
            for name in ("models_code", "features_code", "checkpoints_code", "r16_code")
        },
        "r17_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": checkpoint_fingerprint(Path(__file__)),
        },
    }
    write_json(output / "CODE_HASH_MANIFEST.json", code_manifest)
    plan = {
        "format_version": 1,
        "status": "frozen",
        "frozen_at_utc": _now(),
        "protocol": {
            "plan": "R17-final",
            "stage_order": ["P0", "P1", "P2", "P3", "P4"],
            "parallel_stage": "P5",
            "replay_seed": REPLAY_SEED,
            "replay_pairs_per_relation": REPLAY_PER_RELATION,
            "replay_relations": list(RELATIONS),
            "replay_compute": "eval float32 raw logits",
            "replay_failure_threshold": 1e-4,
            "replay_target_threshold": 1e-5,
            "unknown_as_negative": False,
        },
        "teacher_edge_sha256": checkpoint_fingerprint(paths["teacher_edge"]),
        "teacher_path_sha256": checkpoint_fingerprint(paths["teacher_path"]),
        "sample_sha256": checkpoint_fingerprint(output / "P0_REPLAY_SAMPLE.jsonl.gz"),
        "historical_sample_sha256": checkpoint_fingerprint(
            output / "P0_HISTORICAL_REPLAY_SAMPLE.jsonl.gz"
        ),
        "sample_counts": {
            relation: sum(row["relation"] == relation for row in samples)
            for relation in RELATIONS
        },
    }
    write_json(output / "PLAN_FROZEN.json", plan)
    print(json.dumps(plan, indent=2))
    return plan


def _device_feature(
    store: FeatureStore, object_id: str, device: torch.device
) -> ObjectFeatures:
    return store.get(object_id, include_hidden=True).for_scoring(
        device, include_hidden=True, hidden_dtype=torch.float32
    )


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else float("nan")


def _comparison(saved: Sequence[float], replayed: Sequence[float]) -> dict[str, Any]:
    errors = [abs(a - b) for a, b in zip(saved, replayed)]
    saved_order = sorted(range(len(saved)), key=lambda i: (-saved[i], i))
    replay_order = sorted(range(len(replayed)), key=lambda i: (-replayed[i], i))
    return {
        "pairs": len(saved),
        "max_absolute_error": max(errors, default=0.0),
        "mean_absolute_error": statistics.fmean(errors) if errors else 0.0,
        "pearson": _pearson(saved, replayed),
        "spearman": _pearson(_rankdata(saved), _rankdata(replayed)),
        "full_order_consistent": saved_order == replay_order,
        "top10_order_consistent": saved_order[:10] == replay_order[:10],
        "top10_set_consistent": set(saved_order[:10]) == set(replay_order[:10]),
    }


def _score_batch(
    teacher: torch.nn.Module,
    sources: Sequence[ObjectFeatures],
    destinations: Sequence[ObjectFeatures],
    *,
    batch_size: int,
    compression_cache: dict[str, torch.Tensor] | None,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(sources), batch_size):
        values = teacher.score_pairs(
            sources[start : start + batch_size],
            destinations[start : start + batch_size],
            compression_cache=compression_cache,
        )
        scores.extend(float(value) for value in values.detach().cpu())
    return scores


@torch.inference_mode()
def replay(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    paths = _paths(args.root)
    output = _output(args.root)
    plan_path = output / "PLAN_FROZEN.json"
    if not plan_path.is_file():
        raise FileNotFoundError("Run freeze before replay")
    device = torch.device(args.device)
    teacher = load_teacher(paths["teacher_edge"], device).eval()
    teacher.set_compute_dtype(None)
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=args.feature_cache_size,
        teacher_paths=_teacher_feature_paths(paths),
    )
    samples = list(_read_jsonl_gz(output / "P0_REPLAY_SAMPLE.jsonl.gz"))
    object_ids = sorted(
        {
            str(value)
            for row in samples
            for value in (row["source_id"], row["destination_id"])
        }
    )
    features = {
        object_id: _device_feature(store, object_id, device) for object_id in object_ids
    }
    results = []
    relation_summaries = {}
    interface_relations = {}
    rng = random.Random(REPLAY_SEED)
    for relation in RELATIONS:
        relation_rows = [row for row in samples if row["relation"] == relation]
        sources = [features[str(row["source_id"])] for row in relation_rows]
        destinations = [features[str(row["destination_id"])] for row in relation_rows]
        cache: dict[str, torch.Tensor] = {}
        batched = _score_batch(
            teacher,
            sources,
            destinations,
            batch_size=args.batch_size,
            compression_cache=cache,
        )
        cached = _score_batch(
            teacher,
            sources,
            destinations,
            batch_size=args.batch_size,
            compression_cache=cache,
        )
        single = [
            _score_batch(
                teacher, [source], [destination], batch_size=1, compression_cache={}
            )[0]
            for source, destination in zip(sources, destinations)
        ]
        order = list(range(len(relation_rows)))
        rng.shuffle(order)
        shuffled_values = _score_batch(
            teacher,
            [sources[index] for index in order],
            [destinations[index] for index in order],
            batch_size=args.batch_size,
            compression_cache=cache,
        )
        shuffled = [0.0] * len(order)
        for position, original in enumerate(order):
            shuffled[original] = shuffled_values[position]
        saved = [float(row["saved_score"]) for row in relation_rows]
        for row, current, current_single, current_cached, current_shuffled in zip(
            relation_rows, batched, single, cached, shuffled
        ):
            results.append(
                {
                    **row,
                    "replayed_score": current,
                    "single_score": current_single,
                    "cached_score": current_cached,
                    "shuffled_score": current_shuffled,
                    "saved_abs_error": abs(float(row["saved_score"]) - current),
                    "batch_single_abs_error": abs(current - current_single),
                    "cached_uncached_abs_error": abs(current - current_cached),
                    "shuffle_abs_error": abs(current - current_shuffled),
                }
            )
        comparison = _comparison(saved, batched)
        comparison.update(
            {
                "max_batch_single_abs_error": max(
                    abs(a - b) for a, b in zip(batched, single)
                ),
                "max_cached_uncached_abs_error": max(
                    abs(a - b) for a, b in zip(batched, cached)
                ),
                "max_shuffle_abs_error": max(
                    abs(a - b) for a, b in zip(batched, shuffled)
                ),
            }
        )
        relation_summaries[relation] = comparison
        interface_relations[relation] = {
            key: comparison[key]
            for key in (
                "pairs",
                "max_batch_single_abs_error",
                "max_cached_uncached_abs_error",
                "max_shuffle_abs_error",
            )
        }
    _write_jsonl_gz(output / "TEACHER_REPLAY_PAIRS.jsonl.gz", results)
    max_saved_error = max(
        summary["max_absolute_error"] for summary in relation_summaries.values()
    )
    replay_status = (
        "teacher_function_replay_failed" if max_saved_error > 1e-4 else "pass"
    )
    replay_payload = {
        "format_version": 1,
        "status": replay_status,
        "target_tolerance_met": max_saved_error < 1e-5,
        "teacher_checkpoint_sha256": checkpoint_fingerprint(paths["teacher_edge"]),
        "sample_sha256": checkpoint_fingerprint(output / "P0_REPLAY_SAMPLE.jsonl.gz"),
        "score_space": "raw_logit",
        "dtype": "float32",
        "eval": not teacher.training,
        "relations": relation_summaries,
        "max_absolute_error": max_saved_error,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "TEACHER_REPLAY.json", replay_payload)
    dropout_modules = [
        module for module in teacher.modules() if isinstance(module, torch.nn.Dropout)
    ]
    interface_payload = {
        "format_version": 1,
        "status": "pass"
        if all(
            max(
                summary["max_batch_single_abs_error"],
                summary["max_cached_uncached_abs_error"],
                summary["max_shuffle_abs_error"],
            )
            <= 2e-5
            for summary in relation_summaries.values()
        )
        else "fail",
        "relations": interface_relations,
        "model_eval": not teacher.training,
        "dropout_modules": len(dropout_modules),
        "dropout_training_modules": sum(module.training for module in dropout_modules),
        "trainable_parameters": sum(
            parameter.numel() for parameter in teacher.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in teacher.parameters() if not parameter.requires_grad
        ),
        "unexpected_frozen_modules": [],
        "unexpected_trainable_modules": [],
    }
    write_json(output / "RELATION_INTERFACE_AUDIT.json", interface_payload)
    historical_samples = list(
        _read_jsonl_gz(output / "P0_HISTORICAL_REPLAY_SAMPLE.jsonl.gz")
    )
    historical_results = {}
    for relation in RELATIONS:
        relation_rows = [
            row for row in historical_samples if row["relation"] == relation
        ]
        historical_sources = [
            _device_feature(store, str(row["source_id"]), device)
            for row in relation_rows
        ]
        historical_destinations = [
            _device_feature(store, str(row["destination_id"]), device)
            for row in relation_rows
        ]
        replayed = _score_batch(
            teacher,
            historical_sources,
            historical_destinations,
            batch_size=args.batch_size,
            compression_cache={},
        )
        historical_results[relation] = _comparison(
            [float(row["saved_score"]) for row in relation_rows], replayed
        )
    historical_max_error = max(
        values["max_absolute_error"] for values in historical_results.values()
    )
    historical_payload = {
        "format_version": 1,
        "status": (
            "teacher_function_replay_failed"
            if historical_max_error > 1e-4
            else "pass"
        ),
        "target_tolerance_met": historical_max_error < 1e-5,
        "relations": historical_results,
        "max_absolute_error": historical_max_error,
        "teacher_checkpoint_sha256": checkpoint_fingerprint(paths["teacher_edge"]),
        "historical_score_manifest_sha256": checkpoint_fingerprint(
            paths["historical_score_manifest"]
        ),
        "historical_pair_manifest_sha256": checkpoint_fingerprint(
            paths["historical_pairs"]
        ),
        "historical_scores_sha256": checkpoint_fingerprint(paths["historical_scores"]),
    }
    write_json(output / "HISTORICAL_REPLAY_AUDIT.json", historical_payload)
    print(json.dumps(replay_payload, indent=2))
    if (
        replay_status != "pass"
        or interface_payload["status"] != "pass"
        or historical_payload["status"] != "pass"
    ):
        raise RuntimeError("R17 P0 score replay or relation interface audit failed")
    return replay_payload


def _feature_indices(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = {}
    for row in _read_jsonl(paths["features"] / "manifest.jsonl"):
        base[str(row["object_id"])] = row
    teacher = {}
    for directory in _teacher_feature_paths(paths):
        for row in _read_jsonl(directory / "teacher_manifest.jsonl"):
            teacher.setdefault(str(row["object_id"]), (directory, row))
    return base, teacher


def audit_lineage(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    paths = _paths(args.root)
    output = _output(args.root)
    samples = list(_read_jsonl_gz(output / "P0_REPLAY_SAMPLE.jsonl.gz"))
    object_ids = sorted(
        {
            str(value)
            for row in samples
            for value in (row["source_id"], row["destination_id"])
        }
    )
    base_index, teacher_index = _feature_indices(paths)
    store = FeatureStore.from_path(
        paths["features"], cache_size=64, teacher_paths=_teacher_feature_paths(paths)
    )
    teacher = load_teacher(paths["teacher_edge"], torch.device("cpu")).eval()
    checkpoint_sha = checkpoint_fingerprint(paths["teacher_edge"])
    feature_version = checkpoint_fingerprint(paths["features"] / "manifest.jsonl")
    token_rows = []
    lineage_rows = []
    missing = []
    for object_id in object_ids:
        if object_id not in base_index:
            missing.append(object_id)
            continue
        features = store.get(object_id, include_hidden=True).for_scoring(
            torch.device("cpu"), include_hidden=True, hidden_dtype=torch.float32
        )
        hidden = features.hidden_states
        if hidden is None:
            missing.append(object_id)
            continue
        groups = features.token_groups
        compressed = teacher.compress(features)
        group_counts = {}
        if groups is not None:
            for group in torch.unique(groups, sorted=True):
                group_counts[str(int(group))] = int((groups == group).sum())
        token_rows.append(
            {
                "object_id": object_id,
                "object_type": features.object_type,
                "raw_token_count": int(hidden.shape[0]),
                "attention_valid_token_count": int(hidden.shape[0]),
                "token_group_ids": sorted(int(value) for value in group_counts),
                "group_token_counts": group_counts,
                "schema_groups": [0] if groups is not None else None,
                "row_groups": (
                    sorted(int(value) for value in group_counts if int(value) != 0)
                    if groups is not None
                    else None
                ),
                "padding_groups": [],
                "ignored_groups": [],
                "negative_group_count": int((groups < 0).sum()) if groups is not None else 0,
                "pooled_token_count": int(compressed.shape[0]),
                "learned_latent_count": (
                    teacher.text_latents
                    if features.object_type == "text"
                    else teacher.image_latents
                    if features.object_type == "image"
                    else None
                ),
                "truncation_status": "not_recorded_in_feature_manifest",
            }
        )
        base_row = base_index[object_id]
        base_path = (paths["features"] / str(base_row["feature_path"])).resolve()
        teacher_path = None
        teacher_manifest_sha = None
        if object_id in teacher_index:
            directory, teacher_row = teacher_index[object_id]
            teacher_path = (directory / str(teacher_row["teacher_feature_path"])).resolve()
            teacher_manifest_sha = checkpoint_fingerprint(
                directory / "teacher_manifest.jsonl"
            )
        cache_payload = {
            "checkpoint_sha256": checkpoint_sha,
            "feature_version": feature_version,
            "object_id": object_id,
            "modality": features.object_type,
            "compression_config": teacher.config(),
        }
        lineage_rows.append(
            {
                "object_id": object_id,
                "modality": features.object_type,
                "raw_content_hash": base_row.get("source_fingerprint"),
                "qwen_feature_path": str(base_path),
                "qwen_feature_hash": checkpoint_fingerprint(base_path),
                "teacher_feature_path": str(teacher_path) if teacher_path else None,
                "teacher_feature_hash": (
                    checkpoint_fingerprint(teacher_path) if teacher_path else None
                ),
                "teacher_feature_manifest_sha256": teacher_manifest_sha,
                "teacher_checkpoint_hash": checkpoint_sha,
                "compressed_cache_key": _sha256_text(cache_payload),
            }
        )
    _write_jsonl_gz(output / "TOKEN_GROUP_AUDIT.jsonl.gz", token_rows)
    _write_jsonl_gz(output / "FEATURE_LINEAGE.jsonl.gz", lineage_rows)
    cache_payload = {
        "format_version": 1,
        "status": "pass" if not missing else "fail",
        "persistent_compressed_cache": False,
        "key_fields": [
            "checkpoint_sha256",
            "feature_version",
            "object_id",
            "modality",
            "compression_config",
        ],
        "checkpoint_sha256": checkpoint_sha,
        "feature_version": feature_version,
        "sample_objects": len(object_ids),
        "lineage_objects": len(lineage_rows),
        "missing_objects": missing,
    }
    write_json(output / "CACHE_LINEAGE.json", cache_payload)
    historical = json.loads(
        (output / "HISTORICAL_REPLAY_AUDIT.json").read_text(encoding="utf-8")
    )
    result = {
        "format_version": 1,
        "status": cache_payload["status"],
        "objects": len(object_ids),
        "token_rows": len(token_rows),
        "lineage_rows": len(lineage_rows),
        "negative_group_objects": sum(row["negative_group_count"] > 0 for row in token_rows),
        "historical_replay": historical["status"],
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(result, indent=2))
    if result["status"] != "pass" or result["negative_group_objects"]:
        raise RuntimeError("R17 P0 feature/cache lineage audit failed")
    return result


def audit_labels(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args.root)
    output = _output(args.root)
    counts = {
        relation: {
            "lists": 0,
            "lists_with_known_positive": 0,
            "lists_with_verified_negative": 0,
            "verified_negatives": 0,
        }
        for relation in RELATIONS
    }
    for row in _read_jsonl(paths["edge_train"]):
        relation = _relation(str(row["source_type"]), str(row["destination_type"]))
        if relation not in counts:
            continue
        labels = list(row.get("confirmed_labels") or [])
        counts[relation]["lists"] += 1
        counts[relation]["lists_with_known_positive"] += bool(row.get("positive_ids"))
        counts[relation]["lists_with_verified_negative"] += any(label == 0 for label in labels)
        counts[relation]["verified_negatives"] += sum(label == 0 for label in labels)
    missing = [
        relation
        for relation, values in counts.items()
        if values["lists_with_verified_negative"] < 16
    ]
    payload = {
        "format_version": 1,
        "status": "blocked_by_verified_negative_coverage" if missing else "ready",
        "required_lists_per_relation": 16,
        "unknown_as_negative": False,
        "label_source": str(paths["edge_train"].resolve()),
        "label_source_sha256": checkpoint_fingerprint(paths["edge_train"]),
        "label_semantics": "R12 corrected supervision",
        "relations": counts,
        "relations_below_requirement": missing,
        "consequence": (
            "Do not run the five-relation tiny overfit until verified negatives "
            "exist for every relation. P0-E remains incomplete."
            if missing
            else "The frozen tiny-overfit subset may be materialized."
        ),
    }
    write_json(output / "TINY_OVERFIT_RESULTS.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def audit_images(args: argparse.Namespace) -> dict[str, Any]:
    """Validate every frozen image input and preserve affected query IDs."""

    from PIL import Image

    paths = _paths(args.root)
    output = _output(args.root)
    image_paths = {}
    for row in _read_jsonl(paths["objects"]):
        if str(row.get("object_type")) == "image":
            image_paths[str(row["object_id"])] = Path(str(row["image"]))
    qe_queries: dict[str, set[str]] = defaultdict(set)
    for row in _read_jsonl_gz(paths["r16_qe_scores"]):
        evidence_id = str(row["destination_id"])
        if evidence_id.startswith("asset_img_"):
            qe_queries[evidence_id].add(str(row["source_id"]))
    et_images = {
        str(row["source_id"])
        for row in _read_jsonl_gz(paths["r16_et_scores"])
        if str(row["source_id"]).startswith("asset_img_")
    }
    corrupt = []
    for object_id, path in sorted(image_paths.items()):
        reason = None
        if not path.is_file():
            reason = "missing_file"
        else:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:  # PIL is the external file boundary here.
                reason = f"{type(exc).__name__}: {exc}"
        if reason is not None:
            corrupt.append(
                {
                    "object_id": object_id,
                    "path": str(path),
                    "reason": reason,
                    "affected_query_ids": sorted(qe_queries.get(object_id, set())),
                    "affects_qe": bool(qe_queries.get(object_id)),
                    "affects_et": object_id in et_images,
                }
            )
    payload = {
        "format_version": 1,
        "status": "pass" if not corrupt else "data_quality_issue",
        "image_objects": len(image_paths),
        "raw_image_validation": "PIL.Image.verify",
        "corrupt_or_missing_images": corrupt,
        "corrupt_or_missing_count": len(corrupt),
        "historical_wrapper_fallback_note": (
            "R16 reported at least two batch-to-NULL fallbacks but did not preserve "
            "their object IDs. This audit revalidates the current raw files; it cannot "
            "retrospectively attribute unlogged wrapper fallbacks when raw files now pass."
        ),
        "completed_at_utc": _now(),
    }
    write_json(output / "CORRUPT_IMAGE_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def finalize_p0(args: argparse.Namespace) -> dict[str, Any]:
    output = _output(args.root)
    required = {
        "frozen_score_replay": "TEACHER_REPLAY.json",
        "historical_score_replay": "HISTORICAL_REPLAY_AUDIT.json",
        "relation_interface": "RELATION_INTERFACE_AUDIT.json",
        "cache_lineage": "CACHE_LINEAGE.json",
        "image_quality": "CORRUPT_IMAGE_AUDIT.json",
        "tiny_overfit": "TINY_OVERFIT_RESULTS.json",
    }
    records = {}
    for name, filename in required.items():
        path = output / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = json.loads(path.read_text(encoding="utf-8"))
    replay_pass = all(
        records[name]["status"] == "pass"
        for name in (
            "frozen_score_replay",
            "historical_score_replay",
            "relation_interface",
            "cache_lineage",
        )
    )
    tiny_status = str(records["tiny_overfit"]["status"])
    overall = (
        "blocked_by_verified_negative_coverage"
        if replay_pass and tiny_status == "blocked_by_verified_negative_coverage"
        else "pass"
        if replay_pass and tiny_status == "pass"
        else "failed"
    )
    payload = {
        "format_version": 1,
        "status": overall,
        "stages": {
            "P0-A": records["frozen_score_replay"]["status"],
            "P0-B": records["historical_score_replay"]["status"],
            "P0-C": records["relation_interface"]["status"],
            "P0-D": (
                "pass_with_image_sensitivity_required"
                if records["cache_lineage"]["status"] == "pass"
                and records["image_quality"]["status"] == "data_quality_issue"
                else records["cache_lineage"]["status"]
            ),
            "P0-E": tiny_status,
        },
        "teacher_function_replay_max_error": records["frozen_score_replay"][
            "max_absolute_error"
        ],
        "historical_function_replay_max_error": records[
            "historical_score_replay"
        ]["max_absolute_error"],
        "corrupt_or_unsafe_image_count": records["image_quality"][
            "corrupt_or_missing_count"
        ],
        "teacher_method_experiments_allowed": overall == "pass",
        "P1_to_P4_action": (
            "proceed" if overall == "pass" else "paused_by_R17_P0_gate"
        ),
        "P5_action": "independent; may proceed",
        "completed_at_utc": _now(),
    }
    write_json(output / "P0_GATE.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--device", required=True)
    replay_parser.add_argument("--batch-size", type=int, default=32)
    replay_parser.add_argument("--feature-cache-size", type=int, default=4096)
    subparsers.add_parser("audit-lineage")
    subparsers.add_parser("audit-labels")
    subparsers.add_parser("audit-images")
    subparsers.add_parser("finalize-p0")
    args = parser.parse_args()
    args.root = args.root.resolve()
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "freeze":
        freeze(arguments)
    elif arguments.command == "replay":
        replay(arguments)
    elif arguments.command == "audit-lineage":
        audit_lineage(arguments)
    elif arguments.command == "audit-labels":
        audit_labels(arguments)
    elif arguments.command == "audit-images":
        audit_images(arguments)
    else:
        finalize_p0(arguments)
