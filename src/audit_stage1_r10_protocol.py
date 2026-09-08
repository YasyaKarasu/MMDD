#!/usr/bin/env python
"""Audit and freeze the Stage-1 R10 Task-A data and feature protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _artifact_records(
    dataset_root: Path, manifest: dict[str, Any], name: str
) -> Iterator[dict[str, Any]]:
    if name in manifest["artifacts"]:
        for shard in manifest["artifacts"][name]["shards"]:
            yield from _jsonl(dataset_root / shard["path"])
        return
    yield from _jsonl(dataset_root / manifest["single_files"][name])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counter(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _numeric_summary(values: Iterable[int]) -> dict[str, float | int]:
    rows = list(values)
    if not rows:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(rows),
        "min": min(rows),
        "max": max(rows),
        "mean": sum(rows) / len(rows),
    }


def _support_mode(modalities: set[str]) -> str:
    selected = modalities & {"text", "image"}
    if selected == {"text"}:
        return "text_only"
    if selected == {"image"}:
        return "image_only"
    if selected == {"text", "image"}:
        return "text_and_image"
    return "none"


def _calibration_split(
    train_groups: Iterable[str], *, seed: int, fraction: float = 0.1
) -> tuple[list[str], list[str]]:
    groups = sorted(set(train_groups))
    if not groups:
        raise ValueError("Cannot split an empty set of training source groups")
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must be between zero and one")
    random.Random(seed).shuffle(groups)
    calibration_count = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
    calibration = sorted(groups[:calibration_count])
    fit = sorted(groups[calibration_count:])
    return fit, calibration


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _queries(
    dataset_root: Path, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in _artifact_records(dataset_root, manifest, "query_tables"):
        query_id = str(row["table_id"])
        result[query_id] = {
            "split": str(row["split"]),
            "source_table_id": str(row["source_table_id"]),
            "rows": len(row.get("rows", ())),
            "query_entity_col": int(row["query_entity_col"]),
            "hidden_attributes": row.get("hidden_attributes", ()),
            "target_table_ids": tuple(str(value) for value in row["target_table_ids"]),
            "query_context_columns": len(row.get("query_context_col_names", ())),
            "row_view_index": int(row.get("row_view_index", 0)),
        }
    return result


def _targets(
    dataset_root: Path, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in _artifact_records(dataset_root, manifest, "data_lake_tables"):
        target_id = str(row["table_id"])
        result[target_id] = {
            "split": str(row.get("split", "")),
            "source_table_id": str(row.get("source_table_id", "")),
            "join_col": int(row["join_col"]) if "join_col" in row else None,
            "columns": len(row.get("columns", ())),
            "target_context_columns": len(row.get("target_context_col_names", ())),
        }
    return result


def _input_fingerprints(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    stage1_data = Path(args.stage1_data).resolve()
    features = Path(args.features).resolve()
    pca_path = Path(args.pca).resolve()
    teacher_checkpoint = Path(args.teacher_checkpoint).resolve()
    output_dir = Path(args.output_dir)

    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = json.loads((dataset_root / "stats.json").read_text(encoding="utf-8"))
    feature_metadata = json.loads(
        (features / "metadata.json").read_text(encoding="utf-8")
    )
    merge_summary = json.loads(
        (features / "merge_summary.json").read_text(encoding="utf-8")
    )
    pca = torch.load(pca_path, map_location="cpu", weights_only=True)

    missing_artifact_files = [
        shard["path"]
        for artifact in manifest["artifacts"].values()
        for shard in artifact["shards"]
        if not (dataset_root / shard["path"]).is_file()
    ]
    missing_single_files = [
        relative
        for relative in manifest["single_files"].values()
        if not (dataset_root / relative).is_file()
    ]
    if missing_artifact_files or missing_single_files:
        raise FileNotFoundError(
            f"Dataset manifest has missing files: {missing_artifact_files + missing_single_files}"
        )

    queries = _queries(dataset_root, manifest)
    targets = _targets(dataset_root, manifest)
    qrels = list(_artifact_records(dataset_root, manifest, "qrels"))
    recoveries = list(
        _artifact_records(dataset_root, manifest, "evidence_recoveries")
    )
    target_lists = list(_jsonl(stage1_data / "target_lists.jsonl"))
    edge_lists = list(_jsonl(stage1_data / "edge_lists.jsonl"))

    expected_counts = {
        "query_tables": manifest["artifacts"]["query_tables"]["total_records"],
        "data_lake_tables": manifest["artifacts"]["data_lake_tables"]["total_records"],
        "evidence_recoveries": manifest["artifacts"]["evidence_recoveries"][
            "total_records"
        ],
        "qrels": int(stats["qrels"]),
    }
    observed_counts = {
        "query_tables": len(queries),
        "data_lake_tables": len(targets),
        "evidence_recoveries": len(recoveries),
        "qrels": len(qrels),
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Dataset record counts differ: observed={observed_counts}, expected={expected_counts}"
        )
    if merge_summary.get("status") != "pass":
        raise ValueError("Feature merge did not pass")

    qrel_targets: dict[str, set[str]] = defaultdict(set)
    qrel_reasons: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        query_id = str(row["query_table_id"])
        target_id = str(row["target_table_id"])
        if query_id not in queries or target_id not in targets:
            raise ValueError(f"Qrel references a missing object: {query_id} -> {target_id}")
        if str(row["split"]) != queries[query_id]["split"]:
            raise ValueError(f"Qrel split differs from query split: {query_id}")
        qrel_targets[query_id].add(target_id)
        qrel_reasons[query_id].add(str(row.get("reason", "unknown")))

    recovery_rows: dict[tuple[str, str], set[int]] = defaultdict(set)
    recovery_modalities: dict[tuple[str, str], set[str]] = defaultdict(set)
    recovery_evidence: dict[str, set[str]] = defaultdict(set)
    recovery_edges: set[tuple[str, str]] = set()
    for row in recoveries:
        query_id = str(row["query_table_id"])
        target_id = str(row["target_table_id"])
        evidence = row.get("evidence", {})
        evidence_id = str(evidence.get("asset_id", ""))
        modality = str(evidence.get("asset_type", ""))
        recovery_rows[(query_id, target_id)].add(int(row["query_row_id"]))
        recovery_modalities[(query_id, target_id)].add(modality)
        recovery_evidence[query_id].add(evidence_id)
        recovery_edges.add((evidence_id, target_id))

    target_examples = {str(row["query_id"]): row for row in target_lists}
    missing_target_examples = sorted(set(queries) - target_examples.keys())
    extra_target_examples = sorted(target_examples.keys() - set(queries))
    positive_mismatches = []
    missing_positive_candidates = []
    for query_id, known_targets in qrel_targets.items():
        row = target_examples.get(query_id)
        if row is None:
            continue
        saved_positives = set(str(value) for value in row["positive_target_ids"])
        candidate_ids = {
            str(candidate["target_id"]) for candidate in row["candidates"]
        }
        if saved_positives != known_targets:
            positive_mismatches.append(query_id)
        if not known_targets <= candidate_ids:
            missing_positive_candidates.append(query_id)

    q_to_e_total = q_to_e_supported = 0
    e_to_t_total = e_to_t_supported = 0
    q_to_t_total = q_to_t_supported = 0
    edge_relations = Counter()
    for row in edge_lists:
        source_type = str(row.get("source_type", ""))
        destination_type = str(row.get("destination_type", ""))
        source_id = str(row["query_id"])
        positive_id = str(row["positive_id"])
        edge_relations[f"{source_type}_to_{destination_type}"] += 1
        if source_type == "table" and destination_type in {"text", "image"}:
            q_to_e_total += 1
            q_to_e_supported += positive_id in recovery_evidence[source_id]
        elif source_type in {"text", "image"} and destination_type == "table":
            e_to_t_total += 1
            e_to_t_supported += (source_id, positive_id) in recovery_edges
        elif source_type == "table" and destination_type == "table":
            q_to_t_total += 1
            q_to_t_supported += positive_id in qrel_targets[source_id]

    groups_by_split: dict[str, set[str]] = defaultdict(set)
    query_ids_by_split: dict[str, list[str]] = defaultdict(list)
    for query_id, row in queries.items():
        groups_by_split[row["split"]].add(row["source_table_id"])
        query_ids_by_split[row["split"]].append(query_id)
    split_overlaps = {
        f"{left}_{right}": sorted(groups_by_split[left] & groups_by_split[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if any(split_overlaps.values()):
        raise ValueError("Source groups overlap across dataset splits")
    train_fit_groups, calibration_groups = _calibration_split(
        groups_by_split["train"], seed=args.seed
    )
    calibration_group_set = set(calibration_groups)
    train_fit_query_ids = sorted(
        query_id
        for query_id in query_ids_by_split["train"]
        if queries[query_id]["source_table_id"] not in calibration_group_set
    )
    calibration_query_ids = sorted(
        query_id
        for query_id in query_ids_by_split["train"]
        if queries[query_id]["source_table_id"] in calibration_group_set
    )

    recovery_pair_coverages = {
        pair: len(rows) / queries[pair[0]]["rows"]
        for pair, rows in recovery_rows.items()
    }
    implicit_reason = "model_recoverable_join_column"
    query_kinds = {
        query_id: (
            "implicit"
            if reasons == {implicit_reason}
            else "explicit"
            if implicit_reason not in reasons
            else "mixed"
        )
        for query_id, reasons in qrel_reasons.items()
    }
    target_join_positions = [
        targets[str(row["target_table_id"])]["join_col"] for row in qrels
    ]

    label_audit = {
        "format_version": 1,
        "status": "pass" if not (
            missing_target_examples
            or extra_target_examples
            or positive_mismatches
            or missing_positive_candidates
        ) else "fail",
        "dataset_counts": {
            "expected": expected_counts,
            "observed": observed_counts,
        },
        "query_splits": _counter(row["split"] for row in queries.values()),
        "query_kinds": _counter(query_kinds.values()),
        "qrel_reasons": _counter(row.get("reason", "unknown") for row in qrels),
        "positive_targets_per_query": _numeric_summary(
            len(values) for values in qrel_targets.values()
        ),
        "multi_positive_queries": sum(
            len(values) > 1 for values in qrel_targets.values()
        ),
        "hidden_attributes_per_query": _numeric_summary(
            len(row["hidden_attributes"]) for row in queries.values()
        ),
        "query_entity_column_positions": _counter(
            row["query_entity_col"] for row in queries.values()
        ),
        "target_join_column_positions": _counter(target_join_positions),
        "query_context_columns": _counter(
            row["query_context_columns"] for row in queries.values()
        ),
        "target_context_columns": _counter(
            row["target_context_columns"] for row in targets.values()
        ),
        "recovery": {
            "records": len(recoveries),
            "query_target_pairs": len(recovery_rows),
            "query_row_count": _counter(len(rows) for rows in recovery_rows.values()),
            "coverage_at_least_40_percent_pairs": sum(
                value >= 0.4 for value in recovery_pair_coverages.values()
            ),
            "coverage_at_least_60_percent_pairs": sum(
                value >= 0.6 for value in recovery_pair_coverages.values()
            ),
            "full_row_coverage_pairs": sum(
                value == 1.0 for value in recovery_pair_coverages.values()
            ),
            "modality_by_query_target": _counter(
                _support_mode(values) for values in recovery_modalities.values()
            ),
        },
        "edge_lists": {
            "records": len(edge_lists),
            "relations": dict(sorted(edge_relations.items())),
            "q_to_e_recovery_supported": {
                "supported": q_to_e_supported,
                "total": q_to_e_total,
                "fraction": q_to_e_supported / q_to_e_total if q_to_e_total else 0.0,
            },
            "e_to_t_recovery_supported": {
                "supported": e_to_t_supported,
                "total": e_to_t_total,
                "fraction": e_to_t_supported / e_to_t_total if e_to_t_total else 0.0,
            },
            "q_to_t_qrel_supported": {
                "supported": q_to_t_supported,
                "total": q_to_t_total,
                "fraction": q_to_t_supported / q_to_t_total if q_to_t_total else 0.0,
            },
        },
        "stage1_target_lists": {
            "records": len(target_lists),
            "missing_queries": missing_target_examples,
            "extra_queries": extra_target_examples,
            "positive_set_mismatch_queries": positive_mismatches,
            "missing_positive_candidate_queries": missing_positive_candidates,
        },
        "source_group_split_overlap": split_overlaps,
        "threshold_interpretation": {
            "construction_min_recovered_value_ratio": float(
                manifest["query_construction"]["min_recovered_value_ratio"]
            ),
            "stage2_primary_row_coverage": 0.6,
            "stage2_secondary_construction_aligned_row_coverage": 0.4,
            "note": (
                "The 40% construction acceptance threshold and the 60% Stage-2 "
                "success criterion have different denominators and purposes."
            ),
        },
    }
    if label_audit["status"] != "pass":
        raise ValueError("Stage-1 target lists do not preserve the dataset positives")

    split_payload = {
        "format_version": 1,
        "source_group_key": "source_table_id",
        "seed": args.seed,
        "train_calibration_fraction_by_source_group": 0.1,
        "policy": (
            "Seeded shuffle of sorted train source groups; calibration is never "
            "used for P/R gradients. Dev and test retain dataset assignments."
        ),
        "source_groups": {
            "train_fit": train_fit_groups,
            "train_calibration": calibration_groups,
            "dev": sorted(groups_by_split["dev"]),
            "test": sorted(groups_by_split["test"]),
        },
        "query_ids": {
            "train_fit": train_fit_query_ids,
            "train_calibration": calibration_query_ids,
            "dev": sorted(query_ids_by_split["dev"]),
            "test": sorted(query_ids_by_split["test"]),
        },
        "counts": {
            "source_groups": {
                "train_fit": len(train_fit_groups),
                "train_calibration": len(calibration_groups),
                "dev": len(groups_by_split["dev"]),
                "test": len(groups_by_split["test"]),
            },
            "queries": {
                "train_fit": len(train_fit_query_ids),
                "train_calibration": len(calibration_query_ids),
                "dev": len(query_ids_by_split["dev"]),
                "test": len(query_ids_by_split["test"]),
            },
        },
    }

    teacher_selection_path = Path(str(teacher_checkpoint) + ".selection.json")
    teacher_selection = json.loads(
        teacher_selection_path.read_text(encoding="utf-8")
    )
    raw_index_manifest = output_dir / "baselines/raw_index/manifest.json"
    inputs = {
        "format_version": 1,
        "dataset": {
            "name": args.dataset_name,
            "root": str(dataset_root),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "artifact_counts": {
                name: int(value["total_records"])
                for name, value in manifest["artifacts"].items()
            },
            "qrels": int(stats["qrels"]),
            "query_construction": manifest["query_construction"],
        },
        "stage1": {
            "data": _input_fingerprints(
                {
                    name: stage1_data / f"{name}.jsonl"
                    for name in (
                        "stage1_objects",
                        "edge_lists",
                        "target_lists",
                        "stage1_corpus",
                    )
                }
            ),
            "features": {
                "root": str(features),
                "metadata": feature_metadata,
                "merge_summary": merge_summary,
                "manifest_sha256": _sha256(features / "manifest.jsonl"),
                "teacher_manifest_sha256": _sha256(
                    features / "teacher_manifest.jsonl"
                ),
            },
            "pca": {
                "path": str(pca_path),
                "sha256": _sha256(pca_path),
                "objects": int(pca["objects"]),
                "input_dim": int(pca["input_dim"]),
                "student_dim": int(pca["student_dim"]),
                "explained_variance_ratio": float(pca["explained_variance_ratio"]),
                "corpus_sha256": str(pca["corpus_sha256"]),
            },
            "raw_index": (
                {
                    "status": "complete",
                    "manifest": str(raw_index_manifest.resolve()),
                    "manifest_sha256": _sha256(raw_index_manifest),
                }
                if raw_index_manifest.is_file()
                else {"status": "pending", "manifest": str(raw_index_manifest.resolve())}
            ),
            "teacher_transfer": {
                "checkpoint": str(teacher_checkpoint),
                "sha256": _sha256(teacher_checkpoint),
                "trained_on_corpus_sha256": teacher_selection.get("corpus_sha256"),
                "selection_dataset": teacher_selection.get("best_metrics", {})
                .get("teacher_rerank", {})
                .get("by_dataset", {}),
                "interpretation": (
                    "Historical r5 Teacher transferred to fresh v9 hidden states; "
                    "not a Teacher trained on the v9 train split."
                ),
            },
        },
        "evaluation_protocol": {
            "selection_split": "dev",
            "test_policy": "untouched until final configuration is frozen",
            "recall_ks": [10, 20, 50],
            "direct_k": 100,
            "evidence_k_per_modality": 20,
            "targets_per_evidence": 20,
            "stage2_evidence_per_target": 4,
            "training_seeds": [13, 17, 23],
            "screening_seed": 13,
        },
        "wdc": {
            "status": "pending_input",
            "note": "EntiTables proceeds independently; no older WDC lake is substituted.",
        },
    }

    _write_json(output_dir / "inputs.json", inputs)
    _write_json(output_dir / "label_audit.json", label_audit)
    _write_json(output_dir / "splits.json", split_payload)
    summary = {
        "status": "pass",
        "inputs": str((output_dir / "inputs.json").resolve()),
        "label_audit": str((output_dir / "label_audit.json").resolve()),
        "splits": str((output_dir / "splits.json").resolve()),
        "query_counts": split_payload["counts"]["queries"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--stage1-data", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
