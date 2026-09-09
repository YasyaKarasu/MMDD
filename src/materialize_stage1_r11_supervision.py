#!/usr/bin/env python
"""Materialize source-isolated R11 Stage-1 supervision buckets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.construction import build_stage1_training_artifacts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _teacher_object_ids(path: Path) -> set[str]:
    object_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "object_id" not in record:
                raise ValueError(f"{path}:{line_number}: missing object_id")
            object_ids.add(str(record["object_id"]))
    return object_ids


def _filter_teacher_candidates(
    targets: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    teacher_ids: set[str],
) -> dict[str, int]:
    """Remove only optional candidates that the token Teacher cannot score."""

    removed_target_candidates = 0
    removed_target_evidence = 0
    removed_edge_candidates = 0
    for target in targets:
        required = {
            str(target["query_id"]),
            *(str(value) for value in target["positive_target_ids"]),
            *(
                str(evidence_id)
                for evidence_ids in target["positive_evidence_by_target"].values()
                for evidence_id in evidence_ids
            ),
        }
        missing_required = required - teacher_ids
        if missing_required:
            raise ValueError(
                "Token Teacher features are missing for required target-list "
                f"objects: {sorted(missing_required)[:10]}"
            )
        filtered_candidates = []
        for candidate in target["candidates"]:
            if str(candidate["target_id"]) not in teacher_ids:
                removed_target_candidates += 1
                continue
            evidence_ids = [
                value
                for value in candidate["evidence_ids"]
                if str(value) in teacher_ids
            ]
            removed_target_evidence += len(candidate["evidence_ids"]) - len(
                evidence_ids
            )
            filtered_candidates.append({**candidate, "evidence_ids": evidence_ids})
        if len(filtered_candidates) < 2:
            raise ValueError(
                f"{target['query_id']}: fewer than two Teacher-scoreable targets"
            )
        target["candidates"] = filtered_candidates

    for edge in edges:
        source_id = str(edge["query_id"])
        positive_ids = {str(value) for value in edge["positive_ids"]}
        missing_required = ({source_id} | positive_ids) - teacher_ids
        if missing_required:
            raise ValueError(
                "Token Teacher features are missing for required edge-list "
                f"objects: {sorted(missing_required)[:10]}"
            )
        pairs = [
            (candidate_id, label)
            for candidate_id, label in zip(
                edge["candidate_ids"], edge["confirmed_labels"]
            )
            if str(candidate_id) in teacher_ids
        ]
        removed_edge_candidates += len(edge["candidate_ids"]) - len(pairs)
        if len(pairs) < 2:
            raise ValueError(
                f"{source_id}: fewer than two Teacher-scoreable edge candidates"
            )
        edge["candidate_ids"] = [candidate_id for candidate_id, _ in pairs]
        edge["confirmed_labels"] = [label for _, label in pairs]
        edge["positive_id"] = next(
            candidate_id
            for candidate_id in edge["candidate_ids"]
            if str(candidate_id) in positive_ids
        )
    return {
        "target_candidates_removed": removed_target_candidates,
        "target_evidence_removed": removed_target_evidence,
        "edge_candidates_removed": removed_edge_candidates,
    }


def calibration_query_buckets(
    dataset_root: Path,
    protocol: dict[str, Any],
    *,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    query_source = {
        str(row["table_id"]): str(row["source_table_id"])
        for row in iter_dataset_artifact(dataset_root, "query_tables")
    }
    calibration_queries = {
        str(value) for value in protocol["query_ids"]["train_calibration"]
    }
    calibration_groups = sorted(
        {query_source[query_id] for query_id in calibration_queries}
    )
    random.Random(seed).shuffle(calibration_groups)
    fit_count = (len(calibration_groups) + 1) // 2
    fit_groups = set(calibration_groups[:fit_count])
    check_groups = set(calibration_groups[fit_count:])
    cal_fit = sorted(
        query_id
        for query_id in calibration_queries
        if query_source[query_id] in fit_groups
    )
    cal_check = sorted(calibration_queries - set(cal_fit))
    return cal_fit, cal_check, sorted(fit_groups), sorted(check_groups)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    split_path = Path(args.splits).resolve()
    output_dir = Path(args.output_dir).resolve()
    protocol = json.loads(split_path.read_text(encoding="utf-8"))
    teacher_manifest = Path(args.teacher_manifest).resolve()
    teacher_ids = _teacher_object_ids(teacher_manifest)
    cal_fit, cal_check, cal_fit_groups, cal_check_groups = (
        calibration_query_buckets(dataset_root, protocol, seed=args.seed)
    )
    buckets = {
        "train_fit": sorted(
            str(value) for value in protocol["query_ids"]["train_fit"]
        ),
        "cal_fit": cal_fit,
        "cal_check": cal_check,
        "dev": sorted(str(value) for value in protocol["query_ids"]["dev"]),
        "r10_test_regression": sorted(
            str(value) for value in protocol["query_ids"]["test"]
        ),
    }
    query_sets = {name: set(values) for name, values in buckets.items()}
    query_source = {
        str(row["table_id"]): str(row["source_table_id"])
        for row in iter_dataset_artifact(dataset_root, "query_tables")
    }
    source_sets = {
        name: {query_source[value] for value in values}
        for name, values in query_sets.items()
    }
    for left, left_values in query_sets.items():
        for right, right_values in query_sets.items():
            if left >= right:
                continue
            overlap = left_values & right_values
            if overlap:
                raise ValueError(
                    f"R11 supervision buckets overlap: {left}/{right}: "
                    f"{sorted(overlap)[:10]}"
                )
            source_overlap = source_sets[left] & source_sets[right]
            if source_overlap:
                raise ValueError(
                    f"Supervision source groups overlap: {left}/{right}: "
                    f"{sorted(source_overlap)[:10]}"
                )

    outputs: dict[str, Any] = {}
    for bucket, query_ids in buckets.items():
        artifacts = build_stage1_training_artifacts(
            dataset_root,
            dataset_name=args.dataset_name,
            max_rows=args.max_rows,
            seed=args.seed,
            supervision_query_ids=set(query_ids),
        )
        split_value = (
            "train"
            if bucket in {"train_fit", "cal_fit", "cal_check"}
            else "dev"
            if bucket == "dev"
            else "test"
        )
        targets = [
            {**row, "split": split_value, "protocol_bucket": bucket}
            for row in artifacts["target_lists"]
        ]
        edges = [
            {**row, "split": split_value, "protocol_bucket": bucket}
            for row in artifacts["edge_lists"]
        ]
        teacher_filter = _filter_teacher_candidates(targets, edges, teacher_ids)
        target_path = output_dir / f"target_lists.{bucket}.jsonl"
        edge_path = output_dir / f"edge_lists.{bucket}.jsonl"
        target_count = _write_jsonl(target_path, targets)
        edge_count = _write_jsonl(edge_path, edges)
        if target_count != len(query_ids):
            raise ValueError(
                f"{bucket}: wrote {target_count} target lists for "
                f"{len(query_ids)} protocol queries"
            )
        outputs[bucket] = {
            "queries": target_count,
            "edge_lists": edge_count,
            "target_lists": str(target_path),
            "target_lists_sha256": _sha256(target_path),
            "edge_lists_path": str(edge_path),
            "edge_lists_sha256": _sha256(edge_path),
            "teacher_candidate_filter": teacher_filter,
        }

    manifest = {
        "format_version": 1,
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": _sha256(
            dataset_root / "dataset_manifest.json"
        ),
        "source_protocol": str(split_path),
        "source_protocol_sha256": _sha256(split_path),
        "teacher_manifest": str(teacher_manifest),
        "teacher_manifest_sha256": _sha256(teacher_manifest),
        "teacher_objects": len(teacher_ids),
        "seed": args.seed,
        "supervision_policy": (
            "qrels and recoveries are filtered to each query bucket before "
            "positive, corruption, and Teacher candidate lists are built; "
            "corrupted E-to-T targets are ranking-only unknowns, not confirmed negatives"
        ),
        "source_group_overlap_checked": True,
        "source_group_counts": {name: len(values) for name, values in source_sets.items()},
        "calibration": {
            "policy": (
                "seeded shuffle of sorted train-calibration source groups; "
                "ceil-half to cal_fit and remainder to cal_check"
            ),
            "cal_fit_source_groups": cal_fit_groups,
            "cal_check_source_groups": cal_check_groups,
        },
        "outputs": outputs,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "pass", **manifest}, ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--max-rows", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
