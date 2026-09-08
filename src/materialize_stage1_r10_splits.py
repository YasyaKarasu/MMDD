#!/usr/bin/env python
"""Materialize source-group-isolated Stage-1 lists from the frozen R10 split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _query_buckets(split: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for bucket, query_ids in split["query_ids"].items():
        for query_id in query_ids:
            query_id = str(query_id)
            if query_id in result:
                raise ValueError(f"Query appears in multiple protocol buckets: {query_id}")
            result[query_id] = str(bucket)
    return result


def partition_records(
    target_records: list[dict[str, Any]],
    edge_records: list[dict[str, Any]],
    query_buckets: dict[str, str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    target_partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_target_buckets: dict[tuple[str, str], set[str]] = defaultdict(set)

    for record in target_records:
        query_id = str(record["query_id"])
        try:
            bucket = query_buckets[query_id]
        except KeyError as exc:
            raise ValueError(f"Target list has an unassigned query: {query_id}") from exc
        target_partitions[bucket].append(record)
        positive_target_ids = set(
            str(value) for value in record.get("positive_target_ids", ())
        )
        positive_target_ids.update(
            str(record[key])
            for key in (
                "direct_positive_target_id",
                "evidence_positive_target_id",
                "positive_target_id",
            )
            if record.get(key) is not None
        )
        for candidate in record["candidates"]:
            target_id = str(candidate["target_id"])
            if target_id not in positive_target_ids:
                continue
            for evidence_id in candidate.get("evidence_ids", ()):
                evidence_target_buckets[(str(evidence_id), target_id)].add(bucket)

    for record in edge_records:
        source_type = str(record.get("source_type", ""))
        destination_type = str(record.get("destination_type", ""))
        if source_type == "table":
            buckets = {query_buckets.get(str(record["query_id"]))}
        elif destination_type == "table":
            buckets = evidence_target_buckets.get(
                (str(record["query_id"]), str(record["positive_id"])), set()
            )
        else:
            buckets = set()
        buckets.discard(None)
        if not buckets:
            raise ValueError(
                "Edge list cannot be assigned to a protocol query: "
                f"{record['query_id']} -> {record['positive_id']}"
            )
        for bucket in sorted(buckets):
            edge_partitions[bucket].append(record)

    expected_queries = set(query_buckets)
    observed_queries = {
        str(record["query_id"])
        for records in target_partitions.values()
        for record in records
    }
    if observed_queries != expected_queries:
        missing = sorted(expected_queries - observed_queries)
        raise ValueError(f"Frozen split references missing target lists: {missing[:10]}")
    return dict(target_partitions), dict(edge_partitions)


def run(args: argparse.Namespace) -> dict[str, Any]:
    target_path = Path(args.target_lists).resolve()
    edge_path = Path(args.edge_lists).resolve()
    split_path = Path(args.splits).resolve()
    output_dir = Path(args.output_dir).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    query_buckets = _query_buckets(split)
    target_partitions, edge_partitions = partition_records(
        _records(target_path), _records(edge_path), query_buckets
    )

    outputs: dict[str, Any] = {}
    for bucket in split["query_ids"]:
        target_output = output_dir / f"target_lists.{bucket}.jsonl"
        edge_output = output_dir / f"edge_lists.{bucket}.jsonl"
        target_count = _write_jsonl(target_output, target_partitions.get(bucket, ()))
        edge_count = _write_jsonl(edge_output, edge_partitions.get(bucket, ()))
        expected_count = len(split["query_ids"][bucket])
        if target_count != expected_count:
            raise ValueError(
                f"{bucket}: wrote {target_count} target lists, expected {expected_count}"
            )
        outputs[bucket] = {
            "queries": target_count,
            "edge_lists": edge_count,
            "target_lists": str(target_output),
            "edge_lists_path": str(edge_output),
            "target_lists_sha256": _sha256(target_output),
            "edge_lists_sha256": _sha256(edge_output),
        }

    fit_edges = {
        (str(row["query_id"]), str(row["positive_id"]))
        for row in edge_partitions.get("train_fit", ())
    }
    calibration_edges = {
        (str(row["query_id"]), str(row["positive_id"]))
        for row in edge_partitions.get("train_calibration", ())
    }
    shared_train_edges = fit_edges & calibration_edges
    if shared_train_edges:
        raise ValueError(
            "train_fit and train_calibration share positive edge labels: "
            f"{sorted(shared_train_edges)[:10]}"
        )

    manifest = {
        "format_version": 1,
        "split": str(split_path),
        "split_sha256": _sha256(split_path),
        "inputs": {
            "target_lists": str(target_path),
            "target_lists_sha256": _sha256(target_path),
            "edge_lists": str(edge_path),
            "edge_lists_sha256": _sha256(edge_path),
        },
        "outputs": outputs,
        "train_fit_calibration_shared_positive_edges": 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "outputs": outputs}, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-lists", required=True)
    parser.add_argument("--edge-lists", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
