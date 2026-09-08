#!/usr/bin/env python
"""Attach auditable R10 ranking-positive and confirmed edge labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


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


def _support_sets(
    qrels: Iterable[dict[str, Any]],
    recoveries: Iterable[dict[str, Any]],
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    targets_by_query: dict[str, set[str]] = defaultdict(set)
    evidence_by_query: dict[str, set[str]] = defaultdict(set)
    targets_by_evidence: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        targets_by_query[str(row["query_table_id"])].add(
            str(row["target_table_id"])
        )
    for row in recoveries:
        query_id = str(row["query_table_id"])
        target_id = str(row["target_table_id"])
        evidence_id = str(row["evidence"]["asset_id"])
        evidence_by_query[query_id].add(evidence_id)
        targets_by_evidence[evidence_id].add(target_id)
    return targets_by_query, evidence_by_query, targets_by_evidence


def _corrupted_pairs(
    target_records: Iterable[dict[str, Any]],
    targets_by_query: dict[str, set[str]],
    evidence_by_query: dict[str, set[str]],
) -> set[tuple[str, str]]:
    pairs = set()
    for row in target_records:
        query_id = str(row["query_id"])
        known_targets = targets_by_query.get(query_id, set())
        known_evidence = evidence_by_query.get(query_id, set())
        for candidate in row["candidates"]:
            target_id = str(candidate["target_id"])
            if target_id in known_targets:
                continue
            for evidence_id in candidate.get("evidence_ids", []):
                evidence_id = str(evidence_id)
                if evidence_id in known_evidence:
                    pairs.add((evidence_id, target_id))
    return pairs


def label_edge_records(
    edge_records: Iterable[dict[str, Any]],
    *,
    targets_by_query: dict[str, set[str]],
    evidence_by_query: dict[str, set[str]],
    targets_by_evidence: dict[str, set[str]],
    corrupted_pairs: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label known positives and only independently constructed corrupt negatives."""

    labeled = []
    relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    corrupted_targets_by_evidence: dict[str, set[str]] = defaultdict(set)
    for evidence_id, target_id in corrupted_pairs:
        corrupted_targets_by_evidence[evidence_id].add(target_id)
    supported_corruption_overlaps = sum(
        target_id in targets_by_evidence.get(evidence_id, set())
        for evidence_id, target_id in corrupted_pairs
    )
    for source in edge_records:
        row = dict(source)
        source_id = str(row["query_id"])
        source_type = str(row.get("source_type", ""))
        destination_type = str(row.get("destination_type", ""))
        candidate_ids = [str(value) for value in row["candidate_ids"]]
        relation = f"{source_type}_to_{destination_type}"
        if source_type == "table" and destination_type == "table":
            ranking_positive_set = targets_by_query.get(source_id, set())
            confirmed_positive_set: set[str] = set()
            confirmed_negative_set: set[str] = set()
        elif source_type == "table" and destination_type in {"text", "image"}:
            ranking_positive_set = evidence_by_query.get(source_id, set())
            confirmed_positive_set = ranking_positive_set
            confirmed_negative_set = set()
        elif source_type in {"text", "image"} and destination_type == "table":
            ranking_positive_set = targets_by_evidence.get(source_id, set())
            confirmed_positive_set = ranking_positive_set
            confirmed_negative_set = (
                corrupted_targets_by_evidence.get(source_id, set())
                - ranking_positive_set
            )
        else:
            ranking_positive_set = {str(row["positive_id"])}
            confirmed_positive_set = set()
            confirmed_negative_set = set()

        positive_ids = [
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id in ranking_positive_set
        ]
        designated_positive = str(row["positive_id"])
        if designated_positive not in positive_ids:
            raise ValueError(
                f"{source_id}: designated {relation} positive lacks support provenance"
            )
        confirmed_labels = [
            1
            if candidate_id in confirmed_positive_set
            else 0
            if candidate_id in confirmed_negative_set
            else None
            for candidate_id in candidate_ids
        ]
        if any(
            candidate_id in confirmed_positive_set
            and candidate_id in confirmed_negative_set
            for candidate_id in candidate_ids
        ):
            raise ValueError(f"{source_id}: confirmed edge labels conflict")
        row["positive_ids"] = positive_ids
        row["confirmed_labels"] = confirmed_labels
        labeled.append(row)

        counts = relation_counts[relation]
        counts["records"] += 1
        counts["candidates"] += len(candidate_ids)
        counts["ranking_positives"] += len(positive_ids)
        counts["confirmed_positive"] += confirmed_labels.count(1)
        counts["confirmed_negative"] += confirmed_labels.count(0)
        counts["unknown"] += confirmed_labels.count(None)
        if 1 in confirmed_labels:
            counts["lists_with_confirmed_positive"] += 1
        if 0 in confirmed_labels:
            counts["lists_with_confirmed_negative"] += 1
        if 1 in confirmed_labels and 0 in confirmed_labels:
            counts["lists_with_both_labels"] += 1

    summary = {
        "records": len(labeled),
        "corrupted_pairs": len(corrupted_pairs),
        "supported_corruption_overlaps_excluded": supported_corruption_overlaps,
        "by_relation": {
            relation: dict(sorted(counts.items()))
            for relation, counts in sorted(relation_counts.items())
        },
    }
    return labeled, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    edge_path = Path(args.edge_lists).resolve()
    target_path = Path(args.target_lists).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qrels = list(_artifact_records(dataset_root, manifest, "qrels"))
    recoveries = list(
        _artifact_records(dataset_root, manifest, "evidence_recoveries")
    )
    targets_by_query, evidence_by_query, targets_by_evidence = _support_sets(
        qrels, recoveries
    )
    corrupted_pairs = _corrupted_pairs(
        _jsonl(target_path), targets_by_query, evidence_by_query
    )
    labeled, summary = label_edge_records(
        _jsonl(edge_path),
        targets_by_query=targets_by_query,
        evidence_by_query=evidence_by_query,
        targets_by_evidence=targets_by_evidence,
        corrupted_pairs=corrupted_pairs,
    )
    written = _write_jsonl(output_path, labeled)
    if written != summary["records"]:
        raise RuntimeError("Edge-label output count changed while writing")
    metadata = {
        "format_version": 1,
        "label_policy": {
            "ranking_positives": "all recovery/qrel-supported candidates in each list",
            "confirmed_positive": "recovery-supported Q-E and E-T edges",
            "confirmed_negative": "constructed E-T corruption pairs after excluding every pair with recovery support anywhere in the frozen dataset",
            "confirmed_negative_scope": "closed-world supervision for the explicit construction; it is not proof that an unannotated E-T pair is universally invalid in an open-world lake",
            "unknown": "all remaining candidates; excluded from BCE",
            "direct_table_edges": "ranking supervision only; no absolute labels",
        },
        "inputs": {
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": _sha256(manifest_path),
            "edge_lists": str(edge_path),
            "edge_lists_sha256": _sha256(edge_path),
            "target_lists": str(target_path),
            "target_lists_sha256": _sha256(target_path),
        },
        "summary": summary,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    _write_json(metadata_path, metadata)
    print(json.dumps({"output": str(output_path), **summary}, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--edge-lists", required=True)
    parser.add_argument("--target-lists", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
