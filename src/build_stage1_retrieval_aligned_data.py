#!/usr/bin/env python
"""Build raw-ANN-aligned edge/path lists for Stage-1 Teacher retraining."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from mmdd_progress import progress
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    checkpoint_fingerprint,
    load_corpus_ids,
    load_or_build_raw_embedding_indices,
)
from mmdd_stage1.retrieval_aligned import align_edge_record, align_target_record
from mmdd_stage1.selection import write_json


def _records(paths: Iterable[str], splits: set[str]):
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                if record.get("split") in splits:
                    yield record


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _referenced_ids(record: dict[str, Any]) -> set[str]:
    values = {str(record["query_id"])}
    if "candidate_ids" in record:
        values.update(str(value) for value in record["candidate_ids"])
    else:
        for candidate in record["candidates"]:
            values.add(str(candidate["target_id"]))
            values.update(str(value) for value in candidate["evidence_ids"])
    return values


def _selector_ids(paths: Iterable[str]) -> set[str]:
    selected = set()
    for value in paths:
        with Path(value).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    selected.add(str(json.loads(line)["object_id"]))
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.list_width < 2 or args.handcrafted_negatives < 0:
        raise ValueError("invalid candidate-list width or handcrafted quota")
    if args.evidence_per_type <= 0:
        raise ValueError("--evidence-per-type must be positive")
    splits = set(args.splits)
    unavailable_ids = _selector_ids(args.exclude_object_ids)
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    indices = load_or_build_raw_embedding_indices(
        store,
        load_corpus_ids(corpus_path, store),
        Path(args.raw_index_root),
        corpus_sha256=corpus_sha256,
        batch_size=args.index_batch_size,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "edge_lists.jsonl"
    target_path = output_dir / "target_lists.jsonl"
    edge_records = [
        align_edge_record(
            record,
            indices,
            list_width=args.list_width,
            handcrafted_negatives=args.handcrafted_negatives,
            unavailable_ids=unavailable_ids,
        )
        for record in progress(
            _records(args.edge_data, splits),
            desc="Align Teacher edge lists",
            unit="list",
        )
    ]
    target_records = [
        align_target_record(
            record,
            indices,
            store.object_type,
            list_width=args.list_width,
            handcrafted_negatives=args.handcrafted_negatives,
            evidence_per_type=args.evidence_per_type,
            unavailable_ids=unavailable_ids,
        )
        for record in progress(
            _records(args.target_data, splits),
            desc="Align Teacher path lists",
            unit="list",
        )
    ]
    _write_jsonl(edge_path, edge_records)
    _write_jsonl(target_path, target_records)
    referenced_ids: set[str] = set()
    for record in [*edge_records, *target_records]:
        referenced_ids.update(_referenced_ids(record))
    missing_teacher_ids = sorted(
        object_id
        for object_id in referenced_ids
        if not store.has_teacher_features(object_id)
    )
    missing_path = output_dir / "missing_teacher_ids.jsonl"
    _write_jsonl(
        missing_path,
        ({"object_id": object_id} for object_id in missing_teacher_ids),
    )
    summary = {
        "format_version": 1,
        "corpus_sha256": corpus_sha256,
        "list_width": args.list_width,
        "handcrafted_negatives": args.handcrafted_negatives,
        "evidence_per_type": args.evidence_per_type,
        "splits": sorted(splits),
        "edge_lists": str(edge_path.resolve()),
        "target_lists": str(target_path.resolve()),
        "edge_records": len(edge_records),
        "target_records": len(target_records),
        "edge_by_split": dict(sorted(Counter(value["split"] for value in edge_records).items())),
        "target_by_split": dict(sorted(Counter(value["split"] for value in target_records).items())),
        "referenced_objects": len(referenced_ids),
        "missing_teacher_objects": len(missing_teacher_ids),
        "missing_teacher_ids": str(missing_path.resolve()),
        "excluded_objects": len(unavailable_ids),
    }
    write_json(output_dir / "preflight.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--edge-data", nargs="+", required=True)
    parser.add_argument("--target-data", nargs="+", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--raw-index-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--list-width", type=int, default=16)
    parser.add_argument("--handcrafted-negatives", type=int, default=4)
    parser.add_argument("--evidence-per-type", type=int, default=1)
    parser.add_argument("--exclude-object-ids", nargs="*", default=[])
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
