#!/usr/bin/env python
"""Build raw-ANN-aligned edge/path lists for Stage-1 Teacher retraining."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections import defaultdict
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


def _relation_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{record['source_type']}_to_{record['destination_type']}"
                for record in records
            ).items()
        )
    )


def _evidence_concentration(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        str(evidence_id)
        for record in records
        for candidate in record["candidates"]
        for evidence_id in candidate["evidence_ids"]
    )
    total = sum(counts.values())
    ranked = [count for _value, count in counts.most_common()]
    return {
        "occurrences": total,
        "unique": len(counts),
        "top_1_share": ranked[0] / total if total else 0.0,
        "top_10_share": sum(ranked[:10]) / total if total else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.list_width < 2 or args.handcrafted_negatives < 0:
        raise ValueError("invalid candidate-list width or handcrafted quota")
    if args.evidence_per_type <= 0:
        raise ValueError("--evidence-per-type must be positive")
    if args.min_train_relation_records < 0:
        raise ValueError("--min-train-relation-records must be non-negative")
    if bool(args.negative_corpus) != bool(args.negative_raw_index_root):
        raise ValueError(
            "--negative-corpus and --negative-raw-index-root must be used together"
        )
    splits = set(args.splits)
    unavailable_ids = _selector_ids(args.exclude_object_ids)
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    negative_corpus_path = Path(args.negative_corpus or args.corpus)
    negative_index_root = Path(
        args.negative_raw_index_root or args.raw_index_root
    )
    negative_corpus_sha256 = checkpoint_fingerprint(negative_corpus_path)
    indices = load_or_build_raw_embedding_indices(
        store,
        load_corpus_ids(negative_corpus_path, store),
        negative_index_root,
        corpus_sha256=negative_corpus_sha256,
        batch_size=args.index_batch_size,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "edge_lists.jsonl"
    target_path = output_dir / "target_lists.jsonl"
    source_edge_records = list(_records(args.edge_data, splits))
    source_target_records = list(_records(args.target_data, splits))
    train_relation_counts = _relation_counts(
        record
        for record in source_edge_records
        if record.get("split") == "train"
    )
    sparse_relations = {
        relation
        for relation, count in train_relation_counts.items()
        if count < args.min_train_relation_records
    }
    disabled_modality_relations = {
        relation
        for relation in train_relation_counts
        for evidence_type in ("text", "image")
        if evidence_type not in args.evidence_types
        and evidence_type in relation.split("_to_")
    }
    suppressed_relations = sparse_relations | disabled_modality_relations
    filtered_edge_records = [
        record
        for record in source_edge_records
        if f"{record['source_type']}_to_{record['destination_type']}"
        not in suppressed_relations
    ]
    target_evidence: dict[str, list[str]] = defaultdict(list)
    for record in source_target_records:
        for candidate in record["candidates"]:
            target_id = str(candidate["target_id"])
            for evidence_id in candidate.get("evidence_ids", []):
                evidence_id = str(evidence_id)
                if evidence_id not in target_evidence[target_id]:
                    target_evidence[target_id].append(evidence_id)

    edge_records = [
        align_edge_record(
            record,
            indices,
            list_width=args.list_width,
            handcrafted_negatives=args.handcrafted_negatives,
            unavailable_ids=unavailable_ids,
        )
        for record in progress(
            filtered_edge_records,
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
            evidence_binding=args.evidence_binding,
            target_evidence=target_evidence,
            evidence_types=args.evidence_types,
        )
        for record in progress(
            source_target_records,
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
        "negative_corpus": str(negative_corpus_path.resolve()),
        "negative_corpus_sha256": negative_corpus_sha256,
        "negative_raw_index_root": str(negative_index_root.resolve()),
        "mixed_negative_source": negative_corpus_sha256 != corpus_sha256,
        "list_width": args.list_width,
        "handcrafted_negatives": args.handcrafted_negatives,
        "evidence_per_type": args.evidence_per_type,
        "evidence_binding": args.evidence_binding,
        "evidence_types": args.evidence_types,
        "min_train_relation_records": args.min_train_relation_records,
        "train_relation_counts_before": train_relation_counts,
        "train_relation_counts_after": _relation_counts(
            record
            for record in filtered_edge_records
            if record.get("split") == "train"
        ),
        "suppressed_train_relations": sorted(suppressed_relations),
        "sparse_train_relations": sorted(sparse_relations),
        "disabled_modality_relations": sorted(disabled_modality_relations),
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
        "evidence_concentration": _evidence_concentration(target_records),
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
    parser.add_argument(
        "--negative-corpus",
        help="Optional corpus supplying ANN negatives; defaults to --corpus.",
    )
    parser.add_argument(
        "--negative-raw-index-root",
        help="Raw index for --negative-corpus; defaults to --raw-index-root.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--list-width", type=int, default=16)
    parser.add_argument("--handcrafted-negatives", type=int, default=4)
    parser.add_argument("--evidence-per-type", type=int, default=1)
    parser.add_argument(
        "--evidence-binding",
        choices=["query-hard", "target-bound"],
        default="query-hard",
    )
    parser.add_argument(
        "--evidence-types",
        nargs="+",
        choices=["text", "image"],
        default=["text", "image"],
    )
    parser.add_argument(
        "--min-train-relation-records",
        type=int,
        default=0,
        help="Drop train-only edge relations whose observed count is below this value.",
    )
    parser.add_argument("--exclude-object-ids", nargs="*", default=[])
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
