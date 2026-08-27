#!/usr/bin/env python
"""Migrate a legacy WDC output to query-only splits with a shared data lake."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SPLITS = ("train", "dev", "test")
SPLIT_SCHEMA_VERSION = "query-only-shared-data-lake-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document is not an object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record is not an object: {path}:{line_number}")
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path, *, relative_path: str, records: int) -> dict[str, Any]:
    return {
        "path": relative_path,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "mtime_ns": path.stat().st_mtime_ns,
    }


def _temporary(path: Path) -> Path:
    return path.with_name(path.name + ".query-only.tmp")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_records_without_split(source: Path, target: Path) -> tuple[int, int]:
    records = 0
    removed = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in _iter_jsonl(source):
            if "split" in record:
                record.pop("split")
                removed += 1
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            records += 1
        handle.flush()
        os.fsync(handle.fileno())
    return records, removed


def _artifact_paths(root: Path, manifest: dict[str, Any], name: str) -> list[Path]:
    artifact = manifest.get("artifacts", {}).get(name)
    if not isinstance(artifact, dict):
        raise TypeError(f"manifest has no sharded artifact: {name}")
    shards = artifact.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"manifest artifact has no shards: {name}")
    return [root / str(shard["path"]) for shard in shards]


def _single_path(root: Path, manifest: dict[str, Any], name: str) -> Path:
    value = manifest.get("single_files", {}).get(name)
    if not isinstance(value, str):
        raise TypeError(f"manifest has no single-file artifact: {name}")
    return root / value


def migrate_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "dataset_manifest.json"
    splits_path = root / "splits.json"
    manifest = _read_json(manifest_path)
    old_splits = _read_json(splits_path)
    if manifest.get("complete") is not True:
        raise ValueError("dataset manifest is not complete")

    query_splits: dict[str, str] = {}
    query_sources: dict[str, str] = {}
    query_counts = {split: 0 for split in SPLITS}
    for path in _artifact_paths(root, manifest, "query_tables"):
        for record in _iter_jsonl(path):
            query_id = str(record["table_id"])
            split = str(record["split"])
            source_id = str(record["source_table_id"])
            if split not in query_counts:
                raise ValueError(f"query has invalid split: {query_id}={split}")
            if query_id in query_splits:
                raise ValueError(f"duplicate query table ID: {query_id}")
            query_splits[query_id] = split
            query_sources[query_id] = source_id
            query_counts[split] += 1

    source_splits: dict[str, str] = {}
    for query_id, source_id in query_sources.items():
        split = query_splits[query_id]
        previous = source_splits.setdefault(source_id, split)
        if previous != split:
            raise ValueError(f"source table crosses query splits: {source_id}")

    old_query_ids = {
        str(query_id)
        for split in SPLITS
        for query_id in old_splits.get(split, {}).get("query_table_ids", [])
    }
    if old_query_ids and old_query_ids != query_splits.keys():
        raise ValueError("splits.json query IDs do not match query_tables")

    target_replacements: list[tuple[Path, Path, dict[str, Any]]] = []
    target_ids: set[str] = set()
    target_split_fields = 0
    target_artifact = manifest["artifacts"]["data_lake_tables"]
    for shard, source in zip(target_artifact["shards"], _artifact_paths(root, manifest, "data_lake_tables")):
        temporary = _temporary(source)
        records = 0
        removed = 0
        with temporary.open("w", encoding="utf-8") as handle:
            for record in _iter_jsonl(source):
                target_id = str(record["table_id"])
                if target_id in target_ids:
                    raise ValueError(f"duplicate data-lake table ID: {target_id}")
                target_ids.add(target_id)
                if "split" in record:
                    record.pop("split")
                    removed += 1
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                records += 1
            handle.flush()
            os.fsync(handle.fileno())
        if records != int(shard["records"]):
            raise ValueError(f"data-lake shard record count changed: {source}")
        target_split_fields += removed
        target_replacements.append(
            (
                source,
                temporary,
                _metadata(temporary, relative_path=str(shard["path"]), records=records),
            )
        )

    old_target_ids = {
        str(target_id)
        for split in SPLITS
        for target_id in old_splits.get(split, {}).get("data_lake_table_ids", [])
    }
    if old_target_ids and old_target_ids != target_ids:
        raise ValueError("splits.json target IDs do not match data_lake_tables")

    for record in _iter_jsonl(_single_path(root, manifest, "qrels")):
        query_id = str(record["query_table_id"])
        target_id = str(record["target_table_id"])
        if query_id not in query_splits or target_id not in target_ids:
            raise ValueError(f"qrel reference is invalid: {query_id} -> {target_id}")
        if str(record.get("split")) != query_splits[query_id]:
            raise ValueError(f"qrel split differs from its query: {query_id}")

    for path in _artifact_paths(root, manifest, "evidence_recoveries"):
        for record in _iter_jsonl(path):
            query_id = str(record["query_table_id"])
            target_id = str(record["target_table_id"])
            if query_id not in query_splits or target_id not in target_ids:
                raise ValueError(
                    f"evidence recovery reference is invalid: {query_id} -> {target_id}"
                )
            if str(record.get("split")) != query_splits[query_id]:
                raise ValueError(f"recovery split differs from its query: {query_id}")

    decisions_path = _single_path(root, manifest, "table_queryability_decisions")
    decisions_temporary = _temporary(decisions_path)
    decision_records, decision_split_fields = _write_records_without_split(
        decisions_path, decisions_temporary
    )

    assignments_dir = root / "split_assignments"
    assignments_dir.mkdir(exist_ok=True)
    assignments_path = assignments_dir / "part-00000.jsonl"
    assignments_temporary = _temporary(assignments_path)
    with assignments_temporary.open("w", encoding="utf-8") as handle:
        for query_id in sorted(query_splits):
            record = {
                "object_id": query_id,
                "object_type": "query_table",
                "source_table_id": query_sources[query_id],
                "split": query_splits[query_id],
            }
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())

    split_key = str(old_splits.get("split_key", "source_table_id"))
    new_splits = {
        "split_key": split_key,
        "split_policy": "query_only",
        "data_lake_scope": "shared",
        "query_table_counts": query_counts,
        "data_lake_table_count": len(target_ids),
        "assignments_artifact": "split_assignments",
        "data_lake_artifact": "data_lake_tables",
    }
    splits_temporary = _temporary(splits_path)
    _write_json(splits_temporary, new_splits)

    for shard, (_, _, metadata) in zip(target_artifact["shards"], target_replacements):
        shard.clear()
        shard.update(metadata)
    target_artifact["total_records"] = len(target_ids)

    assignment_metadata = _metadata(
        assignments_temporary,
        relative_path="split_assignments/part-00000.jsonl",
        records=len(query_splits),
    )
    manifest["artifacts"]["split_assignments"] = {
        "directory": "split_assignments",
        "total_records": len(query_splits),
        "max_records_per_shard": int(manifest.get("records_per_shard", 10000)),
        "shards": [assignment_metadata],
    }
    manifest["published_single_files"]["splits.json"] = _metadata(
        splits_temporary, relative_path="splits.json", records=1
    )
    manifest["published_single_files"][decisions_path.name] = _metadata(
        decisions_temporary,
        relative_path=decisions_path.name,
        records=decision_records,
    )
    construction = manifest.setdefault("query_construction", {})
    construction["split_policy"] = "query_only"
    construction["data_lake_scope"] = "shared"
    manifest["split_schema_version"] = SPLIT_SCHEMA_VERSION
    manifest["note"] = (
        "Read only files listed in this manifest. Train/dev/test assignments apply "
        "only to queries; every split retrieves from the complete shared data lake."
    )
    manifest_temporary = _temporary(manifest_path)
    _write_json(manifest_temporary, manifest)

    for source, temporary, _ in target_replacements:
        os.replace(temporary, source)
    os.replace(decisions_temporary, decisions_path)
    os.replace(assignments_temporary, assignments_path)
    os.replace(splits_temporary, splits_path)
    os.replace(manifest_temporary, manifest_path)

    return {
        "dataset_root": str(root),
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "query_table_counts": query_counts,
        "data_lake_table_count": len(target_ids),
        "data_lake_split_fields_removed": target_split_fields,
        "decision_split_fields_removed": decision_split_fields,
        "split_assignment_count": len(query_splits),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    result = migrate_dataset(parse_args().dataset_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
