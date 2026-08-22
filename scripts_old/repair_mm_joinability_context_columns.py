#!/usr/bin/env python
"""Repair generated MM joinability query/target context column overlap in-place.

This is an offline projection repair. It does not call Wikipedia or local
models; it only rebuilds existing query/target tables from source_tables and
existing qrels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from build_mm_joinability_dataset import context_columns, project_selected_rows
from stage1_io import clean_text, load_json, make_columns, write_json, write_jsonl


def artifact_shard_paths(dataset_dir: Path, manifest: dict[str, Any], artifact: str) -> list[Path]:
    item = manifest.get("artifacts", {}).get(artifact)
    if not item:
        raise KeyError(f"Artifact {artifact!r} is missing from {dataset_dir / 'dataset_manifest.json'}")
    paths = [dataset_dir / shard["path"] for shard in item.get("shards", [])]
    if not paths:
        raise ValueError(f"Artifact {artifact!r} has no shards")
    return paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sharded_records(paths: list[Path]) -> list[tuple[Path, list[dict[str, Any]]]]:
    return [(path, read_jsonl(path)) for path in paths]


def source_indices(record: dict[str, Any]) -> list[int]:
    indices: list[int] = []
    for item in record.get("source_column_indices", []) or []:
        try:
            indices.append(int(item))
        except (TypeError, ValueError):
            continue
    return indices


def first_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def query_entity_col(query: dict[str, Any]) -> int | None:
    return first_int(query.get("query_entity_col"), query.get("query_entity_col_index"))


def target_join_col(query: dict[str, Any], target: dict[str, Any]) -> int | None:
    hidden_attrs = query.get("hidden_attributes") or []
    hidden_attr = hidden_attrs[0] if hidden_attrs and isinstance(hidden_attrs[0], dict) else {}
    return first_int(target.get("join_col"), hidden_attr.get("source_column_index"))


def source_row_filter(query: dict[str, Any], target: dict[str, Any], source_table: dict[str, Any]) -> set[int]:
    rows: set[int] = set()
    for record in (query, target):
        for item in record.get("source_row_indices", []) or []:
            try:
                rows.add(int(item))
            except (TypeError, ValueError):
                continue
    if rows:
        return rows
    fallback_rows: set[int] = set()
    for fallback, row in enumerate(source_table.get("rows", [])):
        try:
            fallback_rows.add(int(row.get("row_id", fallback)))
        except (TypeError, ValueError):
            fallback_rows.add(fallback)
    return fallback_rows


def ordered_existing_context(record: dict[str, Any], primary_col: int) -> list[int]:
    return [idx for idx in source_indices(record) if idx != primary_col]


def append_preferred_columns(
    selected: list[int],
    preferred: list[int],
    available: list[int],
    *,
    limit: int,
    forbidden: set[int] | None = None,
) -> list[int]:
    forbidden = forbidden or set()
    available_set = set(available)
    for idx in [*preferred, *available]:
        if idx in forbidden or idx not in available_set or idx in selected:
            continue
        selected.append(idx)
        if len(selected) >= limit:
            break
    return selected


def choose_non_overlapping_columns(
    source_table: dict[str, Any],
    query: dict[str, Any],
    target: dict[str, Any],
    entity_col: int,
    join_col: int,
) -> tuple[list[int], list[int]] | None:
    query_context_count = len(ordered_existing_context(query, entity_col))
    target_context_count = len(ordered_existing_context(target, join_col))
    available_context = context_columns(source_table, {entity_col, join_col}, 0)
    if len(available_context) < query_context_count + target_context_count:
        return None

    query_context: list[int] = []
    append_preferred_columns(
        query_context,
        ordered_existing_context(query, entity_col),
        available_context,
        limit=query_context_count,
    )
    if len(query_context) < query_context_count:
        return None

    target_context: list[int] = []
    append_preferred_columns(
        target_context,
        ordered_existing_context(target, join_col),
        available_context,
        limit=target_context_count,
        forbidden=set(query_context),
    )
    if len(target_context) < target_context_count:
        return None

    return [entity_col, *query_context], [join_col, *target_context]


def rebuild_projection(
    record: dict[str, Any],
    source_table: dict[str, Any],
    column_indices: list[int],
    source_rows: set[int],
    context_field: str,
    context_cols: list[int],
) -> dict[str, Any] | None:
    rows, projected_source_rows = project_selected_rows(
        source_table,
        column_indices,
        source_rows,
        min_required_cols=1,
    )
    if not rows:
        return None
    updated = dict(record)
    updated["columns"] = make_columns(source_table, column_indices)
    updated["rows"] = rows
    updated["source_column_indices"] = column_indices
    updated["source_row_indices"] = projected_source_rows
    updated[context_field] = [clean_text(column["column_name"]) for column in make_columns(source_table, context_cols)]
    return updated


def backup_artifacts(dataset_dir: Path) -> Path:
    backup_root = dataset_dir / "backups"
    backup_root.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"mm_joinability_context_repair_{stamp}"
    suffix = 1
    while backup_dir.exists():
        suffix += 1
        backup_dir = backup_root / f"mm_joinability_context_repair_{stamp}_{suffix}"
    backup_dir.mkdir()
    for name in ("query_tables", "data_lake_tables"):
        source = dataset_dir / name
        if source.exists():
            shutil.copytree(source, backup_dir / name)
    for name in ("dataset_manifest.json", "qrels.jsonl"):
        source = dataset_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)
    return backup_dir


def write_sharded_records(shards: list[tuple[Path, list[dict[str, Any]]]]) -> None:
    for path, records in shards:
        write_jsonl(path, records)


def repair_dataset(dataset_dir: Path | str, *, backup: bool = True, dry_run: bool = False) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    manifest = load_json(dataset_dir / "dataset_manifest.json")
    source_shards = load_sharded_records(artifact_shard_paths(dataset_dir, manifest, "source_tables"))
    query_shards = load_sharded_records(artifact_shard_paths(dataset_dir, manifest, "query_tables"))
    target_shards = load_sharded_records(artifact_shard_paths(dataset_dir, manifest, "data_lake_tables"))
    qrels_path = dataset_dir / manifest.get("single_files", {}).get("qrels", "qrels.jsonl")
    qrels = read_jsonl(qrels_path)

    sources = {record["source_table_id"]: record for _path, records in source_shards for record in records}
    queries = {record["table_id"]: record for _path, records in query_shards for record in records}
    targets = {record["table_id"]: record for _path, records in target_shards for record in records}
    updated_queries: dict[str, dict[str, Any]] = {}
    updated_targets: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    already_distinct = 0
    skipped = 0

    for qrel in qrels:
        query_id = clean_text(qrel.get("query_table_id"))
        target_id = clean_text(qrel.get("target_table_id") or qrel.get("data_lake_table_id"))
        query = queries.get(query_id)
        target = targets.get(target_id)
        if not query or not target or target.get("role") != "target_data_lake_table":
            skipped += 1
            continue
        query_cols = set(source_indices(query))
        target_cols = set(source_indices(target))
        if not (query_cols & target_cols):
            already_distinct += 1
            continue
        source_table = sources.get(clean_text(query.get("source_table_id")))
        entity_col = query_entity_col(query)
        join_col = target_join_col(query, target)
        if source_table is None or entity_col is None or join_col is None:
            unresolved.append({"query_table_id": query_id, "target_table_id": target_id, "reason": "missing_source_or_column_metadata"})
            continue
        choice = choose_non_overlapping_columns(source_table, query, target, entity_col, join_col)
        if choice is None:
            unresolved.append({"query_table_id": query_id, "target_table_id": target_id, "reason": "not_enough_distinct_context_columns"})
            continue
        new_query_cols, new_target_cols = choice
        rows_to_project = source_row_filter(query, target, source_table)
        rebuilt_query = rebuild_projection(
            query,
            source_table,
            new_query_cols,
            rows_to_project,
            "query_context_col_names",
            new_query_cols[1:],
        )
        rebuilt_target = rebuild_projection(
            target,
            source_table,
            new_target_cols,
            rows_to_project,
            "target_context_col_names",
            new_target_cols[1:],
        )
        if rebuilt_query is None or rebuilt_target is None:
            unresolved.append({"query_table_id": query_id, "target_table_id": target_id, "reason": "projection_has_no_rows"})
            continue
        if set(source_indices(rebuilt_query)) & set(source_indices(rebuilt_target)):
            unresolved.append({"query_table_id": query_id, "target_table_id": target_id, "reason": "projection_still_overlaps"})
            continue
        updated_queries[query_id] = rebuilt_query
        updated_targets[target_id] = rebuilt_target

    backup_dir = None
    if (updated_queries or updated_targets) and backup and not dry_run:
        backup_dir = backup_artifacts(dataset_dir)

    for _path, records in query_shards:
        for idx, record in enumerate(records):
            replacement = updated_queries.get(record.get("table_id"))
            if replacement:
                records[idx] = replacement
    for _path, records in target_shards:
        for idx, record in enumerate(records):
            replacement = updated_targets.get(record.get("table_id"))
            if replacement:
                records[idx] = replacement

    report = {
        "dataset_dir": str(dataset_dir),
        "dry_run": dry_run,
        "backup_dir": str(backup_dir) if backup_dir else "",
        "qrels": len(qrels),
        "already_distinct_pairs": already_distinct,
        "repaired_pairs": len(updated_targets),
        "unresolved_pairs": len(unresolved),
        "skipped_pairs": skipped,
        "unresolved": unresolved,
    }
    if not dry_run:
        write_sharded_records(query_shards)
        write_sharded_records(target_shards)
        write_json(dataset_dir / "mm_joinability_context_repair_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair MM joinability query/target context overlap without model calls.")
    parser.add_argument("--dataset_dir", required=True, help="Existing build_mm_joinability_dataset output directory.")
    parser.add_argument("--no_backup", dest="backup", action="store_false", help="Do not copy overwritten artifacts before repair.")
    parser.add_argument("--dry_run", action="store_true", help="Report what would be repaired without writing files.")
    parser.set_defaults(backup=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = repair_dataset(args.dataset_dir, backup=args.backup, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
