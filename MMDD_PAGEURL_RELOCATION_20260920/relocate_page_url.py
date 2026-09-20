#!/usr/bin/env python
"""Relocate the ``page_url`` column out of the wdc200k data-lake targets.

Three edits, all offline (no network, no model calls):

1. every target whose join column is ``page_url``, and every target that would
   be left with at most one column once ``page_url`` is removed, is deleted;
2. ``page_url`` is removed from the remaining targets and appended to the query
   tables that do not already carry it -- query and target reach the same source
   table, so the values come from that target's own ``page_url`` cells, matched
   by ``source_row_id``;
3. queries whose every positive target was deleted are removed, together with
   their qrels rows, because a query with no positive scores as a retrieval
   failure for every model.

The new query column is appended last and is deliberately *not* listed in
``query_context_col_names``: that is exactly the slot the synthetic
``entity_url`` column occupied before ``strip_entity_url_column.py`` removed it,
so the row differs from the pre-strip state by one cell's name and value only.

Every shard is rewritten in the byte-style its publisher used, so untouched
records round-trip identically and the diff is only the intended change.  The
plan is read-only and the lake is streamed one shard at a time: at no point is
more than a single shard of it held in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PAGE_URL = "page_url"
PLAN_VERSION = "pageurl-relocation-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_lake(record: dict[str, Any]) -> str:
    """Match the data-lake publisher: sorted keys, compact separators."""
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def serialize_query(record: dict[str, Any]) -> str:
    """Match the query-table publisher (and the entity_url strip) exactly."""
    return json.dumps(record, ensure_ascii=False)


def shard_entries(dataset_dir: Path, manifest: dict[str, Any], artifact: str) -> list[dict[str, Any]]:
    item = manifest.get("artifacts", {}).get(artifact)
    if not item or not item.get("shards"):
        raise KeyError(f"artifact {artifact!r} is missing from the manifest")
    return list(item["shards"])


def read_lines(path: Path, expected: int | None = None) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if expected is not None and len(lines) != expected:
        raise ValueError(f"{path}: {len(lines)} records, manifest says {expected}")
    return lines


def write_shard(path: Path, lines: list[str]) -> dict[str, Any]:
    mode = path.stat().st_mode
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_path(path)}


def load_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def page_url_position(columns: list[dict[str, Any]]) -> int | None:
    for index, column in enumerate(columns):
        if str(column.get("column_name", "")).casefold() == PAGE_URL:
            return index
    return None


def strip_target_page_url(record: dict[str, Any]) -> None:
    """Drop the page_url column and cell, keeping cells aligned with columns."""
    columns = record.get("columns") or []
    position = page_url_position(columns)
    if position is None:
        raise ValueError(f"{record.get('object_id')}: expected a page_url column")
    record["columns"] = [c for i, c in enumerate(columns) if i != position]
    for index, column in enumerate(record["columns"]):
        column["column_index"] = index
    for row in record.get("rows") or []:
        cells = row.get("cells") or []
        if len(cells) != len(columns):
            raise ValueError(f"{record.get('object_id')}: row {row.get('row_id')} is misaligned")
        row["cells"] = [c for i, c in enumerate(cells) if i != position]
        for index, cell in enumerate(row["cells"]):
            cell["column_index"] = index
    context = record.get("target_context_col_names")
    if isinstance(context, list):
        record["target_context_col_names"] = [
            name for name in context if str(name).casefold() != PAGE_URL
        ]
    sources = record.get("source_column_indices")
    if isinstance(sources, list):
        if len(sources) != len(columns):
            raise ValueError(f"{record.get('object_id')}: source_column_indices misaligned")
        record["source_column_indices"] = [s for i, s in enumerate(sources) if i != position]


def query_page_url_cell(source_cell: dict[str, Any], column_index: int) -> dict[str, Any]:
    """A query-shaped cell, in the key order the query publisher emits."""
    return {
        "column_index": column_index,
        "column_name": PAGE_URL,
        "has_wiki_link": bool(source_cell.get("has_wiki_link")),
        "raw": source_cell.get("raw", ""),
        "source_column_index": source_cell.get("source_column_index"),
        "text": source_cell.get("text", ""),
        "wiki_title": source_cell.get("wiki_title"),
    }


def add_page_url_to_query(
    record: dict[str, Any],
    url_by_row: dict[tuple[str, int], dict[str, Any]],
) -> tuple[int, int, bool]:
    """Append page_url to one query table.

    Returns ``(filled, missing, all_empty)``.  A row whose ``source_row_id`` is
    absent from every surviving target is one the target never materialised --
    in practice a row whose join column is empty, so it cannot take part in the
    join anyway.  It gets an empty cell; an empty cell is dropped by both
    ``query_visible_row_attributes`` and the auto-check key canonicaliser, so it
    is inert rather than misleading.
    """
    columns = record.get("columns") or []
    if page_url_position(columns) is not None:
        raise ValueError(f"{record.get('object_id')}: already carries page_url")
    source_table = record.get("source_table_id", "")
    position = len(columns)
    source_index = None
    filled = missing = 0
    for row in record.get("rows") or []:
        source_cell = url_by_row.get((source_table, int(row["source_row_id"])))
        if source_cell is None:
            missing += 1
            source_cell = {}
        else:
            filled += 1
            if source_index is None:
                source_index = source_cell.get("source_column_index")
        row["cells"] = list(row.get("cells") or []) + [
            query_page_url_cell(source_cell, position)
        ]
    record["columns"] = list(columns) + [{
        "column_index": position,
        "column_name": PAGE_URL,
        "source_column_index": source_index,
    }]
    sources = record.get("source_column_indices")
    if isinstance(sources, list):
        record["source_column_indices"] = list(sources) + [source_index]
    return filled, missing, filled == 0


def relocate(
    dataset_dir: Path,
    plan_dir: Path,
    *,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    lists = plan_dir / "lists"
    drop = load_ids(lists / "targets_to_drop.txt")
    strip = load_ids(lists / "targets_to_strip_page_url.txt")
    orphan = load_ids(lists / "queries_to_delete_orphaned.txt")
    gaining = load_ids(lists / "queries_rerun_autocheck.txt")
    if not drop or not strip or not orphan:
        raise ValueError(f"plan lists look empty; check {lists}")

    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"dataset is incomplete: {dataset_dir}")
    query_shards = shard_entries(dataset_dir, manifest, "query_tables")
    lake_shards = shard_entries(dataset_dir, manifest, "data_lake_tables")

    # ---- pass 1: hold the query tables (592 MB) and learn which source rows need page_url
    query_lines: dict[str, list[str]] = {}
    needed_rows: dict[str, set[int]] = defaultdict(set)
    kept_queries = 0
    for shard in query_shards:
        lines = read_lines(dataset_dir / shard["path"], int(shard["records"]))
        query_lines[shard["path"]] = lines
        for line in lines:
            record = json.loads(line)
            query_id = record["object_id"]
            if query_id in orphan:
                continue
            kept_queries += 1
            if query_id in gaining:
                source_table = record.get("source_table_id", "")
                for row in record.get("rows") or []:
                    needed_rows[source_table].add(int(row["source_row_id"]))

    # ---- pass 2: stream the lake, lifting page_url cells and counting the edits
    url_by_row: dict[tuple[str, int], dict[str, Any]] = {}
    dropped = stripped = kept_lake = 0
    for shard in lake_shards:
        for line in read_lines(dataset_dir / shard["path"], int(shard["records"])):
            if '"columns"' not in line:
                kept_lake += 1
                continue
            record = json.loads(line)
            target_id = record["object_id"]
            if target_id in drop:
                dropped += 1
                continue
            kept_lake += 1
            if target_id not in strip:
                continue
            stripped += 1
            wanted = needed_rows.get(record.get("source_table_id", ""))
            if not wanted:
                continue
            position = page_url_position(record.get("columns") or [])
            if position is None:
                raise ValueError(f"{target_id}: in the strip list without a page_url column")
            for row in record.get("rows") or []:
                source_row = int(row["source_row_id"])
                key = (record.get("source_table_id", ""), source_row)
                if source_row in wanted and key not in url_by_row:
                    url_by_row[key] = row["cells"][position]

    # ---- pass 3: build the query output in memory, then stream the lake to disk
    filled = missing = 0
    all_empty: list[str] = []
    query_out: dict[str, list[str]] = {}
    for shard in query_shards:
        out: list[str] = []
        for line in query_lines[shard["path"]]:
            record = json.loads(line)
            if record["object_id"] in orphan:
                continue
            if record["object_id"] in gaining:
                f, m, empty = add_page_url_to_query(record, url_by_row)
                filled += f
                missing += m
                if empty:
                    all_empty.append(record["object_id"])
            out.append(serialize_query(record) + "\n")
        query_out[shard["path"]] = out
    del query_lines

    query_counts: Counter[str] = Counter()
    for lines in query_out.values():
        for line in lines:
            query_counts[json.loads(line).get("split", "")] += 1

    report: dict[str, Any] = {
        "plan_version": PLAN_VERSION,
        "dataset_dir": str(dataset_dir),
        "plan_dir": str(plan_dir),
        "dry_run": dry_run,
        "targets": {"dropped": dropped, "stripped": stripped, "kept": kept_lake},
        "queries": {"kept": kept_queries, "orphaned_removed": len(orphan),
                    "gained_page_url": len(gaining)},
        "cells": {"filled": filled, "missing_empty": missing,
                  "queries_with_all_empty_page_url": len(all_empty)},
        "queries_with_all_empty_page_url_sample": sorted(all_empty)[:10],
        "by_split": dict(query_counts),
        "backup_dir": "",
    }

    if not dry_run:
        if backup:
            backup_root = dataset_dir / "backups"
            backup_root.mkdir(exist_ok=True)
            name = f"pageurl_relocation_{time.strftime('%Y%m%d_%H%M%S')}"
            target = backup_root / name
            suffix = 1
            while target.exists():
                suffix += 1
                target = backup_root / f"{name}_{suffix}"
            target.mkdir()
            for artifact in ("data_lake_tables", "query_tables"):
                shutil.copytree(dataset_dir / artifact, target / artifact)
            for filename in ("qrels.jsonl", "splits.json", "stats.json", "dataset_manifest.json"):
                shutil.copy2(dataset_dir / filename, target / filename)
            report["backup_dir"] = str(target)

        for shard in lake_shards:
            path = dataset_dir / shard["path"]
            out: list[str] = []
            for line in read_lines(path, int(shard["records"])):
                if '"columns"' not in line:
                    out.append(line if line.endswith("\n") else line + "\n")
                    continue
                record = json.loads(line)
                if record["object_id"] in drop:
                    continue
                if record["object_id"] in strip:
                    strip_target_page_url(record)
                out.append(serialize_lake(record) + "\n")
            shard.update(write_shard(path, out))

        qrels_out: list[str] = []
        pairs_kept = pairs_lost = 0
        for line in read_lines(dataset_dir / "qrels.jsonl"):
            record = json.loads(line)
            if record["query_table_id"] in orphan or record["target_table_id"] in drop:
                pairs_lost += 1
                continue
            pairs_kept += 1
            qrels_out.append(serialize_query(record) + "\n")
        (dataset_dir / "qrels.jsonl").write_text("".join(qrels_out), encoding="utf-8")
        report["qrels"] = {"pairs_kept": pairs_kept, "pairs_lost": pairs_lost}

    if not dry_run:
        for shard in query_shards:
            shard.update(write_shard(dataset_dir / shard["path"], query_out[shard["path"]]))
        manifest["artifacts"]["query_tables"]["total_records"] = kept_queries
        manifest["artifacts"]["data_lake_tables"]["total_records"] = kept_lake

        splits_path = dataset_dir / "splits.json"
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        splits["query_table_counts"] = {k: query_counts.get(k, 0) for k in ("dev", "test", "train")}
        splits["data_lake_table_count"] = kept_lake
        splits_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        stats_path = dataset_dir / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["query_tables"] = kept_queries
        stats["data_lake_tables"] = kept_lake
        stats["qrels"] = report["qrels"]["pairs_kept"]
        stats.setdefault("notes", []).append(
            "page_url relocated out of data-lake targets into query rows "
            f"({PLAN_VERSION}); see pageurl_relocation_report.json"
        )
        stats_path.write_text(json.dumps(stats, ensure_ascii=False) + "\n", encoding="utf-8")

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (dataset_dir / "pageurl_relocation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--plan_dir", default="MMDD_PAGEURL_RELOCATION_20260920")
    parser.add_argument("--no_backup", dest="backup", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    parser.set_defaults(backup=True)
    args = parser.parse_args()
    report = relocate(
        Path(args.dataset_dir), Path(args.plan_dir),
        backup=args.backup, dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
