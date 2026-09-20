#!/usr/bin/env python
"""Remove the synthetic ``entity_url`` column from a published dataset in place.

The shared builder derives that column from the entity cell's ``wiki_title``, so
it only names a real page for a Wikipedia-shaped corpus.  WDC tables mint a
synthetic ``wdc_<hash>`` wiki title, and the column therefore holds a fabricated
``https://en.wikipedia.org/wiki/wdc_...`` URL.  Rebuilding the dataset without it
would change every ``query_id`` (the visible-query fingerprint covers these
cells) and invalidate the qrels, recoveries and downstream splits that reference
them, so this drops the trailing column and cell instead and rewrites the
manifest shard metadata that ``iter_dataset_artifact`` validates.

This is an offline projection repair: no network, no model calls.  It refuses to
touch a dataset whose query tables do not have exactly the expected shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from wdc200k_io import _json_dumps

ARTIFACT = "query_tables"
COLUMN_NAME = "entity_url"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Rewrite a shard in the exact format ``wdc200k_io`` published it in."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_json_dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def is_synthetic_entity_url(column: dict[str, Any]) -> bool:
    return (
        column.get("column_name") == COLUMN_NAME
        and int(column.get("source_column_index", 0)) == -1
    )


def strip_record(record: dict[str, Any]) -> bool:
    """Drop the trailing synthetic column and cell.  True if anything was removed.

    Raises when a query carries the column but not in the shape the builder
    emits, because that would mean this is not the artifact we think it is.
    """
    table_id = record.get("table_id")
    columns = record.get("columns") or []
    rows = record.get("rows") or []
    if not columns or not is_synthetic_entity_url(columns[-1]):
        if any(is_synthetic_entity_url(column) for column in columns):
            raise ValueError(
                f"{table_id}: synthetic {COLUMN_NAME} column is not the last column"
            )
        return False

    removed = 0
    for row in rows:
        cells = row.get("cells") or []
        if not cells or cells[-1].get("column_name") != COLUMN_NAME:
            raise ValueError(
                f"{table_id}: row {row.get('row_id')} does not end in a "
                f"synthetic {COLUMN_NAME} cell"
            )
        row["cells"] = cells[:-1]
        removed += 1
    if removed != len(rows):
        raise ValueError(f"{table_id}: only {removed} of {len(rows)} rows matched")
    record["columns"] = columns[:-1]
    return True


def artifact_shards(
    dataset_dir: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    item = manifest.get("artifacts", {}).get(ARTIFACT)
    if not item:
        raise KeyError(f"artifact {ARTIFACT!r} is missing from the manifest")
    paths = list(item.get("shards", []))
    if not paths:
        raise ValueError(f"artifact {ARTIFACT!r} has no shards")
    return paths


def backup_artifacts(dataset_dir: Path) -> Path:
    backup_root = dataset_dir / "backups"
    backup_root.mkdir(exist_ok=True)
    name = f"{ARTIFACT}_strip_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_dir = backup_root / name
    suffix = 1
    while backup_dir.exists():
        suffix += 1
        backup_dir = backup_root / f"{name}_{suffix}"
    backup_dir.mkdir()
    shutil.copytree(dataset_dir / ARTIFACT, backup_dir / ARTIFACT)
    shutil.copy2(dataset_dir / "dataset_manifest.json", backup_dir)
    return backup_dir


def strip_dataset(
    dataset_dir: Path | str,
    *,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"dataset is incomplete: {dataset_dir}")

    shards = artifact_shards(dataset_dir, manifest)
    stripped_records = 0
    already_absent = 0
    # Pass 1 validates every shard before anything is written, so a dataset that
    # does not match the expected shape is left untouched.
    pending: dict[str, list[dict[str, Any]]] = {}
    for shard in shards:
        path = dataset_dir / shard["path"]
        records = read_jsonl(path)
        if len(records) != int(shard["records"]):
            raise ValueError(
                f"{shard['path']}: read {len(records)} records, manifest says "
                f"{shard['records']}"
            )
        for record in records:
            if strip_record(record):
                stripped_records += 1
            else:
                already_absent += 1
        pending[shard["path"]] = records

    report: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "artifact": ARTIFACT,
        "column_name": COLUMN_NAME,
        "dry_run": dry_run,
        "shards": len(shards),
        "stripped_query_tables": stripped_records,
        "already_absent_query_tables": already_absent,
        "backup_dir": "",
    }
    if dry_run or not stripped_records:
        return report

    if backup:
        report["backup_dir"] = str(backup_artifacts(dataset_dir))

    for shard in shards:
        path = dataset_dir / shard["path"]
        write_jsonl(path, pending[shard["path"]])
        stat = path.stat()
        shard["bytes"] = stat.st_size
        shard["mtime_ns"] = stat.st_mtime_ns
        shard["sha256"] = sha256_path(path)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "entity_url_strip_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strip the synthetic entity_url column from a published dataset's "
            "query tables without changing any query_id."
        )
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument(
        "--no_backup",
        dest="backup",
        action="store_false",
        help="Do not copy the query tables and manifest before rewriting them.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate and report without writing anything.",
    )
    parser.set_defaults(backup=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = strip_dataset(
        args.dataset_dir, backup=args.backup, dry_run=args.dry_run
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
