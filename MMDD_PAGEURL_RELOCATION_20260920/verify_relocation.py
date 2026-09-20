#!/usr/bin/env python
"""Independently verify a completed page_url relocation against its backup.

The strong check is the shard diff: every lake line must be byte-identical to
its backup counterpart, except the ones the plan says were dropped outright and
the ones whose only change is losing the ``page_url`` column.  Anything else is
an unintended edit and is reported with its object id.

Run after ``relocate_page_url.py``:

    python3 verify_relocation.py \
      --dataset_dir output_wdc_webtable_200000_qwen35_local_autocheck_v9 \
      --plan_dir MMDD_PAGEURL_RELOCATION_20260920
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PAGE_URL = "page_url"


def load_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def page_url_position(columns: list[dict[str, Any]]) -> int | None:
    for index, column in enumerate(columns):
        if str(column.get("column_name", "")).casefold() == PAGE_URL:
            return index
    return None


def lines_of(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--plan_dir", default="MMDD_PAGEURL_RELOCATION_20260920")
    parser.add_argument("--backup", default="")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    lists = Path(args.plan_dir) / "lists"
    drop = load_ids(lists / "targets_to_drop.txt")
    strip = load_ids(lists / "targets_to_strip_page_url.txt")
    orphan = load_ids(lists / "queries_to_delete_orphaned.txt")

    if args.backup:
        backup = Path(args.backup)
    else:
        report = json.loads((dataset_dir / "pageurl_relocation_report.json").read_text())
        backup = Path(report["backup_dir"])
    print(f"backup: {backup}")

    failures: list[str] = []
    summary: dict[str, Any] = {}

    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    if manifest.get("complete") is not True:
        failures.append("manifest no longer marked complete")

    # ---- 1. shard-by-shard diff against the backup
    for artifact, is_lake in (("data_lake_tables", True), ("query_tables", False)):
        changed_stripped = changed_query_gain = removed = identical = unexpected = 0
        ids_out: set[str] = set()
        for shard in manifest["artifacts"][artifact]["shards"]:
            before = lines_of(backup / shard["path"])
            after = lines_of(dataset_dir / shard["path"])
            if len(after) != int(shard["records"]):
                failures.append(f"{shard['path']}: {len(after)} records, manifest says {shard['records']}")
            after_by_id: dict[str, str] = {}
            for line in after:
                obj = json.loads(line)
                ids_out.add(obj["object_id"])
                after_by_id[obj["object_id"]] = line
            for line in before:
                obj = json.loads(line)
                object_id = obj["object_id"]
                if object_id not in after_by_id:
                    removed += 1
                    if is_lake and object_id not in drop:
                        unexpected += 1
                        failures.append(f"lake record vanished but is not on the drop list: {object_id}")
                    if not is_lake and object_id not in orphan:
                        unexpected += 1
                        failures.append(f"query record vanished but is not an orphan: {object_id}")
                    continue
                if after_by_id[object_id] == line:
                    identical += 1
                    continue
                # changed: classify
                if is_lake:
                    if object_id not in strip:
                        unexpected += 1
                        failures.append(f"lake record changed but is not on the strip list: {object_id}")
                    else:
                        changed_stripped += 1
                else:
                    changed_query_gain += 1
        summary[artifact] = {
            "identical": identical, "removed": removed,
            "changed": changed_stripped if is_lake else changed_query_gain,
            "unexpected": unexpected, "objects_out": len(ids_out),
        }

    # ---- 2. no target may still carry page_url
    targets_with_page_url = 0
    targets_bad_alignment = 0
    lake_count = 0
    for shard in manifest["artifacts"]["data_lake_tables"]["shards"]:
        for line in lines_of(dataset_dir / shard["path"]):
            if '"columns"' not in line:
                lake_count += 1
                continue
            record = json.loads(line)
            lake_count += 1
            columns = record["columns"]
            if page_url_position(columns) is not None:
                targets_with_page_url += 1
            for row in record["rows"]:
                if len(row["cells"]) != len(columns) or any(
                    cell["column_index"] != i for i, cell in enumerate(row["cells"])
                ):
                    targets_bad_alignment += 1
                    break
    if targets_with_page_url:
        failures.append(f"{targets_with_page_url} targets still carry page_url")
    if targets_bad_alignment:
        failures.append(f"{targets_bad_alignment} targets have misaligned cells")
    summary["targets"] = {"total": lake_count, "still_with_page_url": targets_with_page_url,
                          "misaligned": targets_bad_alignment}

    # ---- 3. query side
    queries = 0
    gained = 0
    misaligned = 0
    not_last = 0
    for shard in manifest["artifacts"]["query_tables"]["shards"]:
        for line in lines_of(dataset_dir / shard["path"]):
            record = json.loads(line)
            queries += 1
            columns = record["columns"]
            position = page_url_position(columns)
            for row in record["rows"]:
                if len(row["cells"]) != len(columns):
                    misaligned += 1
                    break
            if position is not None:
                if position != len(columns) - 1:
                    not_last += 1
                for row in record["rows"]:
                    if row["cells"][position]["column_name"].casefold() != PAGE_URL:
                        misaligned += 1
                        break
    if misaligned:
        failures.append(f"{misaligned} query tables have misaligned cells")
    if not_last:
        failures.append(f"{not_last} query tables carry page_url somewhere other than last")
    summary["queries"] = {"total": queries, "misaligned": misaligned, "page_url_not_last": not_last}

    # ---- 4. qrels references only survivors
    dangling = 0
    pairs = 0
    for line in (dataset_dir / "qrels.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        pairs += 1
        if record["query_table_id"] in orphan or record["target_table_id"] in drop:
            dangling += 1
    if dangling:
        failures.append(f"{dangling} qrels rows reference a deleted query or target")
    summary["qrels"] = {"pairs": pairs, "dangling": dangling}

    # ---- 5. splits / stats agree with what is on disk
    splits = json.loads((dataset_dir / "splits.json").read_text())
    stats = json.loads((dataset_dir / "stats.json").read_text())
    counts: Counter[str] = Counter()
    for shard in manifest["artifacts"]["query_tables"]["shards"]:
        for line in lines_of(dataset_dir / shard["path"]):
            counts[json.loads(line).get("split", "")] += 1
    for key, value in (("train", counts["train"]), ("dev", counts["dev"]), ("test", counts["test"])):
        if splits["query_table_counts"].get(key) != value:
            failures.append(f"splits.query_table_counts[{key!r}]={splits['query_table_counts'].get(key)} but {value} on disk")
    if splits["data_lake_table_count"] != lake_count:
        failures.append(f"splits.data_lake_table_count={splits['data_lake_table_count']} but {lake_count} on disk")
    if stats["query_tables"] != queries:
        failures.append(f"stats.query_tables={stats['query_tables']} but {queries} on disk")
    if stats["data_lake_tables"] != lake_count:
        failures.append(f"stats.data_lake_tables={stats['data_lake_tables']} but {lake_count} on disk")
    summary["splits"] = splits["query_table_counts"]
    summary["manifest_totals"] = {
        "query_tables": manifest["artifacts"]["query_tables"]["total_records"],
        "data_lake_tables": manifest["artifacts"]["data_lake_tables"]["total_records"],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for message in failures[:40]:
            print("  -", message)
        raise SystemExit(1)
    print("\nOK: relocation verified")


if __name__ == "__main__":
    main()
