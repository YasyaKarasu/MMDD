#!/usr/bin/env python
"""Generate the page_url-relocation plan for the wdc200k dataset.

Scope, as decided 2026-09-20: only the ``page_url`` column moves.  The
schema.org ``url`` / ``logo`` / ``link`` family is a nested object rather than a
URL string and is deliberately left alone.

The plan has three parts:

* ``targets_to_drop.txt`` -- data-lake targets removed outright, because either
  ``page_url`` is the target's join column (the join dies with the column) or
  removing it would leave the target with at most one column.
* ``targets_to_strip_page_url.txt`` -- targets that keep their join column and
  lose only the ``page_url`` column.
* ``queries_to_delete_orphaned.txt`` -- queries whose every positive target is
  in the drop set.  They are deleted rather than kept as negatives: a query with
  no positive would score as a retrieval failure for every model.

``queries_rerun_autocheck.txt`` / ``queries_reuse_autocheck.txt`` split the
surviving queries by whether their visible row changes (it gains a ``page_url``
cell) -- only that half needs new auto-check calls.

This is read-only: it inspects the dataset and writes the plan, never the data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PAGE_URL = "page_url"

# The lake stores compact separators, the (already rewritten) query tables store
# the defaults.  One regex per shape rather than one that guesses.
RE_COLS_COMPACT = re.compile(r'"columns":\[([^\]]*)\]')
RE_COLS_SPACED = re.compile(r'"columns":\s*\[([^\]]*)\]')
RE_NAME_COMPACT = re.compile(r'"column_name":"((?:[^"\\]|\\.)*)"')
RE_NAME_SPACED = re.compile(r'"column_name":\s*"((?:[^"\\]|\\.)*)"')
RE_OID_COMPACT = re.compile(r'"object_id":"((?:[^"\\]|\\.)*)"')
RE_OID_SPACED = re.compile(r'"object_id":\s*"((?:[^"\\]|\\.)*)"')
RE_JCN_COMPACT = re.compile(r'"join_col_name":"((?:[^"\\]|\\.)*)"')


def shard_paths(dataset_dir: Path, manifest: dict[str, Any], artifact: str) -> list[Path]:
    item = manifest.get("artifacts", {}).get(artifact)
    if not item or not item.get("shards"):
        raise KeyError(f"artifact {artifact!r} is missing from the manifest")
    return [dataset_dir / shard["path"] for shard in item["shards"]]


def classify_targets(dataset_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Whether each data-lake target carries / joins on page_url, and its fate."""
    targets: dict[str, dict[str, Any]] = {}
    for path in shard_paths(dataset_dir, manifest, "data_lake_tables"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = RE_COLS_COMPACT.search(line)
                if not match:
                    # raw_data_lake_table stubs carry no inline columns.
                    continue
                names = [name.casefold() for name in RE_NAME_COMPACT.findall(match.group(1))]
                join_match = RE_JCN_COMPACT.search(line)
                join_name = (join_match.group(1) if join_match else "").casefold()
                has_page = PAGE_URL in names
                join_page = join_name == PAGE_URL
                targets[RE_OID_COMPACT.search(line).group(1)] = {
                    "columns": len(names),
                    "has_page_url": has_page,
                    "join_is_page_url": join_page,
                    "columns_after": len(names) - (1 if has_page else 0),
                }
    return targets


def query_rows(dataset_dir: Path, manifest: dict[str, Any]) -> dict[str, bool]:
    """Whether each query table already carries a visible page_url column."""
    has_page: dict[str, bool] = {}
    for path in shard_paths(dataset_dir, manifest, "query_tables"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = RE_COLS_SPACED.search(line)
                if not match:
                    continue
                names = [name.casefold() for name in RE_NAME_SPACED.findall(match.group(1))]
                has_page[RE_OID_SPACED.search(line).group(1)] = PAGE_URL in names
    return has_page


def plan(dataset_dir: Path) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"dataset is incomplete: {dataset_dir}")

    targets = classify_targets(dataset_dir, manifest)
    q_has_page = query_rows(dataset_dir, manifest)

    drop, strip, reasons = set(), set(), Counter()
    for target_id, info in targets.items():
        if info["join_is_page_url"]:
            drop.add(target_id)
            reasons["join column is page_url"] += 1
        elif info["columns_after"] <= 1:
            drop.add(target_id)
            reasons["would be left with <=1 column"] += 1
        elif info["has_page_url"]:
            strip.add(target_id)
            reasons["page_url stripped, table kept"] += 1
        else:
            reasons["untouched"] += 1

    positives: dict[str, set[str]] = defaultdict(set)
    split: dict[str, str] = {}
    with (dataset_dir / "qrels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = record["query_table_id"]
            positives[query_id].add(record["target_table_id"])
            split[query_id] = record.get("split", "")

    orphaned = {q for q, ts in positives.items() if ts and ts <= drop}
    partial = {q for q, ts in positives.items() if ts & drop and not ts <= drop}
    surviving = set(positives) - orphaned

    rerun, reuse = set(), set()
    for query_id in surviving:
        if q_has_page.get(query_id):
            reuse.add(query_id)
        elif any(
            targets.get(t, {}).get("has_page_url")
            for t in positives[query_id]
            # A dropped target cannot donate its column, so it must not put the
            # query on the rerun list: the query row would gain nothing while the
            # auto-check key still changed.
            if t not in drop
        ):
            rerun.add(query_id)

    pairs_total = sum(len(ts) for ts in positives.values())
    pairs_lost = sum(1 for ts in positives.values() for t in ts if t in drop)

    by_split = {}
    for name in ("train", "dev", "test"):
        total = sum(1 for q in positives if split[q] == name)
        lost = sum(1 for q in orphaned if split[q] == name)
        by_split[name] = {"orphaned": lost, "total": total, "kept": total - lost}

    return {
        "targets": {
            "total": len(targets),
            "drop": len(drop),
            "drop_reasons": {k: reasons[k] for k in ("join column is page_url",
                                                     "would be left with <=1 column")},
            "strip": len(strip),
            "untouched": reasons["untouched"],
            "kept": len(targets) - len(drop),
        },
        "queries": {
            "total": len(positives),
            "orphaned": len(orphaned),
            "partial": len(partial),
            "surviving": len(surviving),
            "rerun_autocheck": len(rerun),
            "reuse_autocheck": len(reuse),
            "by_split": by_split,
        },
        "qrels": {"pairs_total": pairs_total, "pairs_lost": pairs_lost},
        "data_lake": {
            "total_before": 138635 + len(targets),
            "total_after": 138635 + len(targets) - len(drop),
        },
        "ids": {"drop": drop, "strip": strip, "orphaned": orphaned,
                "rerun": rerun, "reuse": reuse},
    }


def write_ids(path: Path, ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in sorted(ids):
            handle.write(value + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir.parent / "MMDD_PAGEURL_RELOCATION_20260920"
    result = plan(dataset_dir)
    ids = result.pop("ids")

    lists = out_dir / "lists"
    write_ids(lists / "targets_to_drop.txt", ids["drop"])
    write_ids(lists / "targets_to_strip_page_url.txt", ids["strip"])
    write_ids(lists / "queries_to_delete_orphaned.txt", ids["orphaned"])
    write_ids(lists / "queries_rerun_autocheck.txt", ids["rerun"])
    write_ids(lists / "queries_reuse_autocheck.txt", ids["reuse"])

    result["dataset_dir"] = str(dataset_dir)
    result["lists_dir"] = str(lists)
    (out_dir / "plan.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
