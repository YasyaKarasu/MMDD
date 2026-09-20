#!/usr/bin/env python
"""Apply human recovery verdicts to a built joinability dataset.

The quality checker (``mm_joinability_dataset_checker.py``) lets a person mark
each recovery reasonable or unreasonable.  This turns those marks into a dataset:

1. drop every recovery marked unreasonable;
2. recompute each implicit qrel's recovered-row count from what is left, and drop
   the qrel when it no longer reaches its own ``required_recovered_rows``;
3. drop queries left without a qrel, and data-lake tables nothing points at;
4. leave explicit (visible-column) joins alone -- they have no recoveries, so a
   recovery verdict says nothing about them.

The input dataset is never modified: everything is written to ``--output-dir``,
and ``review_apply_report.json`` records what went and why.  Unmarked recoveries
are kept, so a partial review narrows the dataset only where a person said so.

Usage::

    python scripts_old/apply_recovery_review.py \\
        --dataset-dir output/abebooks_joinability_10row \\
        --review-db output/abebooks_joinability_10row/.abebooks_human_review_quality_checker.sqlite3 \\
        --output-dir output/abebooks_joinability_10row_reviewed
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mm_joinability_dataset_checker import QualityReviewStore  # noqa: E402
from mmdd_dataset.utils import clean_text, write_json, write_jsonl  # noqa: E402

#: Artifacts written as ``<name>.jsonl`` or ``<name>/part-*.jsonl``.
SHARDED_ARTIFACTS = (
    "query_tables",
    "data_lake_tables",
    "evidence_recoveries",
    "bridge_assets",
    "source_tables",
    "entities",
    "table_asset_links",
    "attribute_extractions",
)
SINGLE_FILES = ("splits.json", "table_queryability_decisions.jsonl")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--dataset-dir", required=True)
    result.add_argument("--review-db", required=True,
                        help="the sqlite the quality checker wrote")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--drop-unreviewed", action="store_true",
                        help="also drop recoveries nobody marked; default keeps them, "
                             "so a partial review only narrows what a person judged")
    return result


def read_paths(root: Path, name: str) -> list[Path]:
    """Both layouts: a single ``<name>.jsonl`` or a ``<name>/`` of shards."""
    flat = root / f"{name}.jsonl"
    if flat.exists():
        return [flat]
    directory = root / name
    if directory.is_dir():
        return sorted(directory.glob("*.jsonl"))
    return []


def read_records(root: Path, name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for path in read_paths(root, name)
            for line in path.open() if line.strip()]


def sharded(root: Path, name: str) -> bool:
    return not (root / f"{name}.jsonl").exists() and (root / name).is_dir()


def write_records(root: Path, name: str, records: Iterable[dict[str, Any]],
                  *, one_file: bool) -> None:
    if one_file:
        write_jsonl(root / f"{name}.jsonl", list(records))
        return
    directory = root / name
    if directory.exists():
        shutil.rmtree(directory)
    write_jsonl(directory / "part-00000.jsonl", list(records))


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.dataset_dir).resolve()
    target = Path(args.output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    store = QualityReviewStore(Path(args.review_db).resolve())
    verdicts = {clean_text(row["recovery_id"]): clean_text(row["verdict"])
                for row in store.export_recovery_rows()}

    recoveries = read_records(source, "evidence_recoveries")
    kept_recoveries: list[dict[str, Any]] = []
    dropped_recoveries: list[dict[str, Any]] = []
    unreviewed = 0
    for recovery in recoveries:
        verdict = verdicts.get(clean_text(recovery.get("recovery_id")), "")
        if not verdict:
            unreviewed += 1
            if args.drop_unreviewed:
                dropped_recoveries.append({"recovery_id": recovery.get("recovery_id"),
                                           "reason": "unreviewed"})
                continue
        elif verdict == "unreasonable":
            dropped_recoveries.append({"recovery_id": recovery.get("recovery_id"),
                                       "reason": "marked_unreasonable"})
            continue
        kept_recoveries.append(recovery)

    # Which query rows still have a recovery, per (query, target).
    rows_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for recovery in kept_recoveries:
        rows_by_pair[(clean_text(recovery["query_table_id"]),
                      clean_text(recovery["target_table_id"]))
                     ].add(str(recovery["query_row_id"]))

    qrels = read_records(source, "qrels")
    kept_qrels: list[dict[str, Any]] = []
    dropped_qrels: list[dict[str, Any]] = []
    for qrel in qrels:
        hidden = dict(qrel.get("join_attribute") or {})
        if hidden.get("role") != "model_recoverable_join_column":
            # An explicit join has no recoveries behind it; a recovery verdict
            # cannot speak to it either way.
            kept_qrels.append(qrel)
            continue
        key = (clean_text(qrel["query_table_id"]), clean_text(qrel["target_table_id"]))
        recovered = len(rows_by_pair.get(key, set()))
        required = int(hidden.get("required_recovered_rows") or 0)
        if recovered < required:
            dropped_qrels.append({
                "query_table_id": key[0], "target_table_id": key[1],
                "column_name": hidden.get("column_name"),
                "recovered_rows": recovered, "required_recovered_rows": required,
                "reason": "below_required_recovered_rows",
            })
            continue
        eligible = int(hidden.get("eligible_rows") or 0)
        kept_qrels.append({
            **qrel,
            "join_attribute": {
                **hidden,
                "recovered_rows": recovered,
                "recovered_value_ratio": (recovered / eligible) if eligible else 0.0,
            },
        })

    kept_query_ids = {clean_text(qrel["query_table_id"]) for qrel in kept_qrels}
    kept_target_ids = {clean_text(qrel["target_table_id"]) for qrel in kept_qrels}

    query_tables = [record for record in read_records(source, "query_tables")
                    if clean_text(record.get("table_id")) in kept_query_ids]
    data_lake_tables = [record for record in read_records(source, "data_lake_tables")
                        if clean_text(record.get("table_id")) in kept_target_ids]

    one_file = not any(sharded(source, name) for name in SHARDED_ARTIFACTS)
    for name in SHARDED_ARTIFACTS:
        if name in {"query_tables", "data_lake_tables", "evidence_recoveries"}:
            continue
        records = read_records(source, name)
        if records or read_paths(source, name):
            write_records(target, name, records, one_file=one_file)
    write_records(target, "query_tables", query_tables, one_file=one_file)
    write_records(target, "data_lake_tables", data_lake_tables, one_file=one_file)
    write_records(target, "evidence_recoveries", kept_recoveries, one_file=one_file)
    write_jsonl(target / "qrels.jsonl", kept_qrels)
    for name in SINGLE_FILES:
        path = source / name
        if path.exists():
            shutil.copyfile(path, target / name)
    for name in ("dataset_manifest.json", "stats.json"):
        path = source / name
        if path.exists():
            shutil.copyfile(path, target / name)

    by_verdict = defaultdict(int)
    for verdict in verdicts.values():
        by_verdict[verdict] += 1
    report = {
        "dataset_dir": str(source),
        "output_dir": str(target),
        "review_db": str(Path(args.review_db).resolve()),
        "verdicts": dict(by_verdict),
        "recoveries": {"total": len(recoveries), "kept": len(kept_recoveries),
                       "dropped": len(dropped_recoveries), "unreviewed": unreviewed},
        "qrels": {"total": len(qrels), "kept": len(kept_qrels),
                  "dropped": len(dropped_qrels)},
        "query_tables": {"kept": len(query_tables)},
        "data_lake_tables": {"kept": len(data_lake_tables)},
        "dropped_recoveries": dropped_recoveries,
        "dropped_qrels": dropped_qrels,
    }
    write_json(target / "review_apply_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = build(args)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"dropped_recoveries", "dropped_qrels"}},
                     indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
