#!/usr/bin/env python
"""Clean a built AbeBooks joinability dataset and re-split it.

Three passes over a reviewed dataset, in this order:

1. **Drop degenerate join columns.**  A join column whose values are one value
   repeated is not a join key: ``language`` is ``English`` on every row of the
   table, so a model that emits ``English`` scores a perfect recovery while
   knowing nothing, and an explicit query that shows the column hands the
   reader the answer.  Every qrel whose join column has a single distinct value
   is dropped, then the query it belonged to is kept only if another qrel still
   references it (a query whose hidden column is the constant one loses that
   chain, not necessarily itself).
2. **Rekey the lake and restore what the removal emptied.**  Every target that
   the surviving qrels no longer reference leaves the lake.  A source table
   whose *every* target left that way is put back whole -- the full column set
   and every row from ``source_tables`` -- so a table that the cleaning removed
   from the candidate pool is still searchable.  Neither step invents a
   recovery: a restored record carries its original columns and rows and no
   ``join_col``, because nothing was split along it any more.
3. **Balance and re-split.**  Explicit queries are subsampled to match the
   implicit count, then every source table is assigned whole to train/dev/test
   so no ``source_table_id`` spans a split (which ``splits.json`` forbids).

The input dataset is never modified: everything is written to ``--output-dir``
and ``clean_report.json`` records what went and why.

Usage::

    python scripts_old/clean_abebooks_joinability.py \\
        --dataset-dir abebooks_joinability_bal04_reviewed \\
        --output-dir output/abebooks_joinability_bal04_clean
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# Running this file directly puts ``scripts_old`` on ``sys.path``, not ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmdd_dataset.utils import (  # noqa: E402
    clean_text,
    get_cell,
    normalize,
    sanitize_cell_text,
    stable_hash,
    write_json,
    write_jsonl,
)

#: Written as ``<name>/part-*.jsonl`` when the source holds that layout.
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
IMPLICIT_ROLE = "model_recoverable_join_column"
EXPLICIT_ROLE = "visible_join_column"
SPLIT_NAMES = ("train", "dev", "test")
#: Query-share targets for train/dev/test, matching the shared builder's default.
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
#: A join column is degenerate when one value takes this share of its non-empty
#: rows.  ``1.0`` is the strict "exactly one distinct value" reading; the gate
#: the builder uses is ``MAX_VALUE_SHARE`` (0.5), which is looser.  Lowering this
#: argument removes more columns.
DEGENERATE_SHARE = 1.0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--dataset-dir", required=True,
                        help="the reviewed dataset to read; never modified")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--max-value-share", type=float, default=DEGENERATE_SHARE,
                        help="drop a join column whose modal value takes more than "
                             "this share of its non-empty rows (default: 1.0, i.e. "
                             "exactly one distinct value)")
    result.add_argument("--seed", type=int, default=13,
                        help="deterministic seed for the explicit subsample and "
                             "the split assignment (default: 13)")
    result.add_argument("--lake-dir", default="output/abebooks_lake_no_copy",
                        help="original lake, read only for the run's provenance record")
    return result


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

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


def qrel_role(qrel: dict[str, Any]) -> str:
    """The role as the readers spell it: the qrel records the query-side name."""
    role = clean_text((qrel.get("join_attribute") or {}).get("role"))
    return role


def modal_share(table: dict[str, Any], column_index: int) -> float:
    """Share of the column's non-empty rows taken by its most common value.

    The values are folded with the same ``normalize`` the builder compares
    with, so ``English``/``english`` count as one value here and in the gate
    that let the column through.
    """
    values = [
        normalize(clean_text(get_cell(row, column_index).get("text")))
        for row in table["rows"]
    ]
    values = [value for value in values if value]
    if not values:
        return 1.0
    return max(Counter(values).values()) / len(values)


# --------------------------------------------------------------------------
# the degeneracy pass
# --------------------------------------------------------------------------

def degenerate_join_pairs(
    qrels: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    max_value_share: float,
) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    """``(dropped (query, target) keys, one record per dropped pair)``."""
    dropped: set[tuple[str, str]] = set()
    detail: list[dict[str, Any]] = []
    for qrel in qrels:
        source_table_id = clean_text(qrel.get("source_table_id"))
        table = sources.get(source_table_id)
        if table is None:
            raise KeyError(f"qrel names an unknown source table: {source_table_id}")
        attribute = qrel.get("join_attribute") or {}
        column_index = int(attribute.get("source_column_index", -1))
        share = modal_share(table, column_index)
        # Exclusive: the default 1.0 means "one value takes every non-empty row",
        # so a constant column is dropped and a column with one row of variety
        # (share 0.9) is kept.  A non-strict comparison here would drop nothing.
        if share < max_value_share:
            continue
        dropped.add((clean_text(qrel["query_table_id"]),
                     clean_text(qrel["target_table_id"])))
        values = [
            normalize(clean_text(get_cell(row, column_index).get("text")))
            for row in table["rows"]
        ]
        values = [value for value in values if value]
        detail.append({
            "query_table_id": clean_text(qrel["query_table_id"]),
            "target_table_id": clean_text(qrel["target_table_id"]),
            "source_table_id": source_table_id,
            "column_name": clean_text(attribute.get("column_name")),
            "role": qrel_role(qrel),
            "distinct_values": len(set(values)),
            "modal_share": round(share, 4),
            "reason": "degenerate_join_column",
        })
    return dropped, detail


def stable_query_key(query_id: str, split: str, seed: int) -> tuple[str, str]:
    return stable_hash("clean-abebooks-explicit-balance", seed, split, query_id,
                       length=40), query_id


def select_explicit_queries(
    *,
    implicit_ids: set[str],
    explicit_ids: set[str],
    query_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> set[str]:
    """Keep exactly as many explicit queries as implicit ones, per split.

    Split sizes are not final yet, so the quota is the *global* implicit count:
    the split pass below is what turns that into per-split ratios.
    """
    needed = len(implicit_ids)
    if len(explicit_ids) <= needed:
        return set(explicit_ids)
    ordered = sorted(explicit_ids, key=lambda query_id: stable_query_key(
        query_id, clean_text(query_by_id[query_id].get("split")), seed))
    return set(ordered[:needed])


# --------------------------------------------------------------------------
# the lake pass
# --------------------------------------------------------------------------

def restore_record(table: dict[str, Any]) -> dict[str, Any]:
    """One source table put back into the lake whole.

    The column list and every row come from ``source_tables``; ``join_col`` is
    dropped rather than carried, because the cleaning removed the pairs that
    were split along it.  ``table_id`` is derived from the source id, which is
    why the report records the mapping instead of a caller relying on the id
    being stable across rebuilds.
    """
    source_table_id = clean_text(table["source_table_id"])
    columns = [
        {
            "column_index": int(column["column_index"]),
            "source_column_index": int(column["column_index"]),
            "column_name": clean_text(column["column_name"]),
        }
        for column in table["columns"]
    ]
    rows: list[dict[str, Any]] = []
    for source_row in table["rows"]:
        cells = [
            {
                **cell,
                "text": sanitize_cell_text(cell.get("text")),
                "source_column_index": int(cell["column_index"]),
            }
            for cell in source_row["cells"]
        ]
        rows.append({
            "row_id": len(rows),
            "source_row_id": source_row["row_id"],
            "cells": cells,
        })
    table_id = "restored_" + stable_hash(source_table_id, length=16)
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "role": "restored_data_lake_table",
        "source_table_id": source_table_id,
        "page_title": "",
        "caption": "",
        "section_title": "",
        "columns": columns,
        "rows": rows,
        "source_column_indices": [int(column["column_index"]) for column in table["columns"]],
        "source_row_indices": [row["row_id"] for row in table["rows"]],
        "provenance": {"builder": "clean_abebooks_joinability",
                       "restored_from": "source_tables"},
        "chain_id": None,
        "queryable_source_table": True,
    }


def synchronize_query_table(
    query: dict[str, Any],
    qrels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep a query's embedded bookkeeping aligned with the qrels it still has.

    A query can outlive some of its chains, so ``chain_ids``,
    ``target_table_ids`` and ``hidden_attributes`` are all recomputed from the
    surviving qrels.  The row projection is untouched: which rows a query shows
    is a property of the query, not of how many targets it still has.
    """
    target_ids = list(dict.fromkeys(
        clean_text(qrel.get("target_table_id")) for qrel in qrels
    ))
    chain_ids = list(dict.fromkeys(
        clean_text(qrel.get("chain_id")) for qrel in qrels
    ))
    result = {**query, "target_table_ids": target_ids, "chain_ids": chain_ids}
    if isinstance(query.get("chain_id"), str) and query["chain_id"] not in chain_ids:
        result["chain_id"] = chain_ids[0] if chain_ids else query["chain_id"]
    if not isinstance(query.get("hidden_attributes"), list):
        return result

    surviving = {
        (int(attribute.get("source_column_index", -1)),
         clean_text(attribute.get("column_name"))): attribute
        for qrel in qrels
        if qrel_role(qrel) == IMPLICIT_ROLE
        for attribute in [dict(qrel.get("join_attribute") or {})]
    }
    result["hidden_attributes"] = [
        surviving[key]
        for attribute in query["hidden_attributes"]
        for key in [(
            int(attribute.get("source_column_index", -1)),
            clean_text(attribute.get("column_name")),
        )]
        if key in surviving
    ]
    return result


# --------------------------------------------------------------------------
# the split pass
# --------------------------------------------------------------------------

def assign_splits(
    *,
    queries_by_source: dict[str, list[str]],
    seed: int,
) -> dict[str, str]:
    """Largest-first greedy packing of whole source tables onto 8:1:1.

    ``source_table_id`` is ``splits.json``'s ``split_key``, so a source table
    has to land in one split.  Requiring the *fraction of source tables* to be
    8:1:1 would undershoot the fraction of queries whenever the biggest tables
    land together, so the packing balances on query count instead and lets the
    table count follow.
    """
    total = sum(len(query_ids) for query_ids in queries_by_source.values())
    load = {split: 0 for split in SPLIT_NAMES}
    order = sorted(queries_by_source)
    # A seeded shuffle first, then a stable largest-first sort: equal-sized tables
    # are ordered by the seed rather than by the lake's own input order, so a
    # rebuild cannot depend on how the lake happened to be written.
    random.Random(seed).shuffle(order)
    order.sort(key=lambda source_id: -len(queries_by_source[source_id]))
    split_of: dict[str, str] = {}
    for source_id in order:
        count = len(queries_by_source[source_id])
        # Overfill is measured relative to each split's own target, not in
        # absolute queries.  Absolute error ties at the start (every load is
        # zero) and after any split reaches its target, so the largest tables
        # would all land in the first split the sort happens to try.
        chosen = min(
            SPLIT_NAMES,
            key=lambda split: ((load[split] + count) / (total * SPLIT_RATIOS[split]),
                               SPLIT_NAMES.index(split)),
        )
        split_of[source_id] = chosen
        load[chosen] += count
    return split_of


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.dataset_dir).resolve()
    target = Path(args.output_dir).resolve()
    if source == target:
        raise ValueError("--output-dir must differ from --dataset-dir")
    target.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)

    sources = {
        clean_text(record["source_table_id"]): record
        for record in read_records(source, "source_tables")
    }
    qrels_all = read_records(source, "qrels")
    query_records = read_records(source, "query_tables")
    query_by_id = {clean_text(record["table_id"]): record for record in query_records}
    lake_all = read_records(source, "data_lake_tables")
    lake_by_id = {clean_text(record["table_id"]): record for record in lake_all}

    # 1. degenerate join columns.
    dropped_pairs, dropped_detail = degenerate_join_pairs(
        qrels_all, sources, float(args.max_value_share))
    qrels = [
        qrel for qrel in qrels_all
        if (clean_text(qrel["query_table_id"]), clean_text(qrel["target_table_id"]))
        not in dropped_pairs
    ]

    # A query survives only while a qrel still names it.
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qrel in qrels:
        qrels_by_query[clean_text(qrel["query_table_id"])].append(qrel)
    missing = sorted(set(qrels_by_query) - query_by_id.keys())
    if missing:
        raise ValueError(f"qrels reference missing queries: {missing[:5]}")

    # 2. balance explicit against implicit before splitting, so the split pass
    #    divides a dataset that is already the size the caller asked for.
    roles = {
        query_id: {qrel_role(qrel) for qrel in query_qrels}
        for query_id, query_qrels in qrels_by_query.items()
    }
    mixed = sorted(query_id for query_id, role_set in roles.items()
                   if IMPLICIT_ROLE in role_set and EXPLICIT_ROLE in role_set)
    if mixed:
        raise ValueError("a query cannot mix implicit and explicit qrels: "
                         + ", ".join(mixed[:5]))
    implicit_ids = {query_id for query_id, role_set in roles.items()
                    if IMPLICIT_ROLE in role_set}
    explicit_ids = {query_id for query_id, role_set in roles.items()
                    if EXPLICIT_ROLE in role_set}
    selected_explicit = select_explicit_queries(
        implicit_ids=implicit_ids,
        explicit_ids=explicit_ids,
        query_by_id=query_by_id,
        seed=seed,
    )
    dropped_explicit_ids = sorted(explicit_ids - selected_explicit)
    dropped_explicit_qrels = [
        qrel for qrel in qrels
        if clean_text(qrel["query_table_id"]) in dropped_explicit_ids
    ]
    qrels = [
        qrel for qrel in qrels
        if clean_text(qrel["query_table_id"]) not in dropped_explicit_ids
    ]
    qrels_by_query = defaultdict(list)
    for qrel in qrels:
        qrels_by_query[clean_text(qrel["query_table_id"])].append(qrel)
    kept_query_ids = set(qrels_by_query)
    # The query-level cascade: a query whose only qrel was degenerate leaves here.
    degenerate_dropped_query_ids = sorted(
        {query_id for query_id, _ in dropped_pairs} - kept_query_ids)

    # 3. re-split on whole source tables.
    queries_by_source: dict[str, list[str]] = defaultdict(list)
    for query_id, query_qrels in qrels_by_query.items():
        queries_by_source[clean_text(query_qrels[0]["source_table_id"])].append(query_id)
    split_of = assign_splits(queries_by_source=queries_by_source, seed=seed)

    # 4. rekey the lake, then restore whatever the removal emptied.
    referenced_targets = {
        clean_text(qrel["target_table_id"]) for qrel in qrels
    }
    targets_by_source: dict[str, list[str]] = defaultdict(list)
    for record in lake_all:
        targets_by_source[clean_text(record["source_table_id"])].append(
            clean_text(record["table_id"]))
    emptied = sorted(
        source_id for source_id, target_ids in targets_by_source.items()
        if not (set(target_ids) & referenced_targets)
    )
    query_split_of = {
        query_id: split_of[clean_text(query_qrels[0]["source_table_id"])]
        for query_id, query_qrels in qrels_by_query.items()
    }
    restored = [
        restore_record(sources[source_id])
        for source_id in emptied
        if source_id in sources
    ]
    restored_ids = {record["table_id"] for record in restored}

    # 5. write every artifact the inspection passes touched.
    split_of_query = {
        query_id: query_split_of[query_id] for query_id in qrels_by_query
    }
    for qrel in qrels:
        qrel["split"] = split_of_query[clean_text(qrel["query_table_id"])]
    query_tables = [
        synchronize_query_table(record, qrels_by_query[clean_text(record["table_id"])])
        for record in query_records
        if clean_text(record["table_id"]) in kept_query_ids
    ]
    for record in query_tables:
        record["split"] = split_of_query[clean_text(record["table_id"])]
    data_lake_tables = [
        record for record in lake_all
        if clean_text(record["table_id"]) in referenced_targets
    ] + restored

    # A recovery only means something while its pair survives.  Recoveries name
    # their pair the same way a qrel does.
    surviving = {
        (clean_text(qrel["query_table_id"]), clean_text(qrel["target_table_id"]))
        for qrel in qrels
    }
    recoveries_all = read_records(source, "evidence_recoveries")
    recoveries = []
    dropped_recoveries: list[dict[str, Any]] = []
    for recovery in recoveries_all:
        pair = (clean_text(recovery.get("query_table_id")),
                clean_text(recovery.get("target_table_id")))
        if pair not in surviving:
            dropped_recoveries.append({
                "recovery_id": clean_text(recovery.get("recovery_id")),
                "reason": "pair_removed",
            })
            continue
        recovery["split"] = split_of_query[pair[0]]
        recoveries.append(recovery)

    one_file = not any(sharded(source, name) for name in SHARDED_ARTIFACTS)
    artifact_counts: dict[str, int] = {}
    for name in SHARDED_ARTIFACTS:
        if name in {"query_tables", "data_lake_tables", "evidence_recoveries"}:
            continue
        records = read_records(source, name)
        if records or read_paths(source, name):
            write_records(target, name, records, one_file=one_file)
            artifact_counts[name] = len(records)
    write_records(target, "query_tables", query_tables, one_file=one_file)
    write_records(target, "data_lake_tables", data_lake_tables, one_file=one_file)
    write_records(target, "evidence_recoveries", recoveries, one_file=one_file)
    artifact_counts.update({
        "query_tables": len(query_tables),
        "data_lake_tables": len(data_lake_tables),
        "evidence_recoveries": len(recoveries),
    })
    write_jsonl(target / "qrels.jsonl", qrels)
    for name in SINGLE_FILES:
        path = source / name
        if path.exists():
            shutil.copyfile(path, target / name)

    split = split_summary(query_tables, qrels)
    update_stats(source, target, artifact_counts=artifact_counts, qrels=qrels,
                 query_tables=query_tables, split=split)
    update_manifest(source, target, artifact_counts=artifact_counts, one_file=one_file)

    report = {
        "dataset_dir": str(source),
        "output_dir": str(target),
        "lake_dir": str(args.lake_dir),
        "seed": seed,
        "max_value_share": float(args.max_value_share),
        "degenerate_join_pairs": {
            "total": len(dropped_pairs),
            "by_role": dict(Counter(item["role"] for item in dropped_detail)),
            "by_column": dict(Counter(item["column_name"] for item in dropped_detail)),
            "pairs": dropped_detail,
        },
        "queries": {
            "total": len(query_records),
            "kept": len(query_tables),
            "dropped_degenerate": len(degenerate_dropped_query_ids),
            "dropped_degenerate_ids": degenerate_dropped_query_ids,
            "implicit": len(implicit_ids),
            "explicit": len(selected_explicit),
            "explicit_dropped_for_balance": len(dropped_explicit_ids),
            "explicit_dropped_ids": dropped_explicit_ids,
        },
        "qrels": {"total": len(qrels_all), "kept": len(qrels),
                  "dropped_degenerate": len(dropped_pairs),
                  "dropped_balance": len(dropped_explicit_qrels)},
        "data_lake": {
            "total": len(lake_all),
            "targets_kept": len(referenced_targets),
            "targets_dropped": len(lake_all) - len(referenced_targets),
            "restored": len(restored),
            "restored_source_table_ids": emptied,
            "restored_table_ids": sorted(restored_ids),
        },
        "recoveries": {"total": len(recoveries_all), "kept": len(recoveries),
                       "dropped": len(dropped_recoveries)},
        "split": split,
    }
    write_json(target / "clean_report.json", report)
    # The dropped rows are long; keep them out of the printed summary but in the
    # artifact, where a reader can diff them against the source dataset.
    write_jsonl(target / "dropped_pairs.jsonl",
                dropped_detail + [{
                    "query_table_id": clean_text(qrel["query_table_id"]),
                    "target_table_id": clean_text(qrel["target_table_id"]),
                    "column_name": clean_text(
                        (qrel.get("join_attribute") or {}).get("column_name")),
                    "reason": "explicit_query_balance",
                } for qrel in dropped_explicit_qrels])
    return report


def split_summary(
    query_tables: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> dict[str, Any]:
    roles_by_query: dict[str, set[str]] = defaultdict(set)
    for qrel in qrels:
        roles_by_query[clean_text(qrel["query_table_id"])].add(qrel_role(qrel))
    by_split: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        query_ids = {
            clean_text(query["table_id"]) for query in query_tables
            if clean_text(query.get("split")) == split
        }
        implicit = sum(IMPLICIT_ROLE in roles_by_query.get(query_id, set())
                       for query_id in query_ids)
        explicit = sum(EXPLICIT_ROLE in roles_by_query.get(query_id, set())
                       for query_id in query_ids)
        by_split[split] = {
            "implicit": implicit,
            "explicit": explicit,
            "total": len(query_ids),
        }
    total = len(query_tables)
    for split, counts in by_split.items():
        counts["ratio"] = counts["total"] / total if total else 0.0

    source_splits: dict[str, set[str]] = defaultdict(set)
    for query in query_tables:
        source_splits[clean_text(query.get("source_table_id"))].add(
            clean_text(query.get("split")))
    return {
        "split_key": "source_table_id",
        "split_policy": "query_only",
        "data_lake_scope": "shared",
        "target_ratios": SPLIT_RATIOS,
        "by_split": by_split,
        "cross_split_source_table_ids": sorted(
            source_id for source_id, splits in source_splits.items() if len(splits) > 1
        ),
    }


def update_stats(
    source: Path,
    target: Path,
    *,
    artifact_counts: dict[str, int],
    qrels: list[dict[str, Any]],
    query_tables: list[dict[str, Any]],
    split: dict[str, Any],
) -> None:
    path = source / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stats.update({
        "query_tables": artifact_counts.get("query_tables", 0),
        "data_lake_tables": artifact_counts.get("data_lake_tables", 0),
        "qrels": len(qrels),
        "evidence_recoveries": artifact_counts.get("evidence_recoveries", 0),
        "queries_by_split": {
            name: values["total"] for name, values in split["by_split"].items()
        },
    })
    roles_by_query: dict[str, set[str]] = defaultdict(set)
    for qrel in qrels:
        roles_by_query[clean_text(qrel["query_table_id"])].add(qrel_role(qrel))
    query_by_id = {clean_text(query["table_id"]): query for query in query_tables}
    implicit_sources = {
        clean_text(query_by_id[query_id].get("source_table_id"))
        for query_id, roles in roles_by_query.items()
        if IMPLICIT_ROLE in roles and query_id in query_by_id
    }
    explicit_sources = {
        clean_text(query_by_id[query_id].get("source_table_id"))
        for query_id, roles in roles_by_query.items()
        if EXPLICIT_ROLE in roles and query_id in query_by_id
    }
    hidden: Counter[str] = Counter(
        clean_text((qrel.get("join_attribute") or {}).get("column_name"))
        for qrel in qrels
    )
    table_stats = dict(stats.get("tables") or {})
    table_stats.update({
        "explicit_join_tables": len(explicit_sources),
        "explicit_materialized": sum(
            EXPLICIT_ROLE in roles for roles in roles_by_query.values()),
        "explicit_needed": sum(
            IMPLICIT_ROLE in roles for roles in roles_by_query.values()),
        "explicit_selected": sum(
            EXPLICIT_ROLE in roles for roles in roles_by_query.values()),
        "multimodal_queryable_tables": len(implicit_sources),
        "queryable_tables": len(implicit_sources),
    })
    stats["tables"] = table_stats
    stats["hidden_columns"] = dict(sorted(hidden.items()))
    stats["clean"] = {
        "implicit_queries": sum(
            IMPLICIT_ROLE in roles for roles in roles_by_query.values()),
        "explicit_queries": sum(
            EXPLICIT_ROLE in roles for roles in roles_by_query.values()),
        "split": split,
    }
    write_json(target / "stats.json", stats)


def update_manifest(
    source: Path,
    target: Path,
    *,
    artifact_counts: dict[str, int],
    one_file: bool,
) -> None:
    path = source / "dataset_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    max_records = int(manifest.get("records_per_shard") or 50_000)
    artifacts = dict(manifest.get("artifacts") or {})
    for name, count in artifact_counts.items():
        paths = read_paths(target, name)
        if one_file:
            artifacts[name] = {"path": f"{name}.jsonl", "records": count}
            continue
        artifacts[name] = {
            "directory": name,
            "total_records": count,
            "max_records_per_shard": max_records,
            "shards": [
                {
                    "path": str(shard.relative_to(target)).replace("\\", "/"),
                    "records": sum(1 for line in shard.open() if line.strip()),
                }
                for shard in paths
            ],
        }
    manifest["artifacts"] = artifacts
    construction = dict(manifest.get("query_construction") or {})
    construction["cleaning"] = "degenerate_join_columns_removed"
    manifest["query_construction"] = construction
    write_json(target / "dataset_manifest.json", manifest)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = build(args)
    print(json.dumps({
        key: value for key, value in report.items()
        if key != "degenerate_join_pairs"
    } | {"degenerate_join_pairs": {
        key: value for key, value in report["degenerate_join_pairs"].items()
        if key != "pairs"
    }}, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
