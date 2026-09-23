#!/usr/bin/env python
"""Apply human recovery verdicts to a built joinability dataset.

The quality checker (``mm_joinability_dataset_checker.py``) lets a person mark
each recovery reasonable or unreasonable.  This turns those marks into a dataset:

1. drop every recovery marked unreasonable;
2. recompute each implicit qrel's recovered-row count from what is left, and drop
   the qrel when it no longer reaches its own ``required_recovered_rows``;
3. drop recoveries, queries, and data-lake tables no surviving qrel references;
4. optionally subsample explicit (visible-column) queries per split so their
   query-level count matches the surviving implicit queries.

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
import hashlib
import json
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
IMPLICIT_ROLE = "model_recoverable_join_column"
EXPLICIT_ROLE = "visible_join_column"
SPLIT_NAMES = ("train", "dev", "test")
TARGET_SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}


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
    result.add_argument(
        "--balance-explicit", action="store_true",
        help="keep one explicit query per surviving implicit query in each split",
    )
    result.add_argument(
        "--seed", type=int, default=13,
        help="deterministic seed used when subsampling explicit queries (default: 13)",
    )
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


def qrel_role(qrel: dict[str, Any]) -> str:
    return clean_text((qrel.get("join_attribute") or {}).get("role"))


def query_roles(qrels: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = defaultdict(set)
    for qrel in qrels:
        roles[clean_text(qrel.get("query_table_id"))].add(qrel_role(qrel))
    return roles


def stable_query_key(query_id: str, split: str, seed: int) -> tuple[str, str]:
    payload = f"recovery-review-explicit-balance\0{seed}\0{split}\0{query_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), query_id


def select_explicit_queries(
    *,
    implicit_ids: set[str],
    explicit_ids: set[str],
    query_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> set[str]:
    """Select exactly as many explicit queries as implicit queries per split."""
    selected: set[str] = set()
    for split in SPLIT_NAMES:
        needed = sum(
            clean_text(query_by_id[query_id].get("split")) == split
            for query_id in implicit_ids
        )
        candidates = [
            query_id for query_id in explicit_ids
            if clean_text(query_by_id[query_id].get("split")) == split
        ]
        if len(candidates) < needed:
            raise ValueError(
                f"insufficient explicit queries in {split}: "
                f"required={needed}, available={len(candidates)}"
            )
        candidates.sort(key=lambda query_id: stable_query_key(query_id, split, seed))
        selected.update(candidates[:needed])
    return selected


def synchronize_query_table(
    query: dict[str, Any],
    qrels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep embedded targets and hidden-attribute counts aligned with qrels."""
    target_ids = list(dict.fromkeys(
        clean_text(qrel.get("target_table_id")) for qrel in qrels
    ))
    result = {**query, "target_table_ids": target_ids}
    if not isinstance(query.get("hidden_attributes"), list):
        return result

    surviving_attributes = {
        (
            int(attribute.get("source_column_index", -1)),
            clean_text(attribute.get("column_name")),
        ): attribute
        for qrel in qrels
        if qrel_role(qrel) == IMPLICIT_ROLE
        for attribute in [dict(qrel.get("join_attribute") or {})]
    }
    result["hidden_attributes"] = [
        surviving_attributes[key]
        for attribute in query["hidden_attributes"]
        for key in [(
            int(attribute.get("source_column_index", -1)),
            clean_text(attribute.get("column_name")),
        )]
        if key in surviving_attributes
    ]
    return result


def split_summary(
    query_tables: list[dict[str, Any]],
    roles_by_query: dict[str, set[str]],
) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        query_ids = {
            clean_text(query.get("table_id"))
            for query in query_tables
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
    cross_split_sources = sorted(
        source_id for source_id, splits in source_splits.items() if len(splits) > 1
    )

    implicit_total = sum(item["implicit"] for item in by_split.values())
    within_one_query = all(
        abs(by_split[split]["implicit"] - implicit_total * target) <= 1.0
        for split, target in TARGET_SPLIT_RATIOS.items()
    )
    return {
        "target_ratios": TARGET_SPLIT_RATIOS,
        "by_split": by_split,
        "within_one_query_of_target": within_one_query,
        "cross_split_source_table_ids": cross_split_sources,
        "rebalanced": False,
    }


def update_stats(
    source: Path,
    target: Path,
    *,
    artifact_counts: dict[str, int],
    qrels: list[dict[str, Any]],
    query_tables: list[dict[str, Any]],
    roles_by_query: dict[str, set[str]],
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

    implicit_ids = {
        query_id for query_id, roles in roles_by_query.items()
        if IMPLICIT_ROLE in roles
    }
    explicit_ids = {
        query_id for query_id, roles in roles_by_query.items()
        if EXPLICIT_ROLE in roles
    }
    query_by_id = {clean_text(query.get("table_id")): query for query in query_tables}
    implicit_sources = {
        clean_text(query_by_id[query_id].get("source_table_id"))
        for query_id in implicit_ids
    }
    explicit_sources = {
        clean_text(query_by_id[query_id].get("source_table_id"))
        for query_id in explicit_ids
    }
    table_stats = dict(stats.get("tables") or {})
    table_stats.update({
        "explicit_join_tables": len(explicit_sources),
        "explicit_materialized": len(explicit_ids),
        "explicit_needed": len(implicit_ids),
        "explicit_selected": len(explicit_ids),
        "explicit_selected_ids": len(explicit_ids),
        "multimodal_queryable_tables": len(implicit_sources),
        "queryable_tables": len(implicit_sources),
    })
    if "source_tables" in stats:
        table_stats["rejected_tables"] = int(stats["source_tables"]) - len(implicit_sources)
    stats["tables"] = table_stats
    stats["recovery_review"] = {
        "implicit_queries": len(implicit_ids),
        "explicit_queries": len(explicit_ids),
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
                    "path": str(path.relative_to(target)).replace("\\", "/"),
                    "records": sum(1 for line in path.open() if line.strip()),
                }
                for path in paths
            ],
        }
    manifest["artifacts"] = artifacts
    write_json(target / "dataset_manifest.json", manifest)


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.dataset_dir).resolve()
    target = Path(args.output_dir).resolve()
    if source == target:
        raise ValueError("--output-dir must differ from --dataset-dir")
    target.mkdir(parents=True, exist_ok=True)

    store = QualityReviewStore(Path(args.review_db).resolve())
    verdicts = {clean_text(row["recovery_id"]): clean_text(row["verdict"])
                for row in store.export_recovery_rows()}

    recoveries = read_records(source, "evidence_recoveries")
    candidate_recoveries: list[dict[str, Any]] = []
    dropped_recoveries: list[dict[str, Any]] = []
    unreviewed_ids: list[str] = []
    unreviewed = 0
    for recovery in recoveries:
        recovery_id = clean_text(recovery.get("recovery_id"))
        verdict = verdicts.get(recovery_id, "")
        if not verdict:
            unreviewed += 1
            unreviewed_ids.append(recovery_id)
            if args.drop_unreviewed:
                dropped_recoveries.append({"recovery_id": recovery_id,
                                           "reason": "unreviewed"})
                continue
        elif verdict == "unreasonable":
            dropped_recoveries.append({"recovery_id": recovery_id,
                                       "reason": "marked_unreasonable"})
            continue
        candidate_recoveries.append(recovery)

    # Which query rows still have a recovery, per (query, target).
    rows_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for recovery in candidate_recoveries:
        rows_by_pair[(clean_text(recovery["query_table_id"]),
                      clean_text(recovery["target_table_id"]))
                     ].add(str(recovery["query_row_id"]))

    qrels = read_records(source, "qrels")
    kept_qrels: list[dict[str, Any]] = []
    dropped_qrels: list[dict[str, Any]] = []
    for qrel in qrels:
        hidden = dict(qrel.get("join_attribute") or {})
        if hidden.get("role") != IMPLICIT_ROLE:
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
        selected_rows = int(hidden.get("selected_rows")
                            or hidden.get("eligible_rows") or 0)
        kept_qrels.append({
            **qrel,
            "join_attribute": {
                **hidden,
                "recovered_rows": recovered,
                "recovered_value_ratio": (
                    recovered / selected_rows if selected_rows else 0.0
                ),
            },
        })

    source_query_tables = read_records(source, "query_tables")
    query_by_id = {
        clean_text(query.get("table_id")): query for query in source_query_tables
    }
    roles_before_review = query_roles(qrels)
    mixed_role_ids = sorted(
        query_id for query_id, roles in roles_before_review.items()
        if IMPLICIT_ROLE in roles and EXPLICIT_ROLE in roles
    )
    balance_explicit = bool(getattr(args, "balance_explicit", False))
    if balance_explicit and mixed_role_ids:
        raise ValueError(
            "cannot query-balance mixed implicit/explicit queries: "
            + ", ".join(mixed_role_ids[:5])
        )

    surviving_implicit_ids = {
        clean_text(qrel.get("query_table_id")) for qrel in kept_qrels
        if qrel_role(qrel) == IMPLICIT_ROLE
    }
    all_explicit_ids = {
        query_id for query_id, roles in roles_before_review.items()
        if EXPLICIT_ROLE in roles
    }
    missing_query_ids = sorted(
        (surviving_implicit_ids | all_explicit_ids) - query_by_id.keys()
    )
    if missing_query_ids:
        raise ValueError(f"qrels reference missing queries: {missing_query_ids[:5]}")

    selected_explicit_ids = all_explicit_ids
    if balance_explicit:
        selected_explicit_ids = select_explicit_queries(
            implicit_ids=surviving_implicit_ids,
            explicit_ids=all_explicit_ids,
            query_by_id=query_by_id,
            seed=int(getattr(args, "seed", 13)),
        )
        balanced_qrels: list[dict[str, Any]] = []
        for qrel in kept_qrels:
            query_id = clean_text(qrel.get("query_table_id"))
            if qrel_role(qrel) == EXPLICIT_ROLE and query_id not in selected_explicit_ids:
                dropped_qrels.append({
                    "query_table_id": query_id,
                    "target_table_id": clean_text(qrel.get("target_table_id")),
                    "column_name": clean_text(
                        (qrel.get("join_attribute") or {}).get("column_name")),
                    "reason": "explicit_query_balance",
                })
                continue
            balanced_qrels.append(qrel)
        kept_qrels = balanced_qrels

    kept_query_ids = {clean_text(qrel["query_table_id"]) for qrel in kept_qrels}
    kept_target_ids = {clean_text(qrel["target_table_id"]) for qrel in kept_qrels}

    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qrel in kept_qrels:
        query = query_by_id[clean_text(qrel.get("query_table_id"))]
        qrel["split"] = clean_text(query.get("split"))
        qrels_by_query[clean_text(qrel.get("query_table_id"))].append(qrel)
    query_tables = [
        synchronize_query_table(record, qrels_by_query[clean_text(record.get("table_id"))])
        for record in source_query_tables
        if clean_text(record.get("table_id")) in kept_query_ids
    ]
    data_lake_tables = [record for record in read_records(source, "data_lake_tables")
                        if clean_text(record.get("table_id")) in kept_target_ids]

    surviving_implicit_pairs = {
        (clean_text(qrel.get("query_table_id")),
         clean_text(qrel.get("target_table_id")))
        for qrel in kept_qrels if qrel_role(qrel) == IMPLICIT_ROLE
    }
    kept_recoveries = []
    for recovery in candidate_recoveries:
        pair = (clean_text(recovery.get("query_table_id")),
                clean_text(recovery.get("target_table_id")))
        if pair not in surviving_implicit_pairs:
            dropped_recoveries.append({
                "recovery_id": clean_text(recovery.get("recovery_id")),
                "reason": "qrel_removed",
            })
            continue
        recovery["split"] = clean_text(query_by_id[pair[0]].get("split"))
        kept_recoveries.append(recovery)

    final_roles = query_roles(kept_qrels)
    split = split_summary(query_tables, final_roles)
    if split["cross_split_source_table_ids"]:
        raise ValueError(
            "source_table_id occurs in multiple splits: "
            + ", ".join(split["cross_split_source_table_ids"][:5])
        )

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
    write_records(target, "evidence_recoveries", kept_recoveries, one_file=one_file)
    artifact_counts.update({
        "query_tables": len(query_tables),
        "data_lake_tables": len(data_lake_tables),
        "evidence_recoveries": len(kept_recoveries),
    })
    write_jsonl(target / "qrels.jsonl", kept_qrels)
    for name in SINGLE_FILES:
        path = source / name
        if path.exists():
            shutil.copyfile(path, target / name)
    update_stats(
        source, target, artifact_counts=artifact_counts, qrels=kept_qrels,
        query_tables=query_tables, roles_by_query=final_roles, split=split,
    )
    update_manifest(
        source, target, artifact_counts=artifact_counts, one_file=one_file,
    )

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
        "query_tables": {
            "total": len(source_query_tables), "kept": len(query_tables),
            "dropped": len(source_query_tables) - len(query_tables),
            "implicit": sum(IMPLICIT_ROLE in roles for roles in final_roles.values()),
            "explicit": sum(EXPLICIT_ROLE in roles for roles in final_roles.values()),
        },
        "data_lake_tables": {"kept": len(data_lake_tables)},
        "split": split,
        "balance_explicit": balance_explicit,
        "dropped_explicit_query_ids": sorted(all_explicit_ids - selected_explicit_ids),
        "unreviewed_recovery_ids": sorted(unreviewed_ids),
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
