"""Tests for turning human recovery verdicts into a dataset.

The checker app records verdicts (``tests/test_mm_joinability_checker.py``);
this is what those verdicts do to the data.  The cascade is the part worth
pinning: one unreasonable recovery can push a qrel under its own
``required_recovered_rows``, and a qrel with no rows behind it takes its query
out of the dataset with it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))
sys.path.insert(0, str(ROOT / "src"))

from apply_recovery_review import build as apply_review  # noqa: E402
from mm_joinability_dataset_checker import QualityReviewStore  # noqa: E402


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")


def make_dataset(root: Path, *, with_explicit_qrel: bool = True) -> Path:
    """One query whose hidden column needs three rows, recovered on rows 0-2."""
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "query_tables.jsonl", [
        {"table_id": "query_1", "role": "query", "split": "train",
         "source_table_id": "st_1", "columns": [{"column_index": 0, "column_name": "Title"}],
         "rows": [{"row_id": 0, "source_row_id": "bk_0001",
                   "cells": [{"column_index": 0, "column_name": "Title", "text": "Book"}]}]},
    ])
    write_jsonl(root / "data_lake_tables.jsonl", [
        {"table_id": "target_1", "role": "target_data_lake_table", "split": "train",
         "columns": [{"column_index": 0, "column_name": "Publisher"}],
         "rows": [{"row_id": 0, "cells": []}]},
        {"table_id": "target_explicit", "role": "target_data_lake_table", "split": "train",
         "columns": [{"column_index": 0, "column_name": "Price"}], "rows": []},
    ])
    write_jsonl(root / "evidence_recoveries.jsonl", [
        {"recovery_id": f"rec_{row}", "query_table_id": "query_1",
         "target_table_id": "target_1", "query_row_id": row,
         "recovered_attribute": {"column_name": "Publisher", "value": f"P{row}"}}
        for row in (0, 1, 2)
    ])
    qrels = [
        {"query_table_id": "query_1", "target_table_id": "target_1",
         "join_attribute": {"column_name": "Publisher", "eligible_rows": 5,
                            "recovered_rows": 3, "required_recovered_rows": 3,
                            "role": "model_recoverable_join_column"}},
    ]
    if with_explicit_qrel:
        qrels.append(
            {"query_table_id": "query_1", "target_table_id": "target_explicit",
             "join_attribute": {"column_name": "Price", "recovered_rows": 0,
                                "required_recovered_rows": 0,
                                "role": "visible_join_column"}})
    write_jsonl(root / "qrels.jsonl", qrels)
    write_jsonl(root / "bridge_assets.jsonl", [])
    (root / "stats.json").write_text("{}", encoding="utf-8")
    return root


def run(dataset: Path, db: Path, out: Path, *extra: str) -> dict:
    return apply_review(argparse.Namespace(
        dataset_dir=str(dataset), review_db=str(db), output_dir=str(out),
        drop_unreviewed="--drop-unreviewed" in extra,
        balance_explicit="--balance-explicit" in extra, seed=13))


def test_an_unreviewed_dataset_passes_through_unchanged(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "data")
    report = run(dataset, tmp_path / "missing.sqlite3", tmp_path / "out")

    assert report["recoveries"]["kept"] == 3
    assert report["recoveries"]["unreviewed"] == 3
    assert report["qrels"]["kept"] == 2
    assert report["qrels"]["dropped"] == 0
    assert report["query_tables"]["kept"] == 1


def test_a_rejected_recovery_can_take_its_query_out_with_it(tmp_path: Path) -> None:
    """Three recovered rows was exactly the requirement, so one verdict voids it."""
    dataset = make_dataset(tmp_path / "data")
    db = tmp_path / "reviews.sqlite3"
    store = QualityReviewStore(db)
    store.save_recovery("rec_1", "query_1", "unreasonable")

    report = run(dataset, db, tmp_path / "out")

    assert report["recoveries"] == {"total": 3, "kept": 0, "dropped": 3, "unreviewed": 2}
    assert [item["column_name"] for item in report["dropped_qrels"]] == ["Publisher"]
    assert report["dropped_qrels"][0]["recovered_rows"] == 2
    assert report["dropped_qrels"][0]["required_recovered_rows"] == 3
    # The query keeps its other, explicit qrel, so it stays; the target the
    # dropped qrel pointed at does not.
    assert report["qrels"]["kept"] == 1
    assert report["query_tables"]["kept"] == 1
    assert report["data_lake_tables"]["kept"] == 1

    written = [json.loads(line) for line in
               (tmp_path / "out" / "evidence_recoveries.jsonl").open()]
    assert written == []
    assert {item["reason"] for item in report["dropped_recoveries"]} == {
        "marked_unreasonable", "qrel_removed"
    }


def test_a_query_with_no_surviving_qrel_is_dropped(tmp_path: Path) -> None:
    """The last qrel going means the query has nothing left to join to."""
    dataset = make_dataset(tmp_path / "data", with_explicit_qrel=False)
    db = tmp_path / "reviews.sqlite3"
    QualityReviewStore(db).save_recovery("rec_1", "query_1", "unreasonable")

    report = run(dataset, db, tmp_path / "out")

    assert report["qrels"] == {"total": 1, "kept": 0, "dropped": 1}
    assert report["query_tables"]["kept"] == 0
    assert report["data_lake_tables"]["kept"] == 0


def test_an_explicit_join_is_not_touched_by_a_recovery_verdict(tmp_path: Path) -> None:
    """Explicit joins have no recoveries, so a recovery verdict cannot speak to them."""
    dataset = make_dataset(tmp_path / "data")
    db = tmp_path / "reviews.sqlite3"
    store = QualityReviewStore(db)
    store.save_recovery("rec_0", "query_1", "unreasonable")
    store.save_recovery("rec_1", "query_1", "unreasonable")
    store.save_recovery("rec_2", "query_1", "unreasonable")

    report = run(dataset, db, tmp_path / "out")

    kept_roles = [row["join_attribute"]["role"] for row in
                  (json.loads(line) for line in (tmp_path / "out" / "qrels.jsonl").open())]
    assert kept_roles == ["visible_join_column"]
    assert report["recoveries"]["dropped"] == 3


def test_drop_unreviewed_is_opt_in(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "data")
    db = tmp_path / "reviews.sqlite3"
    QualityReviewStore(db).save_recovery("rec_0", "query_1", "reasonable")

    report = run(dataset, db, tmp_path / "out", "--drop-unreviewed")

    assert report["recoveries"] == {"total": 3, "kept": 0, "dropped": 3, "unreviewed": 2}
    assert report["dropped_qrels"][0]["reason"] == "below_required_recovered_rows"


def test_balances_explicit_queries_per_split_and_refreshes_metadata(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data"
    dataset.mkdir()
    queries = [
        {"table_id": "implicit_train", "source_table_id": "st_i_train",
         "role": "query", "split": "train", "hidden_attributes": [
             {"source_column_index": 1, "column_name": "Publisher"}],
         "target_table_ids": ["target_i_train"]},
        {"table_id": "explicit_train_a", "source_table_id": "st_e_train_a",
         "role": "query", "split": "train", "hidden_attributes": [],
         "target_table_ids": ["target_e_train_a"]},
        {"table_id": "explicit_train_b", "source_table_id": "st_e_train_b",
         "role": "query", "split": "train", "hidden_attributes": [],
         "target_table_ids": ["target_e_train_b"]},
        {"table_id": "implicit_dev", "source_table_id": "st_i_dev",
         "role": "query", "split": "dev", "hidden_attributes": [
             {"source_column_index": 1, "column_name": "Publisher"}],
         "target_table_ids": ["target_i_dev"]},
        {"table_id": "explicit_dev", "source_table_id": "st_e_dev",
         "role": "query", "split": "dev", "hidden_attributes": [],
         "target_table_ids": ["target_e_dev"]},
    ]
    write_jsonl(dataset / "query_tables.jsonl", queries)
    targets = [
        {"table_id": query["target_table_ids"][0], "role": "target_data_lake_table"}
        for query in queries
    ]
    write_jsonl(dataset / "data_lake_tables.jsonl", targets)
    recoveries = [
        {"recovery_id": f"rec_{split}", "query_table_id": f"implicit_{split}",
         "target_table_id": f"target_i_{split}", "query_row_id": 0,
         "split": split}
        for split in ("train", "dev")
    ]
    write_jsonl(dataset / "evidence_recoveries.jsonl", recoveries)
    implicit_attribute = {
        "source_column_index": 1, "column_name": "Publisher",
        "role": "model_recoverable_join_column", "selected_rows": 2,
        "eligible_rows": 10, "recovered_rows": 1, "required_recovered_rows": 1,
        "recovered_value_ratio": 0.5,
    }
    qrels = [
        {"query_table_id": f"implicit_{split}",
         "target_table_id": f"target_i_{split}", "split": split,
         "join_attribute": implicit_attribute}
        for split in ("train", "dev")
    ] + [
        {"query_table_id": query["table_id"],
         "target_table_id": query["target_table_ids"][0], "split": query["split"],
         "join_attribute": {"column_name": "Price", "role": "visible_join_column"}}
        for query in queries if query["table_id"].startswith("explicit")
    ]
    write_jsonl(dataset / "qrels.jsonl", qrels)
    (dataset / "stats.json").write_text(
        json.dumps({"source_tables": 5, "tables": {}}), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(json.dumps({
        "format": "flat_jsonl",
        "artifacts": {
            "query_tables": {"path": "query_tables.jsonl", "records": 5},
            "data_lake_tables": {"path": "data_lake_tables.jsonl", "records": 5},
            "evidence_recoveries": {
                "path": "evidence_recoveries.jsonl", "records": 2},
        },
    }), encoding="utf-8")

    report = run(dataset, tmp_path / "missing.sqlite3", tmp_path / "out",
                 "--balance-explicit")

    assert report["query_tables"]["implicit"] == 2
    assert report["query_tables"]["explicit"] == 2
    assert report["split"]["by_split"]["train"]["total"] == 2
    assert report["split"]["by_split"]["dev"]["total"] == 2
    assert len(report["dropped_explicit_query_ids"]) == 1
    written_qrels = [json.loads(line) for line in
                     (tmp_path / "out" / "qrels.jsonl").open()]
    assert len(written_qrels) == 4
    implicit = next(row for row in written_qrels
                    if row["query_table_id"] == "implicit_train")
    assert implicit["join_attribute"]["recovered_value_ratio"] == 0.5
    stats = json.loads((tmp_path / "out" / "stats.json").read_text())
    assert stats["query_tables"] == 4
    assert stats["qrels"] == 4
    manifest = json.loads((tmp_path / "out" / "dataset_manifest.json").read_text())
    assert manifest["artifacts"]["query_tables"]["records"] == 4
