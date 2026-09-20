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
        drop_unreviewed="--drop-unreviewed" in extra))


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

    assert report["recoveries"] == {"total": 3, "kept": 2, "dropped": 1, "unreviewed": 2}
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
    assert {record["recovery_id"] for record in written} == {"rec_0", "rec_2"}


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

    assert report["recoveries"] == {"total": 3, "kept": 1, "dropped": 2, "unreviewed": 2}
    assert report["dropped_qrels"][0]["reason"] == "below_required_recovered_rows"
