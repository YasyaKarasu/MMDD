"""Tests for cleaning a built AbeBooks joinability dataset.

Three behaviours are worth pinning, because each one is a place the pass could
silently do nothing rather than fail:

* a join column with a single distinct value is dropped, and the query it
  belonged to is dropped only when no other qrel still names it;
* a source table whose every target left the lake comes back whole, with its
  full column set and every row, and no ``join_col``;
* the surviving queries are re-split 8:1:1 on whole source tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))
sys.path.insert(0, str(ROOT / "src"))

from clean_abebooks_joinability import build as clean  # noqa: E402

#: Columns every fixture table carries, in this order.
COLUMNS = [
    {"column_index": 0, "column_name": "title"},
    {"column_index": 1, "column_name": "language"},
    {"column_index": 2, "column_name": "publisher"},
]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")


def source_table(source_table_id: str, *, degenerate: set[str]) -> dict:
    """One table whose columns the caller chooses to make degenerate.

    ``degenerate`` lists the column names every row shares one value in; the
    rest differ per row and so are legitimate join keys.
    """
    rows = []
    for row_index in range(6):
        language = ("English" if "language" in degenerate else f"Lang{row_index}")
        publisher = ("P" if "publisher" in degenerate else f"Pub{row_index}")
        rows.append({
            "row_id": row_index + 1,
            "cells": [
                {"column_index": 0, "column_name": "title",
                 "raw": f"T{row_index}", "text": f"T{row_index}",
                 "wiki_title": f"abe_{source_table_id}_{row_index}",
                 "has_wiki_link": True},
                {"column_index": 1, "column_name": "language",
                 "raw": language, "text": language},
                {"column_index": 2, "column_name": "publisher",
                 "raw": publisher, "text": publisher},
            ],
        })
    return {
        "source_table_id": source_table_id,
        "provenance_builder": "test",
        "source_file": "book",
        "num_rows": len(rows),
        "num_cols": len(COLUMNS),
        "columns": COLUMNS,
        "rows": rows,
        "metadata": {
            "candidate_entity_columns": [0],
            "column_profiles": [
                {"column_index": index, "non_empty_ratio": 1.0,
                 "wiki_link_ratio": 1.0 if index == 0 else 0.0,
                 "unique_ratio": (0.1 if COLUMNS[index]["column_name"] in degenerate
                                  else 1.0),
                 "numeric_ratio": 0.0}
                for index in range(len(COLUMNS))
            ],
        },
    }


def query_table(query_id: str, source_table_id: str, join_col: int,
                join_name: str, *, role: str, split: str,
                target_table_id: str, chain_id: str) -> dict:
    hidden = [{
        "source_column_index": join_col, "column_name": join_name, "role": role,
    }] if role == "model_recoverable_join_column" else []
    return {
        "table_id": query_id,
        "object_id": query_id,
        "object_type": "table",
        "role": "query",
        "source_table_id": source_table_id,
        "columns": [{"column_index": 0, "source_column_index": 0,
                     "column_name": "title"}],
        "rows": [{"row_id": 0, "source_row_id": 1,
                  "cells": [{"column_index": 0, "source_column_index": 0,
                             "column_name": "title", "text": "T0"}]}],
        "source_column_indices": [0],
        "source_row_indices": [1],
        "chain_id": chain_id,
        "chain_ids": [chain_id],
        "hidden_attributes": hidden,
        "target_table_ids": [target_table_id],
        "row_view_index": 0,
        "split": split,
    }


def target_table(table_id: str, source_table_id: str, join_col: int,
                 join_name: str) -> dict:
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "role": "target_data_lake_table",
        "source_table_id": source_table_id,
        "columns": [{"column_index": 0, "source_column_index": join_col,
                     "column_name": join_name}],
        "rows": [{"row_id": 0, "source_row_id": 1,
                  "cells": [{"column_index": 0, "source_column_index": join_col,
                             "column_name": join_name, "text": "v"}]}],
        "source_column_indices": [join_col],
        "source_row_indices": [1],
        "join_col": join_col,
        "join_col_name": join_name,
        "target_context_col_names": [],
    }


def make_dataset(root: Path) -> Path:
    """Ten tables covering every branch the pass has to take.

    * ``st_mixed``   - a degenerate ``language`` pair *and* a clean ``publisher``
                       pair, so the degenerate qrel goes while the query and the
                       source table both stay;
    * ``st_solo``    - one degenerate pair, so the query goes, its target leaves
                       the lake, and the source table is restored;
    * ``st_dropped`` - one degenerate *explicit* pair, same cascade, which also
                       empties the explicit side enough to exercise the cap;
    * ``st_ok1..7``  - clean implicit pairs, the bulk the split pass divides.
    """
    sources = [
        # language degenerate, publisher clean: one pair goes, the table stays.
        source_table("st_mixed", degenerate={"language"}),
        # both degenerate: its one pair goes and the table is restored.
        source_table("st_solo", degenerate={"language", "publisher"}),
        # publisher degenerate, the explicit pair that goes with it.
        source_table("st_dropped", degenerate={"publisher"}),
    ] + [source_table(f"st_ok{index}", degenerate=set()) for index in range(1, 8)]

    queries = [
        query_table("q_mixed_language", "st_mixed", 1, "language",
                    role="model_recoverable_join_column", split="train",
                    target_table_id="t_mixed_language", chain_id="c_mixed_lang"),
        query_table("q_mixed_publisher", "st_mixed", 2, "publisher",
                    role="model_recoverable_join_column", split="train",
                    target_table_id="t_mixed_publisher", chain_id="c_mixed_pub"),
        query_table("q_solo_language", "st_solo", 1, "language",
                    role="model_recoverable_join_column", split="train",
                    target_table_id="t_solo_language", chain_id="c_solo"),
        query_table("q_dropped_explicit", "st_dropped", 2, "publisher",
                    role="visible_join_column", split="train",
                    target_table_id="t_dropped_explicit", chain_id="c_dropped"),
    ] + [
        query_table(f"q_ok{index}", f"st_ok{index}", 2, "publisher",
                    role="model_recoverable_join_column", split="train",
                    target_table_id=f"t_ok{index}", chain_id=f"c_ok{index}")
        for index in range(1, 8)
    ]

    targets = [
        target_table("t_mixed_language", "st_mixed", 1, "language"),
        target_table("t_mixed_publisher", "st_mixed", 2, "publisher"),
        target_table("t_solo_language", "st_solo", 1, "language"),
        target_table("t_dropped_explicit", "st_dropped", 2, "publisher"),
    ] + [target_table(f"t_ok{index}", f"st_ok{index}", 2, "publisher")
         for index in range(1, 8)]

    def qrel(query_id, target_id, source_id, column, role, reason, chain):
        return {
            "query_table_id": query_id, "target_table_id": target_id, "rel": 3,
            "split": "train", "chain_id": chain, "source_table_id": source_id,
            "join_attribute": {"source_column_index": column, "role": role},
            "reason": reason,
        }

    qrels = [
        qrel("q_mixed_language", "t_mixed_language", "st_mixed", 1,
             "model_recoverable_join_column", "model_recoverable_join_column",
             "c_mixed_lang"),
        qrel("q_mixed_publisher", "t_mixed_publisher", "st_mixed", 2,
             "model_recoverable_join_column", "model_recoverable_join_column",
             "c_mixed_pub"),
        qrel("q_solo_language", "t_solo_language", "st_solo", 1,
             "model_recoverable_join_column", "model_recoverable_join_column",
             "c_solo"),
        qrel("q_dropped_explicit", "t_dropped_explicit", "st_dropped", 2,
             "visible_join_column", "explicit_visible_join_column", "c_dropped"),
    ] + [
        qrel(f"q_ok{index}", f"t_ok{index}", f"st_ok{index}", 2,
             "model_recoverable_join_column", "model_recoverable_join_column",
             f"c_ok{index}")
        for index in range(1, 8)
    ]
    # The label each qrel's column name is read from.
    for record in qrels:
        source_id = record["source_table_id"]
        table = next(s for s in sources if s["source_table_id"] == source_id)
        record["join_attribute"]["column_name"] = COLUMNS[
            record["join_attribute"]["source_column_index"]]["column_name"]

    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "source_tables" / "part-00000.jsonl", sources)
    write_jsonl(root / "query_tables" / "part-00000.jsonl", queries)
    write_jsonl(root / "data_lake_tables" / "part-00000.jsonl", targets)
    write_jsonl(root / "evidence_recoveries" / "part-00000.jsonl", [
        {"recovery_id": f"rec_{index}", "query_table_id": f"q_ok{index}",
         "target_table_id": f"t_ok{index}", "split": "train"}
        for index in range(1, 8)])
    write_jsonl(root / "qrels.jsonl", qrels)
    (root / "splits.json").write_text(json.dumps({
        "split_key": "source_table_id", "split_policy": "query_only",
        "data_lake_scope": "shared"}), encoding="utf-8")
    (root / "dataset_manifest.json").write_text(json.dumps({
        "format": "sharded_jsonl", "records_per_shard": 50_000,
        "artifacts": {}}), encoding="utf-8")
    return root


def args_for(source: Path, target: Path, **overrides) -> argparse.Namespace:
    values = {"dataset_dir": str(source), "output_dir": str(target),
              "max_value_share": 1.0, "seed": 13,
              "lake_dir": "output/abebooks_lake_no_copy"}
    values.update(overrides)
    return argparse.Namespace(**values)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def find(target: Path, name: str) -> list[dict]:
    flat = target / f"{name}.jsonl"
    if flat.exists():
        return read_jsonl(flat)
    return [record for path in sorted((target / name).glob("*.jsonl"))
            for record in read_jsonl(path)]


def test_a_constant_join_column_is_dropped(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "src")
    report = clean(args_for(source, tmp_path / "out"))

    # st_mixed.language, st_solo.language and st_dropped.publisher.
    assert report["degenerate_join_pairs"]["total"] == 3
    assert report["degenerate_join_pairs"]["by_column"] == {
        "language": 2, "publisher": 1}
    assert report["degenerate_join_pairs"]["by_role"] == {
        "visible_join_column": 1, "model_recoverable_join_column": 2}
    # No surviving qrel joins on a column that was degenerate.
    surviving = {
        (q["source_table_id"], q["join_attribute"]["column_name"])
        for q in find(tmp_path / "out", "qrels")
    }
    assert ("st_mixed", "language") not in surviving
    assert ("st_solo", "language") not in surviving
    assert ("st_dropped", "publisher") not in surviving


def test_a_query_survives_on_its_other_qrel(tmp_path: Path) -> None:
    """``st_mixed`` keeps the query whose *other* pair is clean."""
    source = make_dataset(tmp_path / "src")
    clean(args_for(source, tmp_path / "out"))

    kept = {q["table_id"] for q in find(tmp_path / "out", "query_tables")}
    qrel_queries = {q["query_table_id"] for q in find(tmp_path / "out", "qrels")}
    # The clean half of the mixed table stays; both fully-degenerate queries go.
    assert "q_mixed_publisher" in kept
    assert "q_mixed_language" not in kept
    assert "q_solo_language" not in kept
    # Nothing survives without a qrel, and nothing with one is dropped.
    assert kept == qrel_queries
    # The survivor's bookkeeping no longer names the removed chain.
    survivor = next(q for q in find(tmp_path / "out", "query_tables")
                    if q["table_id"] == "q_mixed_publisher")
    assert survivor["chain_ids"] == ["c_mixed_pub"]
    assert survivor["target_table_ids"] == ["t_mixed_publisher"]


def test_an_emptied_source_table_comes_back_whole(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "src")
    report = clean(args_for(source, tmp_path / "out"))

    lake = find(tmp_path / "out", "data_lake_tables")
    restored = {t["source_table_id"]: t for t in lake
                if t.get("role") == "restored_data_lake_table"}
    # st_solo and st_dropped each lost their only target.
    assert sorted(restored) == ["st_dropped", "st_solo"]
    assert report["data_lake"]["restored"] == 2

    record = restored["st_solo"]
    original = next(s for s in find(source, "source_tables")
                    if s["source_table_id"] == "st_solo")
    assert len(record["columns"]) == len(original["columns"])
    assert len(record["rows"]) == len(original["rows"])
    assert record["source_row_indices"] == [r["row_id"] for r in original["rows"]]
    assert record["source_column_indices"] == [
        c["column_index"] for c in original["columns"]]
    # The split along join_col is gone, so the record does not name one.
    assert "join_col" not in record
    assert "join_col_name" not in record
    # Cell text is carried through unchanged.
    for row, source_row in zip(record["rows"], original["rows"]):
        assert [c["text"] for c in row["cells"]] == [
            c["text"] for c in source_row["cells"]]


def test_a_partly_referenced_source_table_is_not_restored(tmp_path: Path) -> None:
    """``st_mixed`` keeps a target, so it must not be re-added whole."""
    source = make_dataset(tmp_path / "src")
    clean(args_for(source, tmp_path / "out"))

    lake = {t["table_id"]: t for t in find(tmp_path / "out", "data_lake_tables")}
    assert not [t for t in lake.values()
                if t.get("role") == "restored_data_lake_table"
                and t["source_table_id"] == "st_mixed"]
    assert "t_mixed_publisher" in lake
    # The degenerate target left the lake and took no replacement with it.
    assert "t_mixed_language" not in lake


def test_explicit_is_capped_at_the_implicit_count(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "src")
    report = clean(args_for(source, tmp_path / "out"))

    qrels = find(tmp_path / "out", "qrels")
    implicit = {q["query_table_id"] for q in qrels
                if q["reason"] == "model_recoverable_join_column"}
    explicit = {q["query_table_id"] for q in qrels
                if q["reason"] == "explicit_visible_join_column"}
    assert report["queries"]["implicit"] == len(implicit)
    assert len(explicit) <= len(implicit)
    # The fixture's one explicit pair is degenerate, so it is gone entirely.
    assert explicit == set()


def test_the_qrels_are_deterministic_across_runs(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "src")
    clean(args_for(source, tmp_path / "one"))
    clean(args_for(source, tmp_path / "two"))
    assert find(tmp_path / "one", "qrels") == find(tmp_path / "two", "qrels")
    assert (find(tmp_path / "one", "query_tables")
            == find(tmp_path / "two", "query_tables"))
    assert (find(tmp_path / "one", "data_lake_tables")
            == find(tmp_path / "two", "data_lake_tables"))


def test_a_source_table_never_spans_a_split(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "src")
    report = clean(args_for(source, tmp_path / "out"))
    assert report["split"]["cross_split_source_table_ids"] == []


def test_the_split_follows_its_ratios(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "src")
    report = clean(args_for(source, tmp_path / "out"))
    by_split = report["split"]["by_split"]
    total = sum(counts["total"] for counts in by_split.values())
    assert total == len(find(tmp_path / "out", "query_tables"))
    # Train takes the bulk; dev and test each take a slice.  Small fixtures
    # cannot hit 0.800 exactly, so this checks the ordering rather than a point.
    assert by_split["train"]["total"] > by_split["dev"]["total"]
    assert by_split["train"]["total"] > by_split["test"]["total"]
    assert by_split["train"]["ratio"] > 0.5


def test_the_threshold_boundary_is_exclusive(tmp_path: Path) -> None:
    """One row of variety (share 0.83) survives the default 1.0 threshold.

    The comparison has to be strict: ``modal_share`` returns exactly 1.0 for a
    constant column, so a non-strict gate would drop nothing at all.
    """
    source = make_dataset(tmp_path / "src")
    tables = find(source, "source_tables")
    for table in tables:
        if table["source_table_id"] == "st_solo":
            table["rows"][-1]["cells"][1]["text"] = "French"
            table["rows"][-1]["cells"][1]["raw"] = "French"
    write_jsonl(source / "source_tables" / "part-00000.jsonl", tables)

    report = clean(args_for(source, tmp_path / "out"))
    # st_solo.language is now 0.83 and survives; only st_mixed.language and
    # st_dropped.publisher are still constant.
    assert report["degenerate_join_pairs"]["total"] == 2
    assert report["degenerate_join_pairs"]["by_column"] == {
        "language": 1, "publisher": 1}


def test_a_strict_threshold_removes_the_partly_varied_column(
    tmp_path: Path,
) -> None:
    source = make_dataset(tmp_path / "src")
    report = clean(args_for(source, tmp_path / "out", max_value_share=0.5))
    # Every fixture column is either constant or fully varied, so a 0.5
    # threshold changes nothing about which pairs go.
    assert report["degenerate_join_pairs"]["total"] == 3


def test_the_input_dataset_is_untouched(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "src")
    before = {path.name: path.read_bytes()
              for path in sorted(source.rglob("*")) if path.is_file()}
    clean(args_for(source, tmp_path / "out"))
    after = {path.name: path.read_bytes()
             for path in sorted(source.rglob("*")) if path.is_file()}
    assert before == after
    assert not (tmp_path / "out").samefile(source)


def test_the_output_directory_may_not_be_the_input(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "src")
    with pytest.raises(ValueError, match="must differ"):
        clean(args_for(source, source))
