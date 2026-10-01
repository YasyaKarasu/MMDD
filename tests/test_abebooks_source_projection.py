import copy

import pytest

from mmdd_dataset.abebooks_ablation import write_rows
from mmdd_dataset.abebooks_source_projection import (
    natural_author_value, normalize_qrels, project_source_dataset, refresh_source_profiles,
)


def test_projected_source_profiles_keep_sparse_annotation_indices():
    table = {"num_cols": 9, "columns": [{"column_index": 0, "column_name": "title"},
                                         {"column_index": 8, "column_name": "authors"}],
             "rows": [{"cells": [{"text": title, "wiki_title": title},
                                   {"text": "Jane Doe"}]} for title in ("Book A", "Book B")]}
    refresh_source_profiles(table)
    assert table["num_cols"] == table["num_rows"] == 2
    assert table["metadata"]["candidate_entity_columns"] == [0]
    profiles = table["metadata"]["column_profiles"]
    assert [p["column_index"] for p in profiles] == [0, 8]
    assert profiles[1]["unique_ratio"] == 0.5
    assert table["columns"][1]["column_index"] == 8


def test_author_display_normalization_preserves_full_names_and_ambiguous_lists():
    assert natural_author_value("Ousterhout, John K.") == "John K. Ousterhout"
    assert natural_author_value("Doe, Jane ; Smith, John") == "Jane Doe; John Smith"
    assert natural_author_value("Joe Kaplan, Ryan Dunn") == "Joe Kaplan, Ryan Dunn"
    assert natural_author_value("Smith") == "Smith"


def test_reason_migration_preserves_every_label_and_leaves_input_untouched():
    records = [{"query_table_id": "q1", "target_table_id": "t1", "rel": 3, "split": "train",
                "reason": "explicit_join_column", "join_attribute": {
                    "role": "explicit_join_column", "column_name": "authors", "source_column_index": 1}},
               {"query_table_id": "q2", "target_table_id": "t2", "rel": 3, "split": "test",
                "reason": "model_recoverable_join_column"}]
    before = copy.deepcopy(records)
    migrated = normalize_qrels(records)
    assert records == before
    assert migrated[1] == before[1]
    assert migrated[0]["reason"] == migrated[0]["join_attribute"]["role"] == "explicit_visible_join_column"
    migrated[0]["reason"] = "explicit_join_column"
    migrated[0]["join_attribute"]["role"] = "explicit_join_column"
    assert migrated == before


def test_projection_cannot_overwrite_or_nest_in_source(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    write_rows(source / "qrels.jsonl", [{"rel": 3}])
    for destination in (source, source / "copy"):
        with pytest.raises(ValueError, match="outside"):
            project_source_dataset(source, destination)
    destination = tmp_path / "already_there"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        project_source_dataset(source, destination)
    assert (source / "qrels.jsonl").read_text().strip() == '{"rel": 3}'
