import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

from repair_mm_joinability_context_columns import repair_dataset
from stage1_io import write_jsonl


def make_cell(source_idx: int, out_idx: int, name: str, text: str) -> dict[str, object]:
    return {
        "column_index": out_idx,
        "source_column_index": source_idx,
        "column_name": name,
        "text": text,
    }


def write_manifest(dataset_dir: Path) -> None:
    manifest = {
        "format": "sharded_jsonl",
        "artifacts": {
            "source_tables": {"shards": [{"path": "source_tables/part-00000.jsonl", "records": 1}]},
            "query_tables": {"shards": [{"path": "query_tables/part-00000.jsonl", "records": 1}]},
            "data_lake_tables": {"shards": [{"path": "data_lake_tables/part-00000.jsonl", "records": 1}]},
        },
        "single_files": {"qrels": "qrels.jsonl"},
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_repair_dataset_rebuilds_context_columns_without_model_calls(tmp_path):
    dataset_dir = tmp_path / "joinability"
    (dataset_dir / "source_tables").mkdir(parents=True)
    (dataset_dir / "query_tables").mkdir()
    (dataset_dir / "data_lake_tables").mkdir()
    write_manifest(dataset_dir)

    source_table = {
        "source_table_id": "source_1",
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "Year"},
            {"column_index": 2, "column_name": "Role"},
            {"column_index": 3, "column_name": "Notes"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    {"column_index": 0, "column_name": "Entity", "text": "Alpha"},
                    {"column_index": 1, "column_name": "Year", "text": "2001"},
                    {"column_index": 2, "column_name": "Role", "text": "Lead"},
                    {"column_index": 3, "column_name": "Notes", "text": "Premiere"},
                ],
            },
            {
                "row_id": 1,
                "cells": [
                    {"column_index": 0, "column_name": "Entity", "text": "Beta"},
                    {"column_index": 1, "column_name": "Year", "text": "2002"},
                    {"column_index": 2, "column_name": "Role", "text": "Guest"},
                    {"column_index": 3, "column_name": "Notes", "text": "Finale"},
                ],
            },
        ],
        "metadata": {
            "column_profiles": [
                {"column_index": 0, "non_empty_ratio": 1.0, "unique_ratio": 1.0},
                {"column_index": 1, "non_empty_ratio": 1.0, "unique_ratio": 1.0},
                {"column_index": 2, "non_empty_ratio": 1.0, "unique_ratio": 0.5},
                {"column_index": 3, "non_empty_ratio": 0.9, "unique_ratio": 1.0},
            ]
        },
    }
    query = {
        "table_id": "query_1",
        "role": "query",
        "source_table_id": "source_1",
        "query_entity_col": 0,
        "source_column_indices": [0, 2],
        "source_row_indices": [0, 1],
        "columns": [{"column_index": 0, "column_name": "Entity"}, {"column_index": 1, "column_name": "Role"}],
        "rows": [
            {"row_id": 0, "source_row_id": 0, "cells": [make_cell(0, 0, "Entity", "Alpha"), make_cell(2, 1, "Role", "Lead")]},
            {"row_id": 1, "source_row_id": 1, "cells": [make_cell(0, 0, "Entity", "Beta"), make_cell(2, 1, "Role", "Guest")]},
        ],
        "hidden_attributes": [{"source_column_index": 1, "column_name": "Year"}],
        "query_context_col_names": ["Role"],
    }
    target = {
        "table_id": "target_1",
        "role": "target_data_lake_table",
        "source_table_id": "source_1",
        "join_col": 1,
        "source_column_indices": [1, 2],
        "source_row_indices": [0, 1],
        "columns": [{"column_index": 0, "column_name": "Year"}, {"column_index": 1, "column_name": "Role"}],
        "rows": [
            {"row_id": 0, "source_row_id": 0, "cells": [make_cell(1, 0, "Year", "2001"), make_cell(2, 1, "Role", "Lead")]},
            {"row_id": 1, "source_row_id": 1, "cells": [make_cell(1, 0, "Year", "2002"), make_cell(2, 1, "Role", "Guest")]},
        ],
        "target_context_col_names": ["Role"],
    }

    write_jsonl(dataset_dir / "source_tables" / "part-00000.jsonl", [source_table])
    write_jsonl(dataset_dir / "query_tables" / "part-00000.jsonl", [query])
    write_jsonl(dataset_dir / "data_lake_tables" / "part-00000.jsonl", [target])
    write_jsonl(dataset_dir / "qrels.jsonl", [{"query_table_id": "query_1", "target_table_id": "target_1"}])

    report = repair_dataset(dataset_dir, backup=True)

    repaired_query = read_jsonl(dataset_dir / "query_tables" / "part-00000.jsonl")[0]
    repaired_target = read_jsonl(dataset_dir / "data_lake_tables" / "part-00000.jsonl")[0]
    backups = list((dataset_dir / "backups").glob("mm_joinability_context_repair_*"))

    assert report["repaired_pairs"] == 1
    assert report["unresolved_pairs"] == 0
    assert repaired_query["source_column_indices"] == [0, 2]
    assert repaired_target["source_column_indices"] == [1, 3]
    assert repaired_target["target_context_col_names"] == ["Notes"]
    assert repaired_target["rows"][0]["cells"][1]["text"] == "Premiere"
    assert not (set(repaired_query["source_column_indices"]) & set(repaired_target["source_column_indices"]))
    assert backups
