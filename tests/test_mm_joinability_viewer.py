import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mm_joinability_dataset_viewer import create_app, load_pairs
from stage1_io import write_jsonl


def write_manifest(root: Path) -> None:
    manifest = {
        "format": "sharded_jsonl",
        "artifacts": {
            "query_tables": {"shards": [{"path": "query_tables/part-00000.jsonl", "records": 1}]},
            "data_lake_tables": {"shards": [{"path": "data_lake_tables/part-00000.jsonl", "records": 1}]},
            "bridge_assets": {"shards": [{"path": "bridge_assets/part-00000.jsonl", "records": 1}]},
            "evidence_recoveries": {"shards": [{"path": "evidence_recoveries/part-00000.jsonl", "records": 1}]},
        },
        "single_files": {"qrels": "qrels.jsonl", "stats": "stats.json"},
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def table_record(table_id: str, role: str, columns: list[str], rows: list[list[str]]) -> dict:
    return {
        "table_id": table_id,
        "role": role,
        "split": "train",
        "source_table_id": "source_1",
        "page_title": "Synthetic Page",
        "caption": f"{role} caption",
        "columns": [{"column_index": idx, "column_name": name} for idx, name in enumerate(columns)],
        "rows": [
            {
                "row_id": row_idx,
                "source_row_id": row_idx,
                "cells": [
                    {"column_index": col_idx, "column_name": columns[col_idx], "text": value}
                    for col_idx, value in enumerate(values)
                ],
            }
            for row_idx, values in enumerate(rows)
        ],
    }


def test_mm_joinability_viewer_loads_query_target_paths(tmp_path):
    for name in ("query_tables", "data_lake_tables", "bridge_assets", "evidence_recoveries"):
        (tmp_path / name).mkdir()
    write_manifest(tmp_path)
    (tmp_path / "stats.json").write_text(json.dumps({"query_tables": 1, "evidence_recoveries": 1}), encoding="utf-8")
    write_jsonl(
        tmp_path / "query_tables" / "part-00000.jsonl",
        [table_record("query_q", "query", ["Entity", "Club"], [["Alpha", "A FC"], ["Beta", "B FC"]])],
    )
    write_jsonl(
        tmp_path / "data_lake_tables" / "part-00000.jsonl",
        [table_record("target_t", "target_data_lake_table", ["Country", "Stadium"], [["Argentina", "North"], ["France", "West"]])],
    )
    write_jsonl(
        tmp_path / "bridge_assets" / "part-00000.jsonl",
        [
            {
                "asset_id": "asset_alpha",
                "asset_type": "text",
                "entity_wiki_title": "Alpha",
                "content": "Argentina is visible in the referenced text chunk.",
                "source": "wikipedia_extract_chunk",
                "text_chunk_index": 0,
                "text_chunk_count": 1,
            }
        ],
    )
    write_jsonl(
        tmp_path / "qrels.jsonl",
        [
            {
                "query_table_id": "query_q",
                "target_table_id": "target_t",
                "data_lake_table_id": "target_t",
                "rel": 3,
                "split": "train",
                "chain_id": "chain_1",
                "source_table_id": "source_1",
                "join_attribute": {"column_name": "Country"},
                "reason": "model_recoverable_join_column",
            }
        ],
    )
    write_jsonl(
        tmp_path / "evidence_recoveries" / "part-00000.jsonl",
        [
            {
                "recovery_id": "rec_1",
                "path_id": "path_1",
                "query_table_id": "query_q",
                "target_table_id": "target_t",
                "query_row_id": 0,
                "target_row_ids": [0],
                "source_table_id": "source_1",
                "source_row_id": 0,
                "split": "train",
                "query_entity": {"entity_id": "entity_alpha", "wiki_title": "Alpha", "cell_text": "Alpha"},
                "recovered_attribute": {
                    "column_name": "Country",
                    "value": "Argentina",
                    "model_value": "Argentina",
                },
                "evidence": {
                    "asset_id": "asset_alpha",
                    "asset_type": "text",
                    "title": "Alpha",
                    "content_snippet": "shorter snippet",
                    "model_evidence": "Argentina",
                },
            }
        ],
    )

    pairs, stats, _assets = load_pairs(tmp_path, max_rows=5, max_paths=5, max_asset_chars=2000)

    assert stats["query_tables"] == 1
    assert len(pairs) == 1
    assert pairs[0]["query_table_id"] == "query_q"
    assert pairs[0]["target_table_id"] == "target_t"
    assert pairs[0]["query_highlight_rows"] == [0]
    assert pairs[0]["target_highlight_rows"] == [0]
    assert pairs[0]["paths"][0]["asset_content"] == "Argentina is visible in the referenced text chunk."

    client = create_app(tmp_path, max_rows=5, max_paths=5, max_asset_chars=2000).test_client()
    response = client.get("/?q=Argentina")
    assert response.status_code == 200
    assert b"MM Joinability Dataset Viewer" in response.data
    assert b"query_q" in response.data
    assert b"target_t" in response.data
    assert b"Argentina is visible in the referenced text chunk." in response.data
