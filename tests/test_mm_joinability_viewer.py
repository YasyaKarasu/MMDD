import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

from mm_joinability_dataset_viewer import (
    DEFAULT_INDEX_FILENAME,
    ViewerDataset,
    create_app,
    load_pairs,
    stream_table_preview_record,
)
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
    assert pairs[0]["row_view_index"] == 0
    assert pairs[0]["row_view_kind"] == "canonical"
    assert pairs[0]["row_view_count"] == 1
    assert pairs[0]["query_highlight_rows"] == [0]
    assert pairs[0]["target_highlight_rows"] == [0]
    assert pairs[0]["paths"][0]["asset_content"] == "Argentina is visible in the referenced text chunk."

    limited_dataset = ViewerDataset(
        tmp_path,
        max_rows=1,
        max_paths=5,
        max_asset_chars=2000,
        index_path=tmp_path / "limited-viewer.sqlite3",
    )
    pair_key = limited_dataset.pair_keys_for_query("query_q")[0]
    limited_pair = limited_dataset.hydrate_pair(pair_key)
    full_query, full_target = limited_dataset.hydrate_full_pair_tables(pair_key)
    assert len(limited_pair["query_table"]["rows"]) == 1
    assert len(limited_pair["target_table"]["rows"]) == 1
    assert len(full_query["rows"]) == 2
    assert len(full_target["rows"]) == 2

    client = create_app(tmp_path, max_rows=5, max_paths=5, max_asset_chars=2000).test_client()
    response = client.get("/?q=Argentina")
    assert response.status_code == 200
    assert b"MM Joinability Dataset Viewer" in response.data
    assert b"query_q" in response.data
    assert b"target_t" in response.data
    assert b"Argentina is visible in the referenced text chunk." in response.data
    assert b"canonical" in response.data
    assert (tmp_path / DEFAULT_INDEX_FILENAME).exists()


def test_viewer_groups_and_filters_disjoint_train_row_views(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in (
        "query_tables",
        "data_lake_tables",
        "bridge_assets",
        "evidence_recoveries",
    ):
        (tmp_path / name).mkdir()
    write_manifest(tmp_path)
    (tmp_path / "stats.json").write_text(
        json.dumps({"query_tables": 2, "qrels": 2}), encoding="utf-8"
    )
    write_jsonl(
        tmp_path / "query_tables" / "part-00000.jsonl",
        [
            table_record(
                "query_z_canonical",
                "query",
                ["Entity", "Club"],
                [["Canonical Entity", "CanonicalNeedle"]],
            ),
            table_record(
                "query_a_augmented",
                "query",
                ["Entity", "Club"],
                [["Augmented Entity", "AugmentedNeedle"]],
            ),
        ],
    )
    unrelated_alias = {
        "table_id": "dl_raw_unreferenced",
        "role": "raw_data_lake_table",
        "source_table_id": "source_unreferenced",
        "source_table_ref": {
            "artifact": "source_tables",
            "source_table_id": "source_unreferenced",
        },
    }
    write_jsonl(
        tmp_path / "data_lake_tables" / "part-00000.jsonl",
        [
            unrelated_alias,
            table_record(
                "target_shared",
                "target_data_lake_table",
                ["Country"],
                [["Argentina"], ["France"]],
            ),
        ],
    )
    write_jsonl(
        tmp_path / "bridge_assets" / "part-00000.jsonl",
        [
            {
                "asset_id": "asset_shared",
                "asset_type": "text",
                "title": "Shared evidence",
                "content": "Evidence shared by both row views.",
            }
        ],
    )
    write_jsonl(
        tmp_path / "qrels.jsonl",
        [
            {
                "query_table_id": "query_a_augmented",
                "target_table_id": "target_shared",
                "rel": 3,
                "split": "train",
                "chain_id": "chain_shared",
                "row_view_index": 1,
                "source_table_id": "source_1",
                "join_attribute": {"column_name": "Country"},
                "reason": "explicit_visible_join_column",
            },
            {
                "query_table_id": "query_z_canonical",
                "target_table_id": "target_shared",
                "rel": 3,
                "split": "train",
                "chain_id": "chain_shared",
                "row_view_index": 0,
                "source_table_id": "source_1",
                "join_attribute": {"column_name": "Country"},
                "reason": "model_recoverable_join_column",
            },
        ],
    )
    write_jsonl(
        tmp_path / "evidence_recoveries" / "part-00000.jsonl",
        [
            {
                "recovery_id": f"rec_{view}",
                "path_id": f"path_{view}",
                "query_table_id": query_id,
                "target_table_id": "target_shared",
                "query_row_id": 0,
                "target_row_ids": [view],
                "source_table_id": "source_1",
                "source_row_id": view,
                "split": "train",
                "query_entity": {"cell_text": f"Entity {view}"},
                "recovered_attribute": {
                    "column_name": "Country",
                    "value": "Argentina" if view == 0 else "France",
                },
                "evidence": {
                    "asset_id": "asset_shared",
                    "asset_type": "text",
                },
            }
            for view, query_id in enumerate(
                ["query_z_canonical", "query_a_augmented"]
            )
        ],
    )

    pairs, _stats, _assets = load_pairs(
        tmp_path, max_rows=5, max_paths=5, max_asset_chars=2000
    )

    assert [pair["query_table_id"] for pair in pairs] == [
        "query_z_canonical",
        "query_a_augmented",
    ]
    assert [pair["row_view_index"] for pair in pairs] == [0, 1]
    assert {pair["row_view_count"] for pair in pairs} == {2}

    app = create_app(
        tmp_path, max_rows=5, max_paths=5, max_asset_chars=2000
    )
    client = app.test_client()
    canonical = client.get("/?row_view=canonical")
    augmented = client.get("/?row_view=augmented")
    searched = client.get("/?q=AugmentedNeedle")
    positive = client.get("/?positive_only=1")

    assert canonical.status_code == 200
    assert b"query_z_canonical" in canonical.data
    assert b"query_a_augmented" not in canonical.data
    assert b"canonical" in canonical.data
    assert b"1 of 2" in canonical.data
    assert b"query_a_augmented" in augmented.data
    assert b"query_z_canonical" not in augmented.data
    assert b"augmented" in augmented.data
    assert b"2 of 2" in augmented.data
    assert b"query_a_augmented" in searched.data
    assert b"Multi-view chains" in searched.data
    assert positive.status_code == 200
    assert b"query_z_canonical" in positive.data
    assert b"query_a_augmented" not in positive.data
    assert b"recoverable pairs only" in positive.data

    def fail_rebuild(*_args, **_kwargs):
        raise AssertionError("current viewer index should be reused")

    monkeypatch.setattr(ViewerDataset, "_build_index", fail_rebuild)
    reused_pairs, _stats, _assets = load_pairs(
        tmp_path, max_rows=5, max_paths=5, max_asset_chars=2000
    )
    assert len(reused_pairs) == 2


def test_streaming_table_preview_keeps_first_and_highlighted_rows() -> None:
    record = table_record(
        "target_large",
        "target_data_lake_table",
        ["Country"],
        [[f"value {row}"] for row in range(20)],
    )
    handle = io.BytesIO(json.dumps(record).encode("utf-8"))

    preview = stream_table_preview_record(
        handle,
        max_rows=2,
        include_rows={17},
    )

    assert preview["table_id"] == "target_large"
    assert [row["row_id"] for row in preview["rows"]] == [0, 1, 17]
    assert preview["_viewer_truncated"] is True
